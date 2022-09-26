import os
import shutil
from os import path
from typing import NamedTuple, Optional, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from hooloovoo.deeplearning.networks.controls import Controls
from hooloovoo.deeplearning.training.eval import plot_train_example, EvalLoop, tb_log_eval
from hooloovoo.deeplearning.training.introspection import print_param_overview
from hooloovoo.deeplearning.training.train_loop import TrainLoop, Event
from hooloovoo.utils.arbitrary import HW
from hooloovoo.utils.tensortools import interpolate_tensor, prepare_class_weights
from hooloovoo.utils.timer import hms, throttle


class DisplayIntervals(NamedTuple):
    console: str
    gui: Optional[str] = None


class LoggingIntervals(NamedTuple):
    stats: str
    example: str
    model: Tuple[str, str]


class LoggingSettings:
    def __init__(self, log_dir: str, resume_from: Optional[str],
                 display_intervals: DisplayIntervals, logging_intervals: LoggingIntervals):
        self.tensorboard_dir = path.join(log_dir, "tensorboard")
        self.checkpoint_dir = path.join(log_dir, "checkpoints")
        self.resume_from = resume_from
        self.display_intervals = display_intervals
        self.logging_intervals = logging_intervals

        self.tb_writer = SummaryWriter(log_dir=self.tensorboard_dir)


class GradientSettings(NamedTuple):
    class_weights: List
    lr: float
    momentum: float
    patience: int


class EvalSettings(NamedTuple):
    eval_test: Dataset
    eval_train: Dataset
    max_size: HW


class TrainAndLog(TrainLoop):
    def __init__(self, model: Controls, dataloader: DataLoader, device: torch.device,
                 logging_settings: LoggingSettings,
                 gradient_settings: GradientSettings,
                 eval_settings: EvalSettings):
        self.logging_settings = logging_settings
        self.gradient_settings = gradient_settings
        self.eval_settings = eval_settings

        class_weights = prepare_class_weights(self.gradient_settings.class_weights, device=device)
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
        self.pixelwise_loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights, reduction='none')
        self.device = device
        model.to(device)

        optimizer = SGD(model.parameters(),
                        lr=self.gradient_settings.lr,
                        momentum=self.gradient_settings.momentum)
        scheduler = ReduceLROnPlateau(optimizer, verbose=True, patience=self.gradient_settings.patience)
        super().__init__(model, dataloader, optimizer, scheduler)

    def run(self) -> None:

        self.add_default_handlers()
        self.add_handler_display_console(every=self.logging_settings.display_intervals.console)
        self.add_handler(Event.TRAIN_STEP_END, self.display_gui())
        self.add_handler(Event.TRAIN_STEP_END, self.log_stats())
        self.add_handler(Event.TRAIN_STEP_END, self.log_example())
        self.add_handler(Event.TRAIN_STEP_END, self.log_model())
        self.add_handler(Event.TRAINING_ABORTED, self.save_checkpoint)
        self.add_handler(Event.TRAINING_RESUMED, self.load_checkpoint, pos=0)

        # remove log dir unless resuming from earlier
        if path.isdir(self.logging_settings.tensorboard_dir) and self.logging_settings.resume_from is None:
            shutil.rmtree(self.logging_settings.tensorboard_dir, ignore_errors=True)

        print_param_overview(self.model, self.optimizer, width=120)
        self.train(self.logging_settings.resume_from)

    def forward_pass(self, example):
        x, _ = example
        return self.model(x.to(self.device))

    def compute_loss(self, example, yhat):
        _, y = example
        yhat = interpolate_tensor(yhat, y.shape)
        return self.loss_fn(yhat, y.to(self.device))

    def step_scheduler(self, **_kwargs):
        self.scheduler.step(self.state.mean_epoch_loss, epoch=self.state.epoch)

    def display_gui(self):
        if self.logging_settings.display_intervals.gui is not None:
            try:
                matplotlib.use('Qt5Agg')
            except ImportError:
                print("Qt5Agg backend not available")

            @throttle(self.logging_settings.display_intervals.gui)
            def display_gui_(example, yhat, **_kwargs):
                yhat = interpolate_tensor(yhat, example[1].shape)
                per_pixel_loss = self.pixelwise_loss_fn(yhat, example[1].to(self.device))
                plot_train_example(example, yhat, per_pixel_loss, num=1)
                plt.pause(0.01)
        else:
            def display_gui_(**_kwargs): pass
        return display_gui_

    @property
    def sw(self):
        """step and walltime"""
        return dict(global_step=self.state.n_examples, walltime=self.state.timer.duration)

    def log_stats(self):
        @throttle(self.logging_settings.logging_intervals.stats)
        def log_stats_(**_kwargs):
            sw = self.sw
            self.logging_settings.tb_writer.add_scalar(tag="training/mean_epoch_loss",
                                                       scalar_value=self.state.mean_epoch_loss, **sw)
            for i, pg in enumerate(self.optimizer.param_groups):
                self.logging_settings.tb_writer.add_scalar(tag="training/lr/param_group_" + str(i),
                                                           scalar_value=pg['lr'], **sw)

            self.logging_settings.tb_writer.close()
        return log_stats_

    def log_example(self):
        @throttle(self.logging_settings.logging_intervals.example)
        def log_example_(example, yhat, **_kwargs):
            yhat = interpolate_tensor(yhat, example[1].shape)
            per_pixel_loss = self.pixelwise_loss_fn(yhat, example[1].to(self.device))
            fig: Figure = plot_train_example(example, yhat, per_pixel_loss)
            self.logging_settings.tb_writer.add_figure(tag="train/example", figure=fig, **self.sw)
            self.logging_settings.tb_writer.close()
        return log_example_

    def log_model(self):
        """
        Evaluates the model and saves the model parameters every time:
            * the loss reaches a new minimum and the minimum amount of time has passed since the last save
            * the maximum amount of time has passed since the last save without reaching a new loss minimum

        Each log is accompanied by a model evaluation, in order to be able to pick a good model for production.
        """
        tmin, tmax = self.logging_settings.logging_intervals.model

        min_loss_decreased = self.track(loss=np.inf)(condition=lambda old, new: new < old,
                                                     new=lambda: self.state.mean_epoch_loss,
                                                     auto_update=False)
        min_time_passed = self.track(min_time=hms("0s"))(condition=lambda old, new: new - old > hms(tmin),
                                                         new=lambda: self.state.timer.duration,
                                                         auto_update=False)
        max_time_passed = self.track(max_time=hms("0s"))(condition=lambda old, new: new - old > hms(tmax),
                                                         new=lambda: self.state.timer.duration,
                                                         auto_update=False)

        should_log = (min_loss_decreased & min_time_passed) | max_time_passed

        def log_model_(**_kwargs):
            if should_log:
                # evaluate the model
                print("running model evaluation")
                n = len(self.eval_settings.eval_test)
                n_plot = min(n, 12)
                eval_loop = EvalLoop(self.model, self.device,
                                     infer_kwargs=dict(inference_device=self.device,
                                                       max_size=self.eval_settings.max_size))
                evaluate = lambda ds: eval_loop.evaluate(ds, n, n_plot)
                loss_medians, loss_means, iou_medians, iou_means = tb_log_eval(
                    results={"test": evaluate(self.eval_settings.eval_test),
                             "train": evaluate(self.eval_settings.eval_train)},
                    verbose=True,
                    writer=self.logging_settings.tb_writer, **self.sw
                )
                print("median iou: {}, mean iou: {}".format(str(iou_medians), str(iou_means)))
                print("median loss: {}, mean loss: {}".format(str(loss_medians), str(loss_means)))
                self.model.enable_training()

                # save the model
                should_log.update_all()  # to get the most recent time into the saved model
                self.save_checkpoint()
                should_log.update_all()  # the timers must take into account time taken for saving the model
        return log_model_

    def save_checkpoint(self, **_kwargs):
        if not path.exists(self.logging_settings.checkpoint_dir):
            os.mkdir(self.logging_settings.checkpoint_dir)
        cp_path_0 = path.join(self.logging_settings.checkpoint_dir, "example_{:06d}"
                              .format(self.state.n_examples))
        cp_path_1 = path.join(self.logging_settings.checkpoint_dir, "last")
        print("saving to checkpoint: " + cp_path_0)
        self.save(cp_path_0)
        self.save(cp_path_1)

    def load_checkpoint(self, resume_from, **_kwargs):
        cp_path = path.join(self.logging_settings.checkpoint_dir, resume_from)
        print("resuming from checkpoint: " + cp_path)
        self.load(cp_path, map_location=self.device)
