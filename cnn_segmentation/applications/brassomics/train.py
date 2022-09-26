import os
import re
from os import path
from typing import List, Iterator, Tuple, NamedTuple, Optional

import numpy as np
import torch
from PIL import Image
from torch.nn import ZeroPad2d
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.transforms import functional as f

from hooloovoo.deeplearning.networks import densenet as dn
from hooloovoo.deeplearning.training.augmentation import random_color_jitter_fn, \
    random_hflip_and_small_rot_fn
from hooloovoo.deeplearning.training.preprocessing import equal_spread_crop_boxes, split_random, TrainEval, split_images
from hooloovoo.deeplearning.utils import attempt_get_cuda_device
from hooloovoo.utils.arbitrary import hex_of_hash, HW, np_seed, seed_from
from hooloovoo.utils.functional import both, mapl, maybe
from hooloovoo.utils.imagetools import has_foreground, background_difficulty, is_image_filename
from hooloovoo.utils.ostools import makedir, touchfile, require_directory
from pippa_cnn_segmentation.libs.train import TrainAndLog, LoggingSettings, DisplayIntervals, LoggingIntervals, \
    GradientSettings, EvalSettings


class Paths(NamedTuple):
    x: str
    y: str
    w: Optional[str] = None


def find_xy_paths(x_directory: str, y_directory) -> Iterator[Paths]:
    """
    Given a `x_directory` with images, searches for the corresponding labelled `.tiff` image in `y_directory`.
    There may be sub-folders with images inside `x_directory`, their corresponding image in `y_directory` will be found
    if it has the same folder substructure as `x_directory`.
    """
    require_directory(x_directory)
    require_directory(y_directory)
    for p, dirs, filenames in os.walk(x_directory):
        reldir = os.path.relpath(p, x_directory)
        for filename in filenames:
            if is_image_filename(filename):
                relpath_x = os.path.join(reldir, filename)
                noext, _ = os.path.splitext(filename)
                relpath_y = os.path.join(reldir, noext + ".tiff")

                abspath_x = os.path.join(x_directory, relpath_x)
                abspath_y = os.path.join(y_directory, relpath_y)

                assert os.path.isfile(abspath_x)
                if not os.path.isfile(abspath_y):
                    print(f"Warning: .tiff file for '{abspath_x}' missing ('{abspath_y}' does not exist)")
                else:
                    yield Paths(abspath_x, abspath_y)


def preproc(x_dir, paths: Iterator[Paths], cache_dir: str,
            cropped_image_size: HW, bidir_overlap: int, min_fg: float) -> Tuple[List[Paths], List[float]]:
    """
    Cuts all images in x_dir into smaller pieces, removing pieces without foreground.
    The remaining pieces are stored in the cache directory.
    Each piece gets a weight depending on how similar the background colors are to foreground colors.

    This does not re-processes already processed images.

    :returns: two lists, one list contains file paths, one for each piece.
        The other list contains the weights, one scalar value for each piece.
    """
    is_done = lambda out_dir: path.exists(path.join(out_dir, ".done"))
    mark_done = lambda out_dir: touchfile(path.join(out_dir, ".done"), exists_ok=True)

    piece_dirs = []
    weights = []
    for xy_path in paths:
        relpath, _ext = os.path.splitext(os.path.relpath(xy_path.x, x_dir))
        image_dir = makedir(cache_dir, relpath, exists_ok=True, recursive=True)
        if not is_done(image_dir):
            print(f"Processing image: {relpath} -> {image_dir}")
            x, y = load_pair(xy_path)
            splits = split_images(x, y,
                                  split_fn=equal_spread_crop_boxes,
                                  cropped_image_size=cropped_image_size,
                                  bidir_overlap=bidir_overlap)
            for i, split in enumerate(splits):
                piece_dir = path.join(image_dir, str(i))
                if has_foreground(split[1], frac=min_fg):
                    w = background_difficulty(*split)
                    save_pair(piece_dir, split, w)
                    piece_dirs.append(piece_dir)
                    weights.append(w)
            mark_done(image_dir)
        else:
            for piece_dirname in os.listdir(image_dir):
                if re.match(r"^\d+$", piece_dirname):
                    piece_dir = path.join(image_dir, piece_dirname)
                    w = load_weight(piece_dir)
                    piece_dirs.append(piece_dir)
                    weights.append(w)

    return mapl(preproc_paths, piece_dirs), weights


def load_pair(xy: Paths):
    input_image = Image.open(xy.x).convert("RGB")
    target_image = Image.open(xy.y).convert("L")

    if "_T0-" not in xy.x:
        w, h = input_image.size
        if h < w:
            input_image = input_image.transpose(Image.ROTATE_270)

    return input_image, target_image


def preproc_paths(directory: str) -> Paths:
    return Paths(path.join(directory, "x.png"),
                 path.join(directory, "y.png"),
                 path.join(directory, "weight.txt"))


def save_pair(directory, xy, w):
    x, y = xy
    makedir(directory, exists_ok=True, recursive=False)
    px, py, pw = preproc_paths(directory)
    x.save(px)
    y.save(py)
    with open(pw, "w") as fh:
        fh.writelines(str(w))


def load_weight(directory) -> float:
    _x, _y, pw = preproc_paths(directory)
    with open(pw, "r") as fh:
        return float(fh.read().strip())


class ImageData(Dataset):
    def __init__(self, examples: List[Paths], padding: int,
                 augment=False, rotation=5, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.examples = examples
        self.padding = ZeroPad2d(padding)
        self.augment = augment
        self.rotation = rotation
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __getitem__(self, index):
        inp_img, tgt_img = load_pair(self.examples[index])

        if self.augment:
            col = random_color_jitter_fn(brightness=self.brightness, contrast=self.contrast,
                                         saturation=self.saturation, hue=self.hue)
            flp = random_hflip_and_small_rot_fn(self.rotation, expand=False)

            inp_img = col(inp_img)
            inp_img, tgt_img = map(flp, (inp_img, tgt_img))

        inp_tensor = f.to_tensor(inp_img)
        tgt_tensor = torch.from_numpy((np.array(tgt_img) > (255 / 2)).astype(np.int64))
        inp_tensor, tgt_tensor = both(self.padding)((inp_tensor, tgt_tensor))
        assert inp_tensor.dtype == torch.float32
        assert tgt_tensor.dtype == torch.int64

        # from hooloovoo.utils.plotter import img_show
        # img_show(inp_tensor, tgt_tensor, wait=True)
        return inp_tensor, tgt_tensor

    def __len__(self):
        return len(self.examples)


class Train(TrainAndLog):
    def __init__(self, settings):

        # Finding location train/eval/test data (boring)
        # ----------------------------------------------

        all_paths: List[Tuple[str, Paths]] = list(find_xy_paths(*settings.training.data))
        with np_seed(seed_from("Slartibartfast")):
            example_splits: TrainEval = split_random(all_paths, size_test=settings.training.params.test_size)
        eval_test_data = ImageData(example_splits.eval.test, padding=settings.training.params.padding)
        eval_train_data = ImageData(example_splits.eval.train, padding=settings.training.params.padding)

        # Preproc for training images, a.k.a. cut them in pieces (also boring)
        # --------------------------------------------------------------------
        cache_dir = makedir(settings.training.preprocess.cache_dir,
                            hex_of_hash((settings.training.preprocess, len(example_splits.train))), exists_ok=True)
        example_paths, example_weights = preproc(
            settings.training.data.x, example_splits.train, cache_dir,
            cropped_image_size=HW(**settings.training.preprocess.max_size),
            bidir_overlap=settings.training.preprocess.bidir_overlap,
            min_fg=settings.training.preprocess.min_foreground_fraction
        )
        train_data = ImageData(example_paths, padding=settings.training.params.padding, augment=True,
                               **settings.training.params.augment)

        if settings.training.data_loader.weighted_sampler:
            if settings.training.data_loader.shuffle:
                sampler_weights = [w + settings.training.data_loader.minimal_weight for w in example_weights]
                kwargs = {"sampler": WeightedRandomSampler(weights=sampler_weights, num_samples=len(example_paths))}
            else:
                raise Exception("Settings shuffle to false is not compatible with weighted random sampling")
        else:
            if settings.training.data_loader.shuffle:
                kwargs = {'shuffle': True}
            else:
                kwargs = {'shuffle': False}

        # Apply all remaining settings (= more boring code)
        # -------------------------------------------------

        super(Train, self).__init__(
            model=dn.DenseNet.from_settings(settings.model.DenseNet),
            dataloader=DataLoader(train_data, batch_size=settings.training.data_loader.batch_size,
                                  num_workers=settings.training.data_loader.num_workers,
                                  **kwargs),
            device=maybe(torch.device, settings.device, if_missing=attempt_get_cuda_device()),
            logging_settings=LoggingSettings(
                log_dir=settings.training.logging.dir,
                resume_from=settings.training.logging.resume_from,
                display_intervals=DisplayIntervals(**settings.training.display.intervals),
                logging_intervals=LoggingIntervals(
                    stats=settings.training.logging.intervals.stats,
                    example=settings.training.logging.intervals.train_example,
                    model=(settings.training.logging.intervals.model.min,
                           settings.training.logging.intervals.model.min)
                )
            ),
            gradient_settings=GradientSettings(
                class_weights=settings.training.params.class_weights,
                **settings.training.params.optimizer
            ),
            eval_settings=EvalSettings(
                eval_test=eval_test_data,
                eval_train=eval_train_data,
                max_size=HW(**settings.training.preprocess.max_size)
            )
        )
