import json
import os
import sys
from os import path
from typing import Dict, Tuple, Union, Iterator

import pandas as pd
import torch
from PIL import Image
from PIL.Image import Image as PilImage
from hooloovoo.deeplearning.inference.infer_piecewise import infer_image
from hooloovoo.deeplearning.networks import densenet as dn
from hooloovoo.deeplearning.networks.controls import Controls
from hooloovoo.deeplearning.utils import attempt_get_cuda_device, limit_cpu_usage_overkill
from hooloovoo.utils.arbitrary import HW
from hooloovoo.utils.functional import maybe
from hooloovoo.utils.imagetools import image_as_numpy, project_segmentation_contours, image_as_pil, \
    is_image_filename, binarize_mask, project_chull, project_bbox, project_skeleton, remove_small_cc
from hooloovoo.utils.ostools import makedir
from hooloovoo.utils.plotter import img_show
from hooloovoo.utils.tree import Tree


class InferredData:
    def __init__(self, model: Controls, in_image: PilImage,
                 threshold: float, size_threshold: int,
                 max_size: HW, device: torch.device, verbose: True):
        y_tensor = infer_image(model, in_image, max_size=max_size, inference_device=device, verbose=verbose)

        # foreground probability
        out_image = image_as_numpy(y_tensor[1])

        # mask
        binary_mask_all_cc = binarize_mask(out_image, threshold)
        binary_mask = remove_small_cc(binary_mask_all_cc, min_size=size_threshold)

        # visualisation
        viz1, hull_coords = project_chull(image_as_numpy(in_image), binary_mask)
        viz2, bbox = project_bbox(viz1, binary_mask)
        viz3, skeleton_mask = project_skeleton(viz2, binary_mask)
        viz4 = project_segmentation_contours(viz3, binary_mask)

        # numeric properties
        props = dict(
            projected_area=int((binary_mask == 1).sum()),
            bounding_box={k: int(v) for k, v in bbox._asdict().items()} if bbox is not None else None,
            convex_hull_vertices_rowcol=hull_coords,
        )

        self.probability_img: PilImage = image_as_pil(out_image)
        self.mask_img: PilImage = image_as_pil(binary_mask)
        self.skeleton_img: PilImage = image_as_pil(skeleton_mask)
        self.viz_img: PilImage = image_as_pil(viz4)
        self.properties: Dict = props

    def save_images(self, out_path: str):
        makedir(path.dirname(out_path), exists_ok=True, recursive=True)
        name, _ = path.splitext(out_path)
        self.probability_img.save(name + ".probability" + ".png")
        self.mask_img.save(name + ".mask" + ".png")
        self.skeleton_img.save(name + ".skeleton" + ".png")
        self.viz_img.save(name + ".viz" + ".jpg")

    def save_properties(self, out_path: str):
        makedir(path.dirname(out_path), exists_ok=True, recursive=True)
        name, _ = path.splitext(out_path)
        with open(name + ".info.json", mode="w") as fh:
            json.dump(self.properties, fh, indent=2)

    def save(self, out_path: str):
        self.save_images(out_path)
        self.save_properties(out_path)


class Infer:

    def __init__(self, settings):
        self.n_cores = settings.inference.n_cores
        if self.n_cores is not None:
            limit_cpu_usage_overkill(self.n_cores)
        self.device = maybe(torch.device, settings.device, if_missing=attempt_get_cuda_device())
        self.model = dn.DenseNet.from_settings(settings.model.DenseNet).to(self.device)
        self.max_size = HW(**settings.inference.max_size.to_dict())
        self.threshold = settings.inference.postprocess.threshold
        self.size_threshold = settings.inference.postprocess.size_threshold

        model_path = settings.inference.model
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)

        if "data" in settings.inference:
            self.xy_pairs = lambda: self._find_data(settings.inference.data)

    def infer(self, image_path, verbose=True, plot=False) -> InferredData:
        in_image = Image.open(image_path).convert("RGB")
        if "_T0-" not in image_path:
            w, h = in_image.size
            if h < w:
                in_image = in_image.transpose(Image.ROTATE_270)

        inferred_data = InferredData(self.model, in_image,
                                     threshold=self.threshold, size_threshold=self.size_threshold,
                                     max_size=self.max_size, device=self.device, verbose=verbose)

        # show output interactively, useful for debugging
        if plot:
            img_show(in_image, inferred_data.probability_img, inferred_data.mask_img, inferred_data.viz_img,
                     ncol=4, wait=True)
        return inferred_data

    @staticmethod
    def _find_data(data: Union[str, Tree]) -> Iterator[Tuple[str, str]]:
        # Build a list of files to process
        if isinstance(data, str):
            if not path.splitext(data)[1] == ".tsv":
                raise ValueError("xy pairs must be given as tsv files")
            xy_paths: pd.DataFrame = pd.read_csv(data, sep="\t")
            for _index, (x_path, y_path) in xy_paths.iterrows():
                yield x_path, y_path
        else:
            input_dir = data.x
            output_dir = data.y
            for x_dir, subdirs, files in os.walk(input_dir):

                # prevent os.walk from recursing into non-3DVIS directories
                skip = [d for d in subdirs if ("LWIR" in d) or ("SWIR" in d) or ("VNIR" in d)]
                for d in skip:
                    print("skipping directory: " + str(d))
                    subdirs.remove(d)

                rel_path = path.relpath(x_dir, input_dir)
                y_dir = path.join(output_dir, rel_path)
                for filename in files:
                    if is_image_filename(filename):
                        x_path = path.normpath(path.join(x_dir, filename))
                        y_path = path.normpath(path.join(y_dir, filename))
                        yield x_path, y_path

    def run(self):
        print("start")

        # Go over each file and process it
        for x_path, y_path in self.xy_pairs():
            name, _ = path.splitext(y_path)
            if path.isfile(name + ".info.json"):
                print("already done: {}".format(y_path))
            else:
                print("processing: {} -> {}".format(x_path, y_path))
                data = self.infer(x_path, verbose=True, plot=False)
                data.save(y_path)
            # force writing print statements
            sys.stdout.flush()
        print("Finished")
