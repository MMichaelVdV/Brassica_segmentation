import argparse
import importlib
import textwrap

from hooloovoo.deeplearning.utils import Mode
from hooloovoo.utils.settings import load_settings_file

_apps = [" brassomics"]


def build_parser():
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(
            '''cnn segmentation'''
        ),
    )
    parser.add_argument("application", help="which application to run", choices=list(_apps))
    parser.add_argument("settings", help="a settings file in yaml or json format")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    module = importlib.import_module("cnn_segmentation.applications.{}".format(args.application))
    settings = load_settings_file(args.settings)
    print(settings)

    mode = Mode.from_string(settings.mode)
    if mode is Mode.TRAINING:
        module.Train(settings).run()
    if mode is Mode.INFERENCE:
        module.Infer(settings).run()


if __name__ == "__main__":
    main()
