import argparse
import sys

import yaml

from configs import parser as _parser

args = None


def parse_arguments():
    parser = argparse.ArgumentParser(description="MindSpore SwinTransformer Training")

    parser.add_argument(
        "-a", "--arch", metavar="ARCH", default="ResNet18", help="model architecture"
    )
    parser.add_argument(
        "--ape", default=False, type=bool, help="absolute position embedding"
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=64,
        type=int,
        metavar="N",
        help="mini-batch size (default: 256), this is the total "
             "batch size of all GPUs on the current node when "
             "using Data Parallel or Distributed Data Parallel",
    )
    parser.add_argument(
        "--beta",
        default=[0.9, 0.999],
        type=lambda x: [float(a) for a in x.split(",")],
        help="beta for optimizer",
    )
    parser.add_argument(
        "--config", help="Config file to use (see configs dir)", default=None
    )
    parser.add_argument(
        "--clip-value", default=5., type=float, help="Clip grad value"
    )
    parser.add_argument(
        "--crop-size", default=320, type=int, help="crop-size"
    )
    # General Config
    parser.add_argument(
        "--data", help="path to dataset base directory", default="/mnt/disk1/datasets"
    )
    parser.add_argument(
        "--device-id", default=0, type=int, help="Device Id"
    )
    parser.add_argument(
        "--device-target", help="Device target to use", default="GPU",
        choices=["GPU", "Ascend", "CPU"], type=str
    )
    parser.add_argument(
        "--epochs",
        default=300,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--eps", default=1e-8, type=float
    )
    parser.add_argument(
        "--in-channel", default=3, type=int
    )
    parser.add_argument(
        "--keep-checkpoint-max",
        default=20,
        type=int,
        help="keep checkpoint max num",
    )
    parser.add_argument("--optimizer", help="Which optimizer to use", default="sgd")
    parser.add_argument("--set", help="name of dataset", type=str, default="ImageNet")
    parser.add_argument(
        "--graph-mode", default=0, type=int, help="graph mode with 0, python with 1"
    )
    parser.add_argument(
        "-j",
        "--num-parallel-workers",
        default=20,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 20)",
    )
    parser.add_argument(
        "--start-epoch",
        default=None,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "--warmup_length", default=0, type=int, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--warmup-lr", default=5e-7, type=float, help="warm up learning rate"
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=0.05,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument("--num-classes", default=3, type=int)
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        default=None,
        type=str,
        help="use pre-trained model",
    )
    parser.add_argument(
        "--seed", default=None, type=int, help="seed for initializing training. "
    )
    parser.add_argument(
        "--is-dynamic-loss-scale", default=1, type=int, help="is_dynamic_loss_scale  "
    )
    parser.add_argument(
        "--loss-type", default="dice", type=str, help="loss type"
    )
    # Learning Rate Policy Specific
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=5e-4,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--loss-scale",
        default=1024,
        type=int,
        help="loss scale",
    )
    parser.add_argument(
        "--lr-scheduler", default="cosine_annealing", help="Schedule for the learning rate."
    )
    parser.add_argument(
        "--lr-adjust", default=30, type=float, help="Interval to drop lr"
    )
    parser.add_argument(
        "--lr-gamma", default=0.3, type=int, help="Multistep multiplier"
    )
    parser.add_argument(
        "--name", default=None, type=str, help="Experiment name to append to filepath"
    )
    parser.add_argument(
        "--nesterov",
        default=False,
        action="store_true",
        help="Whether or not to use nesterov for SGD",
    )
    parser.add_argument(
        "--no-bn-decay", action="store_true", default=False, help="No batchnorm decay"
    )
    parser.add_argument("--norm-type", default=None, help="Norm type")
    args = parser.parse_args()

    # Allow for use from notebook without config file
    if len(sys.argv) > 1:
        get_config(args)

    return args


def get_config(args):
    # get commands from command line
    override_args = _parser.argv_to_vars(sys.argv)

    # load yaml file
    yaml_txt = open(args.config).read()

    # override args
    loaded_yaml = yaml.load(yaml_txt, Loader=yaml.FullLoader)
    for v in override_args:
        loaded_yaml[v] = getattr(args, v)

    print(f"=> Reading YAML config from {args.config}")
    args.__dict__.update(loaded_yaml)
    print(args)


def run_args():
    global args
    if args is None:
        args = parse_arguments()


run_args()
