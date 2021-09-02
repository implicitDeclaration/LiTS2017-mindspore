import os

import yaml

from .swin_transformer import SwinTransformer


def get_swintransformer(args):
    config_path = os.path.join("./configs/swin", f"{args.arch}.yaml")
    yaml_txt = open(config_path).read()
    # override args
    loaded_yaml = yaml.load(yaml_txt, Loader=yaml.FullLoader)
    img_size = loaded_yaml["DATA"]["IMG_SIZE"] if "DATA" in loaded_yaml.keys() else 224
    embed_dim = loaded_yaml["MODEL"]["SWIN"]["EMBED_DIM"]
    depths = loaded_yaml["MODEL"]["SWIN"]["DEPTHS"]
    num_heads = loaded_yaml["MODEL"]["SWIN"]["NUM_HEADS"]
    window_size = loaded_yaml["MODEL"]["SWIN"]["WINDOW_SIZE"]
    drop_path_rate = loaded_yaml["DROP_PATH_RATE"] if "DROP_PATH_RATE" in loaded_yaml.keys() else 0.
    mlp_ratio = args.mlp_ratio
    patch_norm = args.patch_norm
    model = SwinTransformer(
        img_size=img_size,
        embed_dim=embed_dim,
        num_classes=args.num_classes,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        drop_path_rate=drop_path_rate,
        mlp_ratio=mlp_ratio,
        patch_norm=patch_norm,
    )

    return model


def swin_base_patch4_window7_224(args):
    return get_swintransformer(args)


def swin_base_patch4_window12_384(args):
    return get_swintransformer(args)


def swin_large_patch4_window7_224(args):
    return get_swintransformer(args)


def swin_large_patch4_window12_384(args):
    return get_swintransformer(args)


def swin_small_patch4_window7_224(args):
    return get_swintransformer(args)


def swin_tiny_patch4_window7_224(args):
    return get_swintransformer(args)
