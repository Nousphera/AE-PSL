from enum import Enum


class SupportedModel(Enum):
    VIT_B_32 = "vit_b_32"


def get_hidden_dim(model):
    if model != SupportedModel.VIT_B_32.value:
        raise ValueError(f"Unsupported paper backbone: {model}; use vit_b_32")
    return 768


def get_num_patches(model, global_args=None):
    get_hidden_dim(model)
    return 50
