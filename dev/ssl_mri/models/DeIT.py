import timm
import torch.nn as nn

def deit_tiny_patch16_224(pretrained=False, **kwargs):
    """DeiT-tiny model @ 224x224 from paper (https://arxiv.org/abs/2012.12877)."""
    model = timm.create_model('deit_tiny_patch16_224', pretrained=pretrained, **kwargs)
    return model

def deit_small_patch16_224(pretrained=False, **kwargs):
    """DeiT-small model @ 224x224 from paper (https://arxiv.org/abs/2012.12877)."""
    model = timm.create_model('deit_small_patch16_224', pretrained=pretrained, **kwargs)
    return model

def deit_base_patch16_224(pretrained=False, **kwargs):
    """DeiT-base model @ 224x224 from paper (https://arxiv.org/abs/2012.12877)."""
    model = timm.create_model('deit_base_patch16_224', pretrained=pretrained, **kwargs)
    return model

def deit_tiny_distilled_patch16_224(pretrained=False, **kwargs):
    """DeiT-tiny distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877)."""
    model = timm.create_model('deit_tiny_distilled_patch16_224', pretrained=pretrained, **kwargs)
    return model

def deit_small_distilled_patch16_224(pretrained=False, **kwargs):
    """DeiT-small distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877)."""
    model = timm.create_model('deit_small_distilled_patch16_224', pretrained=pretrained, **kwargs)
    return model

def deit_base_distilled_patch16_224(pretrained=False, **kwargs):
    """DeiT-base distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877)."""
    model = timm.create_model('deit_base_distilled_patch16_224', pretrained=pretrained, **kwargs)
    return model