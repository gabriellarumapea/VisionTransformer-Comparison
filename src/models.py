import timm
import torch.nn as nn

def create_model(model_name, num_classes, pretrained=True):
    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = total * 4 / (1024 * 1024)
    return {
        'Total Parameters': total,
        'Trainable Parameters': trainable,
        'Model Size (MB)': size_mb
    }
