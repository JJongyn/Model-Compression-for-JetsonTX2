import os
import yaml

import torch
import torch.nn as  nn

from models import *

CONFIG_PATH = "./config/"


def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        print(file)
        config = yaml.safe_load(file)

    return config

def load_model(cfg, device):
    if cfg['data'] == 'cifar10':
        class_num = 10

    # cfg['model'] == reset50
    model = ResNet50(class_num).to(device)
    return model

def get_optimizer(cfg, model):
    if cfg['opt_type'] == "SGD":
        optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=cfg['lr'],
        )
    elif cfg['opt_type'] == "RMSProp":
        optimizer = torch.optim.RMSprop(
        params=model.parameters(),
        lr=fcg['lr'],
        eps=1.0,
        )
    elif cfg['opt_type'] == "Adam":
        optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=cfg['lr'],
        eps=1e-5
        )

    return optimizer


class FLOP_counter:
    def __init__(self, model, input_size):
        total_flops = 0
    
    # 모델을 CPU로 설정 (필요한 경우 GPU 설정으로 변경)
        model = model.cpu()
        
        # 모델의 레이어별로 순회하며 FLOPs 계산
        for layer in model.modules():
            if isinstance(layer, nn.Conv2d):
                # Conv2D 레이어의 FLOPs 계산
                output_dims = layer(torch.rand(1, *input_size)).detach().size()
                flops = (2 * layer.in_channels * layer.out_channels * output_dims[2] * output_dims[3] *
                        layer.kernel_size[0] * layer.kernel_size[1] /
                        layer.groups)
                total_flops += flops
            elif isinstance(layer, nn.Linear):
                # Linear 레이어의 FLOPs 계산
                flops = 2 * layer.in_features * layer.out_features
                total_flops += flops
                
        return total_flops