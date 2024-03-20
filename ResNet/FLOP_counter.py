import torch
import torch.nn as  nn

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