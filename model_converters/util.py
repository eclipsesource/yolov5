import torch
import math
from pathlib import Path


def load_torch_model(torch_model_path):
    assert (Path(torch_model_path).is_file()), f'File {torch_model_path} does not exist!'
    torch_model = torch.load(torch_model_path, map_location=torch.device('cpu'))['model'].float()
    print(f'Succcesfully Loaded pytorch model from {torch_model_path}\n'+'-'*50)
    return torch_model

def regularize_shape(input_shape, stride):
    regularized_image_shape = [math.ceil(x / stride) * stride for x in input_shape[2:]]
    regularized_input_shape = [*input_shape[:2], *regularized_image_shape]
    print(f'WARNING: image size must be multiple of max stride {stride}, updating to {regularized_input_shape}')
    return regularized_input_shape
    