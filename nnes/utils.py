import torch

from torchvision.transforms import ToPILImage
from torch import Tensor

from typing import Tuple


def get_tensors_from_image(image: Tensor) -> Tuple[Tensor, Tensor]:
    H, W = image.shape
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    xx_normalized, yy_normalized = xx.float() / (W-1), yy.float() / (H - 1)

    X = torch.stack([xx_normalized, yy_normalized], dim=-1).view(-1, 2)
    Y = image.view(-1, 1).float() / 255.0

    return X, Y

def get_image_from_tensor(Y: Tensor, image_shape: Tuple) -> Tensor:
    return 255 * Y.view(*image_shape)

def display_tensor_image(image: Tensor) -> None:
    to_pil = ToPILImage()
    pil_image = to_pil(image)
    pil_image.show()

