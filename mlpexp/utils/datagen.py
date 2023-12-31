import random

from typing import Tuple, List

import torch

from torch import Tensor
from torch.nn import functional as F

def rectangles_on_a_grid(grid_size: Tuple, sizes: List[Tuple], num_samples: int=10, num_classes: int=2) -> List[Tensor]:
    dataset = []

    for _ in range(num_samples):
        grid = torch.zeros(grid_size).to(torch.long)

        for class_id in range(1, num_classes + 1):
            rect_width, rect_height = sizes[class_id - 1]

            max_x_start = max(grid_size[0] - rect_width, 0)
            max_y_start = max(grid_size[1] - rect_height, 0)

            x1 = random.randint(0, max_x_start)
            y1 = random.randint(0, max_y_start)

            x2 = x1 + rect_width
            y2 = y1 + rect_height

            grid[x1:x2, y1:y2] = class_id

        target = F.one_hot(grid, num_classes=num_classes + 1)

        x_coords = torch.arange(grid_size[0]).view(-1, 1).expand(grid_size)
        y_coords = torch.arange(grid_size[1]).view(1, -1).expand(grid_size)
        input_tensor = torch.stack((x_coords, y_coords), dim=-1)

        dataset.append((input_tensor, target))

    return dataset