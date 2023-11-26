from torch.nn.modules.loss import _Loss

from torch import Module, Tensor


def evaluate_loss(
    model: Module, 
    input: Tensor, 
    target: Tensor,
    objective: _Loss
    ) -> float:
    
    output = model(input)
    return objective(output, target)