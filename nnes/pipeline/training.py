from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss
from torch.nn import Module
from torch import Tensor

from typing import Callable, Any

from nnes.pipeline.callbacks import EpochCallback


class TrainSetup:
    def __init__(
            self,
            model: Module, 
            dataloader: DataLoader, 
            objective: _Loss,
            optimizer: Optimizer,
            num_epochs: int
        ) -> None:

        self.model = model
        self.dataloader = dataloader
        self.objective = objective
        self.optimizer = optimizer
        self.num_epochs = num_epochs

def train(
        train_setup: TrainSetup, 
        epoch_callback: EpochCallback
    ) -> None:

    train_setup.model.train()
    for epoch in range(train_setup.num_epochs):
        for batch_x, batch_y in train_setup.dataloader:
            train_setup.optimizer.zero_grad()
            output = train_setup.model(batch_x)
            loss = train_setup.objective(output, batch_y)
            loss.backward()
            train_setup.optimizer.step()
        epoch_callback(train_setup=train_setup, epoch=epoch)
    return train_setup

def setup_training(
        model: Module,
        input: Tensor,
        target: Tensor,
        objective: _Loss,
        optimization: Callable[[Module, Any], Optimizer],
        num_epochs: int,
        learning_rate: float,
        batch_size: int
    ) -> TrainSetup:

    optimizer = optimization(model.parameters(), learning_rate)
    
    dataset = TensorDataset(input, target)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return TrainSetup(
        model=model,
        dataloader=dataloader,
        objective=objective,
        optimizer=optimizer,
        num_epochs=num_epochs)