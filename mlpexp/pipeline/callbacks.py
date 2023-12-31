from typing import Callable, Dict, List

from mlpexp.utils.evaluation import evaluate_loss


class EpochCallback:
    def __init__(
            self, 
            callbacks: List[Callable[[dict, 'TrainSetup', int], None]]
        ) -> None:
        
        self.callbacks = callbacks

    def __call__(self, train_setup: 'TrainSetup', epoch: int):
        for callback in self.callbacks:
            output = callback(self.config, train_setup, epoch)

    def set_config(self, config: Dict):
        self.config = config


def print_loss_callback(config: Dict, train_setup: 'TrainSetup', epoch: int):
  if epoch % config["log_interval"] == 0:
    loss = evaluate_loss(
        train_setup.model,
        config["X_test"],
        config["Y_test"],
        train_setup.objective)
    print(f"[{epoch + 1} / {train_setup.num_epochs}] {loss}")