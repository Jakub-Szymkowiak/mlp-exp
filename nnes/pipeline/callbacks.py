from typing import Callable, Dict, List


class EpochCallback:
    def __init__(
            self, 
            callbacks: List[Callable[[dict, 'TrainSetup', int], None]]
        ) -> None:
        
        self.callbacks = callbacks

    def __call__(self, train_setup: 'TrainSetup', epoch: int):
        callback_output = []
        for callback in self.callbacks:
            output = callback(self.config, train_setup, epoch)
            callback_output.append(output)

    def set_config(self, config: Dict):
        self.config = config