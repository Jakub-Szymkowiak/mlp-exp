from typing import Callable, Dict, List


class EpochCallback:
    def __init__(
            self, 
            callbacks: List[Callable[[dict, 'TrainSetup'], None]]
        ) -> None:
        
        self.callbacks = callbacks

    def __call__(self, train_setup: 'TrainSetup'):
        callback_output = []
        for callback in self.callbacks:
            output = callback(self.config, train_setup)
            callback_output.append(output)

    def set_config(self, config: Dict):
        self.config = config