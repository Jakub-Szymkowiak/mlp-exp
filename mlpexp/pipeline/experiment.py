from nnes.pipeline.callbacks import EpochCallback
from nnes.pipeline.training import TrainSetup, train

from typing import Callable, Dict


class SimpleExperiment:
    def __init__(
            self, 
            setup_training: Callable[..., TrainSetup],
            epoch_callback: EpochCallback,
            callback_config: Dict
        ):

        self.setup_training = setup_training
        self.epoch_callback = epoch_callback
        self.callback_config = callback_config

    def setup(self, *args, **kwargs):
        self.train_setup = self.setup_training(*args, **kwargs)
        self.epoch_callback.set_config(self.callback_config)

    def run(self):
        return train(self.train_setup, self.epoch_callback)

