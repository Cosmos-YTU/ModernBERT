# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0

from composer.core import Callback, State
from composer.loggers import Logger

__all__ = ["MaskingRatio"]


class MaskingRatio(Callback):
    """Records the packing efficiency for each batch."""

    def __init__(self, log_interval: int = 100):
        self.log_interval = log_interval

    def after_dataloader(self, state: State, logger: Logger) -> None:
        if state.timestamp.batch.value % self.log_interval != 0:
            return
        logger.log_metrics(
            {
                "trainer/masking_ratio": state.batch["mask_prob"],
            }
        )
