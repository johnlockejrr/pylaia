from __future__ import absolute_import

import laia.logging as log
from laia.engine.trainer import Trainer
from laia.engine.triggers.trigger import TriggerLogWrapper, LoggedTrigger

_logger = log.get_logger(__name__)


class NumUpdates(LoggedTrigger):
    """Trigger after the given `trainer` reaches a given number of updates.

    Arguments:
        trainer (:obj:`~laia.engine.Trainer`) : trainer to monitor
        num_updates (int) : number of updates to reach
        name (str) : name of the trigger
    """

    def __init__(self, trainer, num_updates, name=None):
        # type: (Trainer, int, str) -> None
        super(NumUpdates, self).__init__(_logger, name)
        assert isinstance(trainer, Trainer)
        self._trainer = trainer
        self._num_updates = num_updates

    def __call__(self):
        if self._trainer.updates >= self._num_updates:
            if self._trainer.updates == self._num_updates:
                self.logger.info(TriggerLogWrapper(
                    self, 'Trainer reached {} updates',
                    self._num_updates))
            return True
        else:
            self.logger.debug(TriggerLogWrapper(
                self, 'Trainer DID NOT reach {} updates yet',
                self._num_updates))
            return False
