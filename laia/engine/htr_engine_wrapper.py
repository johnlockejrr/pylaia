from __future__ import absolute_import

import laia.plugins.logging as log
from laia.decoders import CTCDecoder
from laia.engine.engine import Engine
from laia.losses import CTCLoss
from laia.meters import RunningAverageMeter, SequenceErrorMeter, TimeMeter


class HtrEngineWrapper(object):
    r"""Engine wrapper to perform HTR experiments."""

    ON_BATCH_START = Engine.ON_BATCH_START
    ON_EPOCH_START = Engine.ON_EPOCH_START
    ON_BATCH_END = Engine.ON_BATCH_END
    ON_EPOCH_END = Engine.ON_EPOCH_END

    def __init__(self, train_engine, valid_engine=None):
        self._tr_engine = train_engine
        self._va_engine = valid_engine

        # If the trainer was created without any criterion, or it is not
        # the CTCLoss, set it properly.
        if not self._tr_engine.criterion:
            self._tr_engine.set_criterion(CTCLoss())
        elif not isinstance(self._tr_engine.criterion, CTCLoss):
            log.warn('Overriding the criterion of the trainer to CTC.', name=__name__)
            self._tr_engine.set_criterion(CTCLoss())

        self._ctc_decoder = CTCDecoder()
        self._train_timer = TimeMeter()
        self._train_loss_meter = RunningAverageMeter()
        self._train_cer_meter = SequenceErrorMeter()

        self._tr_engine.add_hook(self.ON_EPOCH_START, self._train_reset_meters)
        self._tr_engine.add_hook(self.ON_BATCH_END, self._train_accumulate_loss)
        self._tr_engine.add_hook(self.ON_BATCH_END, self._train_compute_cer)

        if valid_engine:
            self._valid_timer = TimeMeter()
            self._valid_loss_meter = RunningAverageMeter()
            self._valid_cer_meter = SequenceErrorMeter()

            self._va_engine.add_hook(
                self.ON_EPOCH_START, self._valid_reset_meters)
            self._va_engine.add_hook(
                self.ON_BATCH_END, self._valid_accumulate_loss)
            self._va_engine.add_hook(
                self.ON_BATCH_END, self._valid_compute_cer)
            self._va_engine.add_hook(
                self.ON_EPOCH_END, self._report_epoch_train_and_valid)

            # Add evaluator to the trainer engine
            self._tr_engine.add_evaluator(self._va_engine)
        else:
            self._valid_timer = None
            self._valid_loss_meter = None
            self._valid_cer_meter = None
            self._tr_engine.add_hook(
                self.ON_EPOCH_END, self._report_epoch_train_only)

    @property
    def train_timer(self):
        return self._train_timer

    @property
    def valid_timer(self):
        return self._valid_timer

    @property
    def train_loss(self):
        return self._train_loss_meter

    @property
    def valid_loss(self):
        return self._valid_loss_meter

    @property
    def train_cer(self):
        return self._train_cer_meter

    @property
    def valid_cer(self):
        return self._valid_cer_meter

    def run(self):
        self._tr_engine.run()
        return self

    def _train_reset_meters(self, **_):
        self._train_timer.reset()
        self._train_loss_meter.reset()
        self._train_cer_meter.reset()

    def _valid_reset_meters(self, **_):
        self._valid_timer.reset()
        self._valid_loss_meter.reset()
        self._valid_cer_meter.reset()

    def _train_accumulate_loss(self, batch_loss, **_):
        self._train_loss_meter.add(batch_loss)
        # Note: Stop training timer to avoid including extra costs
        # (e.g. the validation epoch)
        self._train_timer.stop()

    def _valid_accumulate_loss(self, batch_output, batch_target, **_):
        batch_loss = self._tr_engine.criterion(batch_output, batch_target)
        self._valid_loss_meter.add(batch_loss)
        # Note: Stop timer to avoid including extra costs
        self._valid_timer.stop()

    def _train_compute_cer(self, batch_output, batch_target, **_):
        batch_decode = self._ctc_decoder(batch_output)
        self._train_cer_meter.add(batch_target, batch_decode)

    def _valid_compute_cer(self, batch_output, batch_target, **_):
        batch_decode = self._ctc_decoder(batch_output)
        self._valid_cer_meter.add(batch_target, batch_decode)

    def _report_epoch_train_only(self, **_):
        # Average training loss in the last EPOCH
        tr_loss, _ = self.train_loss.value
        # Average training CER in the last EPOCH
        tr_cer = self.train_cer.value
        # Timers
        tr_time = self.train_timer.value
        log.info('Epoch {:4d}, '
                 'TR Loss = {:.3e}, '
                 'TR CER = {:6.2%}, '
                 'TR Time = {:.2f}s',
                 self._tr_engine.epochs,
                 tr_loss,
                 tr_cer,
                 tr_time, name=__name__)

    def _report_epoch_train_and_valid(self, **_):
        # Average loss in the last EPOCH
        tr_loss, _ = self.train_loss.value
        va_loss, _ = self.valid_loss.value
        # Average CER in the last EPOCH
        tr_cer = self.train_cer.value
        va_cer = self.valid_cer.value
        # Timers
        tr_time = self.train_timer.value
        va_time = self.valid_timer.value
        log.info('Epoch {:4d}, '
                 'TR Loss = {:.3e}, '
                 'VA Loss = {:.3e}, '
                 'TR CER = {:5.1%}, '
                 'VA CER = {:5.1%}, '
                 'TR Time = {:.2f}s, '
                 'VA Time = {:.2f}s',
                 self._tr_engine.epochs,
                 tr_loss,
                 va_loss,
                 tr_cer,
                 va_cer,
                 tr_time,
                 va_time, name == __name__)
