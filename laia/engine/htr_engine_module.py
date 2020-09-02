from typing import Any, Callable, Dict, Iterable, Optional

import pytorch_lightning as pl
import torch

from laia.callbacks.meters import SequenceError, char_to_word_seq
from laia.decoders import CTCGreedyDecoder
from laia.engine import EngineModule
from laia.losses import CTCLoss


class HTREngineModule(EngineModule):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: str,
        optimizer_kwargs: Dict,
        delimiters: Iterable,
        criterion: Optional[Callable] = CTCLoss(),
        monitor: str = "va_cer",
        batch_input_fn: Optional[Callable] = None,
        batch_target_fn: Optional[Callable] = None,
        batch_id_fn: Optional[Callable] = None,
    ):
        super().__init__(
            model,
            optimizer,
            optimizer_kwargs,
            criterion,
            monitor=monitor,
            batch_input_fn=batch_input_fn,
            batch_target_fn=batch_target_fn,
            batch_id_fn=batch_id_fn,
        )
        self.delimiters = delimiters
        self.decoder = CTCGreedyDecoder()

    def training_step(self, batch: Any, batch_idx: int) -> pl.TrainResult:
        result = super().training_step(batch, batch_idx)
        batch_x, batch_y = self.prepare_batch(batch)
        batch_decode = self.decoder(self.batch_y_hat)
        cer = torch.tensor(
            SequenceError.compute(batch_y, batch_decode), device=batch_x.device
        )
        result.log(
            "tr_cer",
            cer,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        batch_decode_words = [
            char_to_word_seq(b, self.delimiters) for b in batch_decode
        ]
        batch_y_words = [char_to_word_seq(b, self.delimiters) for b in batch_y]
        wer = torch.tensor(
            SequenceError.compute(batch_y_words, batch_decode_words),
            device=batch_x.device,
        )
        result.log(
            "tr_wer",
            wer,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return result

    def validation_step(self, batch: Any, batch_idx: int) -> Optional[pl.EvalResult]:
        result = super().validation_step(batch, batch_idx)
        batch_x, batch_y = self.prepare_batch(batch)
        batch_decode = self.decoder(self.batch_y_hat)
        cer = torch.tensor(
            SequenceError.compute(batch_y, batch_decode), device=batch_x.device
        )
        result.log(
            "va_cer",
            cer,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        if self.monitor == "va_cer":
            result.early_stop_on = cer
            result.checkpoint_on = cer
        batch_decode_words = [
            char_to_word_seq(b, self.delimiters) for b in batch_decode
        ]
        batch_y_words = [char_to_word_seq(b, self.delimiters) for b in batch_y]
        wer = torch.tensor(
            SequenceError.compute(batch_y_words, batch_decode_words),
            device=batch_x.device,
        )
        result.log(
            "va_wer",
            wer,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        if self.monitor == "va_wer":
            result.early_stop_on = wer
            result.checkpoint_on = wer
        return result
