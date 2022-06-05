from typing import Tuple, List, Dict, Optional

import torch
from commode_utils.losses import SequenceCrossEntropyLoss
from commode_utils.metrics import SequentialF1Score, ClassificationMetrics
from commode_utils.metrics.chrF import ChrF
from commode_utils.modules import LSTMDecoderStep, Decoder
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics import MetricCollection, Metric
from torchmetrics import Accuracy, BLEUScore
from code2seq.data.path_context import BatchedLabeledPathContext
from code2seq.data.vocabulary import Vocabulary
from code2seq.model.modules import PathEncoder
from code2seq.utils.optimization import configure_optimizers_alon
from code2seq.model.metrics import Perfect, Leivensitein, BLEU

import numpy as np


class Code2Seq(LightningModule):
    def __init__(
        self,
        model_config: DictConfig,
        optimizer_config: DictConfig,
        vocabulary: Vocabulary,
        teacher_forcing: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self._optim_config = optimizer_config
        self._vocabulary = vocabulary

        if vocabulary.SOS not in vocabulary.label_to_id:
            raise ValueError(f"Can't find SOS token in label to id vocabulary")

        self.__pad_idx = vocabulary.label_to_id[vocabulary.PAD]
        eos_idx = vocabulary.label_to_id[vocabulary.EOS]
        ignore_idx = [vocabulary.label_to_id[vocabulary.SOS], vocabulary.label_to_id[vocabulary.UNK]]
        metrics: Dict[str, Metric] = {
            f"{holdout}_f1": SequentialF1Score(pad_idx=self.__pad_idx, eos_idx=eos_idx, ignore_idx=ignore_idx)
            for holdout in ["train", "val", "test"]
        }
        id2label = {v: k for k, v in vocabulary.label_to_id.items()}
        metrics.update(
            {f"{holdout}_chrf": ChrF(id2label, ignore_idx + [self.__pad_idx, eos_idx]) for holdout in ["val", "test"]}
        )
        metrics.update(
            {f"{holdout}_perfect": Perfect() for holdout in ["train", "val", "test"]}
        )
        metrics.update(
            {f"{holdout}_lei": Leivensitein() for holdout in ["train", "val", "test"]}
        )
        metrics.update(
            {f"{holdout}_bleu": BLEU() for holdout in ["train", "val", "test"]}
        )
        metrics.update(
            {f"{holdout}_acc": Accuracy() for holdout in ["train", "val", "test"]}
        )
        self.__metrics = MetricCollection(metrics)

        self._encoder = self._get_encoder(model_config)
        decoder_step = LSTMDecoderStep(model_config, len(vocabulary.label_to_id), self.__pad_idx)
        self._decoder = Decoder(
            decoder_step, len(vocabulary.label_to_id), vocabulary.label_to_id[vocabulary.SOS], teacher_forcing
        )

        self.__loss = SequenceCrossEntropyLoss(self.__pad_idx, reduction="batch-mean")

    @property
    def vocabulary(self) -> Vocabulary:
        return self._vocabulary

    def _get_encoder(self, config: DictConfig) -> nn.Module:
        return PathEncoder(
            config,
            len(self._vocabulary.token_to_id),
            self._vocabulary.token_to_id[Vocabulary.PAD],
            len(self._vocabulary.node_to_id),
            self._vocabulary.node_to_id[Vocabulary.PAD],
        )

    # ========== Main PyTorch-Lightning hooks ==========

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[_LRScheduler]]:
        return configure_optimizers_alon(self._optim_config, self.parameters())

    def forward(  # type: ignore
        self,
        from_token: torch.Tensor,
        path_nodes: torch.Tensor,
        to_token: torch.Tensor,
        infer_vec: torch.Tensor,  # infer
        contexts_per_label: torch.Tensor, # データ枚のcontexts_pathの個数
        output_length: int,
        target_sequence: torch.Tensor = None # targetのlabels
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded_paths = self._encoder(from_token, path_nodes, to_token) # encoder のforwardがここで回っている [n_contexts; batch_size]
        output_logits, attention_weights = self._decoder(
            encoded_paths,  infer_vec, contexts_per_label, output_length, target_sequence
        )
        return output_logits, attention_weights

    # ========== Model step ==========

    def logits_from_batch(
        self, batch: BatchedLabeledPathContext, target_sequence: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self(
            batch.from_token,
            batch.path_nodes,
            batch.to_token,
            batch.infer_vec,
            batch.contexts_per_label,
            batch.labels.shape[0],
            target_sequence,
        )

    def _shared_step(self, batch: BatchedLabeledPathContext, step: str) -> Dict:
        target_sequence = batch.labels if step == "train" else None
        # [seq length; batch size; vocab size]
        logits, _ = self.logits_from_batch(batch, target_sequence)
        result = {f"{step}/loss": self.__loss(logits[1:], batch.labels[1:])}

        with torch.no_grad():
            prediction = logits.argmax(-1)
            metric: ClassificationMetrics = self.__metrics[f"{step}_f1"](prediction, batch.labels) # prediction 答え？　labels ラベル？
            result.update(
                {f"{step}/f1": metric.f1_score, f"{step}/precision": metric.precision, f"{step}/recall": metric.recall,}
            )
            torch.set_printoptions(edgeitems=1000)
            result[f"{step}/perfect"] = self.__metrics[f"{step}_perfect"](prediction, batch.labels)
            result[f"{step}/lei"] = self.__metrics[f"{step}_lei"](prediction, batch.labels)
            result[f"{step}/bleu"] = self.__metrics[f"{step}_bleu"](prediction, batch.labels)
            result[f"{step}/acc"] = self.__metrics[f"{step}_acc"](prediction, batch.labels)
            if step != "train":
                result[f"{step}/chrf"] = self.__metrics[f"{step}_chrf"](prediction, batch.labels)
        return result

    #学習用のロスを返している
    def training_step(self, batch: BatchedLabeledPathContext, batch_idx: int) -> Dict:  # type: ignore
        result = self._shared_step(batch, "train")
        # logはあくまでログ、 lossは学習に使われている
        self.log_dict(result, on_step=True, on_epoch=False)
        self.log("f1", result["train/f1"], prog_bar=True, logger=False)
        self.log("perfect", result["train/perfect"], prog_bar=True, logger=False)
        self.log("lei", result["train/lei"], prog_bar=True, logger=False)
        self.log("bleu", result["train/bleu"], prog_bar=True, logger=False)
        self.log("acc", result["train/acc"], prog_bar=True, logger=False)
        return result["train/loss"]

    def validation_step(self, batch: BatchedLabeledPathContext, batch_idx: int) -> Dict:  # type: ignore
        result = self._shared_step(batch, "val")
        return result["val/loss"]

    def test_step(self, batch: BatchedLabeledPathContext, batch_idx: int) -> Dict:  # type: ignore
        result = self._shared_step(batch, "test")
        return result["test/loss"]

    # ========== On epoch end ==========

    # epocの終わりに実行ロスの平均をとったりスル。
    def _shared_epoch_end(self, step_outputs: EPOCH_OUTPUT, step: str):
        with torch.no_grad():
            losses = [so if isinstance(so, torch.Tensor) else so["loss"] for so in step_outputs]
            mean_loss = torch.stack(losses).mean()
            metric = self.__metrics[f"{step}_f1"].compute()
            log = {
                f"{step}/loss": mean_loss,
                f"{step}/f1": metric.f1_score,
                f"{step}/precision": metric.precision,
                f"{step}/recall": metric.recall,
            }
            self.__metrics[f"{step}_f1"].reset()
            log[f"{step}/perfect"]  = self.__metrics[f"{step}_perfect"].compute()
            self.__metrics[f"{step}_perfect"].reset()
            log[f"{step}/lei"]  = self.__metrics[f"{step}_lei"].compute()
            self.__metrics[f"{step}_lei"].reset()
            log[f"{step}/bleu"]  = self.__metrics[f"{step}_bleu"].compute()
            self.__metrics[f"{step}_bleu"].reset()
            log[f"{step}/acc"]  = self.__metrics[f"{step}_acc"].compute()
            self.__metrics[f"{step}_acc"].reset()
            if step != "train":
                log[f"{step}/chrf"] = self.__metrics[f"{step}_chrf"].compute()
                self.__metrics[f"{step}_chrf"].reset()
        self.log_dict(log, on_step=False, on_epoch=True)

    def training_epoch_end(self, step_outputs: EPOCH_OUTPUT):
        self._shared_epoch_end(step_outputs, "train")

    def validation_epoch_end(self, step_outputs: EPOCH_OUTPUT):
        self._shared_epoch_end(step_outputs, "val")

    def test_epoch_end(self, step_outputs: EPOCH_OUTPUT):
        self._shared_epoch_end(step_outputs, "test")
