import logging
import os
import time
from typing import Callable, Dict, List, Optional, Type
import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.autonotebook import trange

from pyhealth.metrics import (binary_metrics_fn, multiclass_metrics_fn,
                              multilabel_metrics_fn, regression_metrics_fn)
from pyhealth.utils import create_directory

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def is_best(best_score: float, score: float, monitor_criterion: str) -> bool:
    if monitor_criterion == "max":
        return score > best_score
    elif monitor_criterion == "min":
        return score < best_score
    else:
        raise ValueError(f"Monitor criterion {monitor_criterion} is not supported")


def set_logger(log_path: str) -> None:
    create_directory(log_path)
    log_filename = os.path.join(log_path, "log.txt")
    handler = logging.FileHandler(log_filename)
    formatter = logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return


def get_metrics_fn(mode: str) -> Callable:
    if mode == "binary":
        return binary_metrics_fn
    elif mode == "multiclass":
        return multiclass_metrics_fn
    elif mode == "multilabel":
        return multilabel_metrics_fn
    elif mode == "regression":
        return regression_metrics_fn
    else:
        raise ValueError(f"Mode {mode} is not supported")


class Trainer:

    def __init__(
            self,
            model: nn.Module,
            checkpoint_path: Optional[str] = None,
            metrics: Optional[List[str]] = None,
            device: Optional[str] = None,
            enable_logging: bool = True,
            seed: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = model
        self.metrics = metrics
        self.device = device
        self.enable_logging = enable_logging
        self.seed = seed

        # set logger
        if enable_logging:
            output_path = os.path.join(os.getcwd(), "output")
            self.exp_path = os.path.join(output_path, str(seed))
            set_logger(self.exp_path)
        else:
            self.exp_path = None

        # set device
        self.model.to(self.device)

        # logging
        logger.info(self.model)
        logger.info(f"Metrics: {self.metrics}")
        logger.info(f"Device: {self.device}")

        # load checkpoint
        if checkpoint_path is not None:
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            self.load_ckpt(checkpoint_path)

        logger.info("")
        return

    def train(
            self,
            train_dataloader: DataLoader,
            val_dataloader: Optional[DataLoader] = None,
            test_dataloader: Optional[DataLoader] = None,
            epochs: int = 5,
            optimizer_class: Type[Optimizer] = torch.optim.Adam,
            optimizer_params: Optional[Dict[str, object]] = None,
            steps_per_epoch: int = None,
            weight_decay: float = 0.0,
            max_grad_norm: float = None,
            monitor: Optional[str] = None,
            monitor_criterion: str = "max",
            load_best_model_at_last: bool = True,
            lr:float = 1e-3
    ):
        if optimizer_params is None:
            optimizer_params = {"lr": lr}

        # logging
        logger.info("Training:")
        logger.info(f"Batch size: {train_dataloader.batch_size}")
        logger.info(f"Optimizer: {optimizer_class}")
        logger.info(f"Optimizer params: {optimizer_params}")
        logger.info(f"Weight decay: {weight_decay}")
        logger.info(f"Max grad norm: {max_grad_norm}")
        logger.info(f"Val dataloader: {val_dataloader}")
        logger.info(f"Monitor: {monitor}")
        logger.info(f"Monitor criterion: {monitor_criterion}")
        logger.info(f"Epochs: {epochs}")

        # set optimizer
        param = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in param if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)

        # initialize
        data_iterator = iter(train_dataloader)
        best_score = -1 * float("inf") if monitor_criterion == "max" else float("inf")
        if steps_per_epoch == None:
            steps_per_epoch = len(train_dataloader)
        global_step = 0
        train_times = []
        # epoch training loop
        for epoch in range(epochs):
            start_time = time.time()
            training_loss = []
            self.model.zero_grad()
            self.model.train()
            # batch training loop
            logger.info("")
            for _ in trange(
                    steps_per_epoch,
                    desc=f"Training Epoch {epoch} / {epochs}",
                    smoothing=0.05,
            ):
                try:
                    data = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(train_dataloader)
                    data = next(data_iterator)
                # forward
                data['epoch'] = epoch
                output = self.model(**data)
                loss = output["loss"]
                # backward
                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_grad_norm
                    )
                # update
                optimizer.step()
                optimizer.zero_grad()
                training_loss.append(loss.item())
                global_step += 1
            epoch_time = time.time() - start_time
            train_times.append(epoch_time)
            # log and save
            logger.info(f"--- Train epoch-{epoch}, step-{global_step} ---")
            logger.info(f"loss: {sum(training_loss) / len(training_loss):.4f}")
            if self.exp_path is not None:
                self.save_ckpt(os.path.join(self.exp_path, "last.ckpt"))

            # validation
            if val_dataloader is not None:
                scores = self.evaluate(val_dataloader)
                logger.info(f"--- Eval epoch-{epoch}, step-{global_step} ---")
                for key in scores.keys():
                    logger.info("{}: {:.4f}".format(key, scores[key]))
                # save best model
                if monitor is not None:
                    score = scores[monitor]
                    if is_best(best_score, score, monitor_criterion):
                        logger.info(
                            f"New best {monitor} score ({score:.4f}) "
                            f"at epoch-{epoch}, step-{global_step}"
                        )
                        best_score = score
                        if self.exp_path is not None:
                            self.save_ckpt(os.path.join(self.exp_path, "best.ckpt"))

        # load best model
        if load_best_model_at_last and self.exp_path is not None and os.path.isfile(
                os.path.join(self.exp_path, "best.ckpt")):
            logger.info("Loaded best model")
            self.load_ckpt(os.path.join(self.exp_path, "best.ckpt"))

        # test
        if test_dataloader is not None:
            scores = self.evaluate(test_dataloader)
            logger.info(f"--- Test ---")
            for key in scores.keys():
                logger.info("{}: {:.4f}".format(key, scores[key]))
        return

    def inference(self, dataloader, additional_outputs=None,
                  return_patient_ids=False) -> Dict[str, float]:
        loss_all = []
        y_true_all = []
        y_prob_all = []
        patient_ids = []
        if additional_outputs is not None:
            additional_outputs = {k: [] for k in additional_outputs}
        for data in tqdm(dataloader, desc="Evaluation"):
            self.model.eval()
            with torch.no_grad():
                output = self.model(**data)
                loss = output["loss"]
                y_true = output["y_true"].cpu().numpy()
                y_prob = output["y_prob"].cpu().numpy()
                loss_all.append(loss.item())
                y_true_all.append(y_true)
                y_prob_all.append(y_prob)
                if additional_outputs is not None:
                    for key in additional_outputs.keys():
                        additional_outputs[key].append(output[key].cpu().numpy())
            if return_patient_ids:
                patient_ids.extend(data["patient_id"])
        loss_mean = sum(loss_all) / len(loss_all)
        y_true_all = np.concatenate(y_true_all, axis=0)
        y_prob_all = np.concatenate(y_prob_all, axis=0)
        outputs = [y_true_all, y_prob_all, loss_mean]
        if additional_outputs is not None:
            additional_outputs = {key: np.concatenate(val)
                                  for key, val in additional_outputs.items()}
            outputs.append(additional_outputs)
        if return_patient_ids:
            outputs.append(patient_ids)
        return outputs

    def evaluate(self, dataloader) -> Dict[str, float]:
        if self.model.mode is not None:
            y_true_all, y_prob_all, loss_mean = self.inference(dataloader)
            mode = self.model.mode
            metrics_fn = get_metrics_fn(mode)
            scores = metrics_fn(y_true_all, y_prob_all, metrics=self.metrics)
            scores["loss"] = loss_mean
        else:
            loss_all = []
            for data in tqdm(dataloader, desc="Evaluation"):
                self.model.eval()
                with torch.no_grad():
                    output = self.model(**data)
                    loss = output["loss"]
                    loss_all.append(loss.item())
            loss_mean = sum(loss_all) / len(loss_all)
            scores = {"loss": loss_mean}
        return scores

    def test(self, dataloader) -> Dict[str, float]:
        # load best model
        if self.exp_path is not None and os.path.isfile(
                os.path.join(self.exp_path, "best.ckpt")):
            logger.info("Loaded best model")
            self.load_ckpt(os.path.join(self.exp_path, "best.ckpt"))

        # test
        if dataloader is not None:
            scores = self.evaluate(dataloader)
            logger.info(f"--- Test ---")
            for key in scores.keys():
                logger.info("{}: {:.4f}".format(key, scores[key]))

    def save_ckpt(self, ckpt_path: str) -> None:
        """Saves the model checkpoint."""
        state_dict = self.model.state_dict()
        torch.save(state_dict, ckpt_path)
        return

    def load_ckpt(self, ckpt_path: str) -> None:
        """Saves the model checkpoint."""
        state_dict = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        return
