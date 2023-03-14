import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import torchsnooper

from sesm import Model, log_selector, my_optim


class PLModel(pl.LightningModule):
    def __init__(self, stage=1, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=["pretrained"])

        self.name_constraint = kwargs["name_constraints"]
        self.w_constraints = kwargs["w_constraints"]
        self.lr = kwargs["lr"]
        self.wd = kwargs["wd"]
        self.warm_up_step = kwargs["warm_up_step"]
        self.class_weights = torch.Tensor(kwargs["class_weights"])

        self.init_metrics(kwargs["d_out"])

        self.model = Model(**kwargs)
        self.stage = stage  # 1 for train encoder, 2 for train all

    def init_metrics(self, k):
        self.named_metrics_train = nn.ModuleDict(
            {
                "acc": torchmetrics.Accuracy(num_classes=k, average="micro"),
                "avg_p": torchmetrics.Precision(num_classes=k, average="macro"),
                "avg_r": torchmetrics.Recall(num_classes=k, average="macro"),
            }
        )
        self.named_metrics_val = nn.ModuleDict(
            {
                "acc": torchmetrics.Accuracy(num_classes=k, average="micro"),
                "avg_p": torchmetrics.Precision(num_classes=k, average="macro"),
                "avg_r": torchmetrics.Recall(num_classes=k, average="macro"),
            }
        )

    def forward(self, x, y):
        mask = x != 0
        # mask = None
        if self.stage == 1:
            y_hat = self.model.classifier(x, mask)
            loss = self.log_loss(y_hat, y, None)
            return loss
        else:
            y_hat, constraints, attn1, attn2 = self.model(x, mask)
            loss = self.log_loss(y_hat, y, constraints)
            return loss, y_hat, constraints, attn1, attn2

    def configure_optimizers(self):
        if self.stage == 1:
            return torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.wd
            )
        else:
            return my_optim(
                self.parameters(),
                self.lr,
                self.wd,
                self.warm_up_step,
                use_scheduler=True,
            )

    def log_loss(self, y_hat, y, constraints):
        if self.training:
            stage = "train"
            named_metrics = self.named_metrics_train
        else:
            stage = "val"
            named_metrics = self.named_metrics_val
        if self.stage == 1:
            stage = "encoder_" + stage

        y_loss = F.cross_entropy(
            y_hat, y.long(), weight=self.class_weights.to(y_hat.device)
        )

        # constraints
        if constraints is not None:
            c_loss = 0.0
            for n, w, c in zip(self.name_constraint, self.w_constraints, constraints):
                c_loss += w * c
                self.log(f"{stage}_loss_{n}", c.item())
            loss = y_loss + c_loss
        else:
            loss = y_loss

        # metrics
        for n, m in named_metrics.items():
            self.log(f"{stage}_step_{n}", m(y_hat.argmax(-1), y), prog_bar=True)
        self.log(f"{stage}_loss", loss)
        self.log(f"{stage}_y_loss", y_loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.stage == 1:
            loss = self(x, y)
        else:
            loss, y_hat, constraints, attn1, attn2 = self(x, y)
            # print(y_hat, y)

            # log
            if True and batch_idx % 100 == 0:
                print("y_pred, y_true")
                print(y_hat.argmax(dim=-1).detach().cpu().tolist())
                print(y.detach().cpu().tolist())
                log_selector(attn1, attn2, x != 0)

        return loss

    def training_epoch_end(self, outputs):
        stage = "encoder_" if self.stage == 1 else ""

        for n, m in self.named_metrics_train.items():
            self.log(f"{stage}train_epoch_{n}", m.compute(), prog_bar=True)
            m.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if self.stage == 1:
            loss = self(x, y)
        else:
            loss, y_hat, constraints, attn1, attn2 = self(x, y)
            # print(y_hat, y)

            # log
            if True and batch_idx % 10 == 0:
                print("y_pred, y_true")
                print(y_hat.argmax(dim=-1).detach().cpu().tolist())
                print(y.detach().cpu().tolist())
                log_selector(attn1, attn2, x != 0)

        return loss

    def validation_epoch_end(self, outputs):
        stage = "encoder_" if self.stage == 1 else ""
        for n, m in self.named_metrics_val.items():
            self.log(f"{stage}val_epoch_{n}", m.compute(), prog_bar=True)
            m.reset()
