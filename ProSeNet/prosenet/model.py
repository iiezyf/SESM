import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl

from prosenet import Prototypes


default_prototypes_args = {
    "K": 8,
    "D": 128,
    "dmin": 1.0,
    "Ld": 0.01,
    "Lc": 0.01,
    "Le": 0.1,
}


class ProSeNet(pl.LightningModule):
    def __init__(
        self,
        n_classes,
        class_weights,
        encoder,
        prototypes_args=default_prototypes_args,
    ):
        """
        Parameters
        ----------
        input_shape : tuple(int)
            Shape of input sequences, probably: (length, features)
            `length` may be `None` for variable length data.
        nclasses : int
            Number of output classes
        k : int
            Number of prototype vectors in `Prototypes` layer
        rnn_args : dict, optional
            Any updates to default `encoder.rnn` construction args.
        prototypes_args : dict, optional
            Any updates to default `Prototypes` layer args.
        L1 : float, optional
            Strength of L1 regularization for `Dense` classifier kernel.
        """
        super(ProSeNet, self).__init__()
        self.save_hyperparameters(ignore=["encoder"])

        # Construct encoder network
        self.encoder = encoder

        # Construct `Prototypes` layer
        default_prototypes_args.update(prototypes_args)
        self.prototypes_layer = Prototypes(**default_prototypes_args)

        # Dense classifier with kernel restricted to >= 0.
        self.classifier = nn.Sequential(
            # nn.Linear(default_prototypes_args["K"], default_prototypes_args["K"]),
            # nn.ReLU(),
            nn.Linear(default_prototypes_args["K"], n_classes),
        )

        self.class_weights = class_weights
        self.named_metrics_train = nn.ModuleDict(
            {
                "acc": torchmetrics.Accuracy(num_classes=n_classes, average="micro"),
                "avg_p": torchmetrics.Precision(num_classes=n_classes, average="macro"),
                "avg_r": torchmetrics.Recall(num_classes=n_classes, average="macro"),
            }
        )
        self.named_metrics_val = nn.ModuleDict(
            {
                "acc": torchmetrics.Accuracy(num_classes=n_classes, average="micro"),
                "avg_p": torchmetrics.Precision(num_classes=n_classes, average="macro"),
                "avg_r": torchmetrics.Recall(num_classes=n_classes, average="macro"),
            }
        )

        self.hparams.lr = 0.01
        self.hparams.warm_up_step = 10

    def forward(self, x, training=None):
        """Full forward call."""
        d2, (dLoss, cLoss, eLoss) = self.similarity_vector(x, training)
        # print(d2.shape)
        return self.classifier(d2), (dLoss, cLoss, eLoss)

    def inspect(self, x):
        d2, _ = self.similarity_vector(x, False)
        return d2

    def similarity_vector(self, x, training):
        """Return the similarity vector(s) of shape (batches, k,)."""

        r_x = self.encoder(x)

        return self.prototypes_layer(r_x, training)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad == True, self.parameters()),
            lr=self.hparams.lr,
        )
        # return optimizer

        def lr_foo(epoch):
            if epoch < self.hparams.warm_up_step:
                # warm up lr
                lr_scale = 0.1 ** (self.hparams.warm_up_step - epoch)
            else:
                lr_scale = 0.95 ** epoch

            return lr_scale

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_foo)

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat, (dLoss, cLoss, eLoss) = self(X, training=True)
        # print(y_hat, y)

        for n, m in self.named_metrics_train.items():
            self.log(f"train_step_{n}", m(y_hat.argmax(-1), y))

        loss = F.cross_entropy(y_hat, y, self.class_weights) + sum(
            [dLoss, cLoss, eLoss]
        )
        self.log("dLoss", dLoss, prog_bar=True)
        self.log("cLoss", cLoss, prog_bar=True)
        self.log("eLoss", eLoss, prog_bar=True)
        return loss

    def training_epoch_end(self, outputs):
        for n, m in self.named_metrics_train.items():
            self.log(f"train_epoch_{n}", m.compute(), prog_bar=True)
            m.reset()

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat, (dLoss, cLoss, eLoss) = self(X)
        # print(y_hat, y)

        for n, m in self.named_metrics_val.items():
            self.log(f"val_step_{n}", m(y_hat.argmax(-1), y))

        loss = F.cross_entropy(y_hat, y, self.class_weights) + sum(
            [dLoss, cLoss, eLoss]
        )
        return loss

    def validation_epoch_end(self, outputs):
        for n, m in self.named_metrics_val.items():
            self.log(f"val_epoch_{n}", m.compute(), prog_bar=True)
            m.reset()
