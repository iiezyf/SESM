import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics

default_rnn_args = {
    "input_size": 128,
    "hidden_size": 128,
    "num_layers": 2,
    "bidirectional": True,
    "batch_first": True,
}


class Encoder(nn.Module):
    def __init__(self, rnn_args, vocab_size=None):
        super(Encoder, self).__init__()
        default_rnn_args.update(rnn_args)
        if vocab_size is not None:
            self.embed = nn.Embedding(vocab_size, default_rnn_args["input_size"])
        else:
            self.embed = None
        self.rnn = nn.LSTM(**default_rnn_args)

    def forward(self, x):
        if self.embed is not None:
            x = self.embed(x)
        out, (h, c) = self.rnn(x)
        return h.mean(0)


class EncoderClassifier(pl.LightningModule):
    def __init__(self, encoder, class_weights, hidden_size, n_classes):
        super(EncoderClassifier, self).__init__()
        self.save_hyperparameters(ignore=["encoder"])

        self.encoder = encoder
        self.pred = nn.Linear(hidden_size, n_classes)

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

        self.hparams.lr = 0.1
        # self.hparams.warm_up_step = 10

    def forward(self, x):
        x = self.encoder(x)
        x = self.pred(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer

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
        y_hat = self(X)
        for n, m in self.named_metrics_train.items():
            self.log(f"train_step_{n}", m(y_hat.argmax(-1), y))
        loss = F.cross_entropy(y_hat, y, self.class_weights)
        return loss

    def training_epoch_end(self, outputs):
        for n, m in self.named_metrics_train.items():
            self.log(f"train_epoch_{n}", m.compute(), prog_bar=True)
            m.reset()

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X)
        for n, m in self.named_metrics_val.items():
            self.log(f"val_step_{n}", m(y_hat.argmax(-1), y))
        loss = F.cross_entropy(y_hat, y, self.class_weights)
        return loss

    def validation_epoch_end(self, outputs):
        for n, m in self.named_metrics_val.items():
            self.log(f"val_epoch_{n}", m.compute(), prog_bar=True)
            m.reset()
