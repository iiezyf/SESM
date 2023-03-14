import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import torchsnooper


class EncoderClassifier(pl.LightningModule):
    def __init__(self, embed, encoder, d_hidden, num_classes, lr=0.01):
        super(EncoderClassifier, self).__init__()
        self.save_hyperparameters(ignore=["embed", "encoder"])

        self.embed = embed
        self.encode = encoder
        self.fc = nn.Linear(d_hidden // 4, num_classes)

        self.lr = lr
        self.named_metrics_train = nn.ModuleDict(
            {
                "acc": torchmetrics.Accuracy(num_classes=num_classes),
                "f1": torchmetrics.F1(num_classes=num_classes, average="macro"),
            }
        )
        self.named_metrics_val = nn.ModuleDict(
            {
                "acc": torchmetrics.Accuracy(num_classes=num_classes),
                "f1": torchmetrics.F1(num_classes=num_classes, average="macro"),
            }
        )
        self.cm = torchmetrics.ConfusionMatrix(num_classes=num_classes)

    # @torchsnooper.snoop()
    def forward(self, x, y):
        x = self.embed(x.unsqueeze(1))
        x = self.encode(x)
        x = x.reshape(x.shape[0], -1)
        y_hat = self.fc(x)

        loss = F.cross_entropy(y_hat, y)
        if self.training:
            stage = "train"
            nm = self.named_metrics_train
        else:
            stage = "val"
            nm = self.named_metrics_val

        for n, m in nm.items():
            self.log(f"encoder_{stage}_step_{n}", m(y_hat.argmax(-1), y), prog_bar=True)
        self.log(f"encoder_{stage}_step_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self(x, y)
        return loss

    def training_epoch_end(self, outputs):
        for n, m in self.named_metrics_train.items():
            self.log(f"encoder_train_epoch_{n}", m.compute(), prog_bar=True)
            m.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self(x, y)
        return loss

    def validation_epoch_end(self, outputs):
        for n, m in self.named_metrics_val.items():
            self.log(f"encoder_val_epoch_{n}", m.compute(), prog_bar=True)
            m.reset()
