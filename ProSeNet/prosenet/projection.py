"""
Implement a PrototypeProjection callback
"""
import torch
import pytorch_lightning as pl


class PrototypeProjection(pl.Callback):
    def __init__(self, model, train_gen, freq=4):
        super().__init__()
        self.model = model
        self.train_gen = train_gen
        self.freq = freq

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if (trainer.current_epoch + 1) % self.freq == 0:
            device = self.model.device
            X_encoded = []
            for batch in self.train_gen:
                X, y = batch
                X_encoded.append(self.model.encoder(X.to(device)))
            X_encoded = torch.cat(X_encoded, dim=0).unsqueeze(-2)

            protos = self.model.prototypes_layer.prototypes.to(device)
            d2 = torch.cdist(X_encoded, protos)
            print("new protos", d2.argmin(0))
            new_protos = X_encoded[d2.argmin(0)]
            new_protos = new_protos.reshape(protos.shape)

            self.model.prototypes_layer.prototypes.data = new_protos
            print("... assigned new prototypes from projections.")
