import torch
from torch import nn
import pytorch_lightning as pl
from torch.nn import functional as F


class parameterGen:
    def __init__(
        self,
        lr=1e-3,
        momentum=0,
        weight_decay=0,
        hidden_units=256,
        optimizers=torch.optim.Adam,
        layer_1_act=torch.relu,
        layer_2_act=torch.relu,
        layer_3_act=torch.log_softmax,
    ):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.hidden_units = hidden_units
        self.optimizers = optimizers
        self.layer_act = {
            "1": layer_1_act,
            "2": layer_2_act,
            "3": layer_3_act,
        }


from torchmetrics.classification import Accuracy


class MNISTClassifier(pl.LightningModule):
    def __init__(self, para: parameterGen):
        super().__init__()

        # mnist images are (1, 28, 28) (channels, width, height)
        self.para = para
        self.layer_1 = torch.nn.Linear(28 * 28, 128)
        self.layer_2 = torch.nn.Linear(128, self.para.hidden_units)
        self.layer_3 = torch.nn.Linear(self.para.hidden_units, 10)
        self.accuracy = Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        batch_size, channels, width, height = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)

        # layer 1 (b, 1*28*28) -> (b, 128)
        x = self.layer_1(x)
        x = self.para.layer_act["1"](x)

        # layer 2 (b, 128) -> (b, 256)
        x = self.layer_2(x)
        x = self.para.layer_act["2"](x)

        # layer 3 (b, 256) -> (b, 10)
        x = self.layer_3(x)

        # probability distribution over labels
        x = self.para.layer_act["3"](x, dim=1)

        return x

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        self.log("lr", self.para.lr)
        self.log("hidden_units", self.para.hidden_units)

        x, y = train_batch
        preds = self(x)
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)

        # log step metric
        self.accuracy(preds, y)
        self.log("train_acc_step", self.accuracy)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        self.log("lr", self.para.lr)
        self.log("hidden_units", self.para.hidden_units)

        x, y = val_batch
        logits = self.forward(x)
        preds = self(x)
        loss = self.cross_entropy_loss(logits, y)

        # log step metric
        self.accuracy(preds, y)
        self.log("val_acc_step", self.accuracy)

        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        preds = self(x)
        acc = self.accuracy(preds, y)
        self.log("test_acc_step", acc)

    def configure_optimizers(self):
        optimizer = self.para.optimizers(self.parameters(), lr=self.para.lr)
        return optimizer
