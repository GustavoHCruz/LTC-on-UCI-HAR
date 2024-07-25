import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.utils.data as data
import torchmetrics
from ncps.torch import LTC
from ncps.wirings import AutoNCP
from plyer import notification
from pytorch_lightning.loggers import CSVLogger

sys.path.append(os.path.abspath("funcs"))

from config_reading import read_and_validate_json
from latest_log import get_latest_csv
from timer_callback import TimingCallback

config = read_and_validate_json("config.json")

torch.set_float32_matmul_precision(config["float_precision"])

torch.manual_seed(config["seed"])
torch.cuda.manual_seed(config["seed"])
torch.cuda.manual_seed_all(config["seed"])
np.random.seed(config["seed"])
pl.seed_everything(config["seed"], workers=True)

train_x = torch.load(os.path.join(config["save_path"], "tensor_train_x.pt"))
train_y = torch.load(os.path.join(config["save_path"], "tensor_train_y.pt"))
val_x = torch.load(os.path.join(config["save_path"], "tensor_val_x.pt"))
val_y = torch.load(os.path.join(config["save_path"], "tensor_val_y.pt"))
test_x = torch.load(os.path.join(config["save_path"], "tensor_test_x.pt"))
test_y = torch.load(os.path.join(config["save_path"], "tensor_test_y.pt"))

train_dataloader = data.DataLoader(data.TensorDataset(train_x, train_y), batch_size=16, shuffle=True, num_workers=16, persistent_workers=True)
val_dataloader = data.DataLoader(data.TensorDataset(val_x, val_y), batch_size=16, num_workers=16, persistent_workers=True)
test_dataloader = data.DataLoader(data.TensorDataset(test_x, test_y), batch_size=16, num_workers=16, persistent_workers=True)

class SequenceLearner(pl.LightningModule):
	def __init__(self, model, lr=0.005):
		super().__init__()
		self.model = model
		self.lr = lr
		self.loss_fn = nn.CrossEntropyLoss()
		self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=6)
		self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=6)
		self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=6)
		self.log_collection = []

	def training_step(self, batch, batch_idx):
		x, y = batch
		y_hat, _ = self.model.forward(x)
		loss = self.loss_fn(y_hat, y)
		y_pred = y_hat.argmax(dim=-1)
		self.log_collection.append({"loss": loss, "acc": self.train_acc(y_pred, y)})
		return {"loss": loss}

	def validation_step(self, batch, batch_idx):
		x, y = batch
		y_hat, _ = self.model.forward(x)
		loss = self.loss_fn(y_hat, y)
		y_pred = y_hat.argmax(dim=-1)
		self.log_collection.append({"loss": loss, "acc": self.val_acc(y_pred, y)})
		return {"loss": loss}
	
	def test_step(self, batch, batch_idx):
		x, y = batch
		y_hat, _ = self.model.forward(x)
		loss = self.loss_fn(y_hat, y)
		y_pred = y_hat.argmax(dim=-1)
		self.log("test_loss", loss)
		self.log("test_acc", self.test_acc(y_pred, y))
		return {"loss": loss}

	def on_validation_epoch_end(self):
		self.log("train_loss", self.log_collection[0]["loss"], on_step=False, on_epoch=True)
		self.log("train_acc", self.log_collection[0]["acc"], on_step=False, on_epoch=True)
		self.log("val_loss", self.log_collection[1]["loss"], on_step=False, on_epoch=True)
		self.log("val_acc", self.log_collection[1]["acc"], on_step=False, on_epoch=True)
		self.log_collection = []


	def configure_optimizers(self):
		return torch.optim.Adam(self.model.parameters(), lr=self.lr)
	
out_features = 6 # Output
in_features = 561 # Input

wiring = AutoNCP(config["num_neurons"], out_features)

ltc_model = LTC(in_features, wiring, batch_first=True)
learn = SequenceLearner(ltc_model, lr=config["learning_rate"])

log_dir = f"logs"

logger = CSVLogger(log_dir, name=config["model_name"])

trainer = pl.Trainer(
	logger=logger,
	max_epochs=config["max_epochs"],
  check_val_every_n_epoch=1,
  callbacks=[TimingCallback()],
	gradient_clip_val=1  # Clip gradient to stabilize training
)

trainer.fit(learn, train_dataloader, val_dataloader)

trainer.test(learn, test_dataloader)

latest_csv = get_latest_csv(logger.log_dir)
metrics = pd.read_csv(latest_csv)

steps = metrics["epoch"]

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(steps, metrics["train_acc"], label='Train Accuracy')
plt.plot(steps, metrics["val_acc"], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

if not os.path.exists("models"):
    os.makedirs("models")

torch.save(ltc_model, f"models/{config["model_name"]}.pt")

config_file_path = os.path.join(logger.log_dir, "config.json")
with open(config_file_path, 'w') as config_file:
  json.dump(config, config_file, indent=2)

notification.notify(
	title="Training ended",
	message=f"The {config["max_epochs"]} epochs training has been completed.",
	timeout=10
)