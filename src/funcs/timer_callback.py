import time

from pytorch_lightning.callbacks import Callback


class TimingCallback(Callback):
  def on_train_start(self, trainer, pl_module):
    self.start_time = time.time()

  def on_train_end(self, trainer, pl_module):
    self.end_time = time.time()
    total_time = self.end_time - self.start_time
    trainer.logger.log_metrics({"training_total_time": total_time})
