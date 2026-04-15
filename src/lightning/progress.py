import math

from lightning.pytorch.callbacks.progress.tqdm_progress import TQDMProgressBar


class EpochTQDMProgressBar(TQDMProgressBar):
    def _epoch_desc(self, trainer) -> str:
        return f"epoch={trainer.current_epoch}"

    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.set_description(self._epoch_desc(self.trainer))
        return bar

    def on_train_epoch_start(self, trainer, *_: object) -> None:
        super().on_train_epoch_start(trainer)
        if self.train_progress_bar is not None:
            self.train_progress_bar.set_description(self._epoch_desc(trainer))

    def get_metrics(self, trainer, pl_module):
        metrics = super().get_metrics(trainer, pl_module)
        metrics.pop("v_num", None)
        return metrics

    @property
    def total_train_batches(self):
        total = super().total_train_batches
        if total is None or (isinstance(total, float) and math.isinf(total)):
            combined_loader = getattr(self.trainer.fit_loop, "_combined_loader", None)
            if combined_loader is not None:
                try:
                    total = len(combined_loader)
                except TypeError:
                    pass
            if total is None or (isinstance(total, float) and math.isinf(total)):
                dataloader = self.trainer.train_dataloader
                if dataloader is not None:
                    try:
                        total = len(dataloader)
                    except TypeError:
                        pass
        return total
