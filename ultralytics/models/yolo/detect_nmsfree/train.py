from ultralytics.models.yolo.detect import DetectionTrainer
from .val import NMSFreeDetectionValidator
from ultralytics.nn.tasks import NMSFreeDetectionModel
from copy import copy
from ultralytics.utils import RANK
from ultralytics.utils.plotting import plot_results

class NMSFreeDetectionTrainer(DetectionTrainer):
    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_om", "cls_om", "dfl_om", "box_oo", "cls_oo", "dfl_oo", 
        return NMSFreeDetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        model = NMSFreeDetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def plot_metrics(self):
        """Plots metrics from a CSV file."""
        plot_results(file=self.csv, nmsfree=True, on_plot=self.on_plot)  # save results.png
