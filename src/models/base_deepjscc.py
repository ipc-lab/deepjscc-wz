from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
import torch
from lightning import LightningModule
from torchmetrics import MetricCollection
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from src.models.components.metrics import CustomLearnedPerceptualImagePatchSimilarity, CustomMetricCollection, CustomMultiScaleSSIM, CustomPeakSignalNoiseRatio, CustomStructuralSimilarityIndexMeasure, MultiScaleSSIM, CustomInceptionScore
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure
from pytorch_msssim import ms_ssim
import os
import torchvision

class BaseDeepJSCC(LightningModule):
    def __init__(
        self,
        scheduler: torch.optim.lr_scheduler,
        csi: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        channel: torch.nn.Module,
        power_constraint: torch.nn.Module,
        input_dims: Tuple[int, int, int],
        loss: torch.nn.Module,
        log_test_batches: int = 0,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=[])

        self.csi = csi()
        self.channel = channel()
        self.power_constraint = power_constraint()
        self.loss = loss()
        
        self.init_metrics()
        
        
        if hasattr(self.hparams, "ckpt_path") and self.hparams.ckpt_path is not None:
            self.load_state_dict(torch.load(self.hparams.ckpt_path, map_location=self.device)["state_dict"])
        
    def init_metrics(self):
        
        self.data_range = (0.0, 1.0)
        self.msssim_kernel_size = self.find_msssim_kernel_size()

        train_metrics = MetricCollection({"psnr": PeakSignalNoiseRatio(data_range=self.data_range)})
        
        msssim_metric = { 
                        "msssim": CustomMultiScaleSSIM(kernel_size=self.msssim_kernel_size, data_range=1.0),
        } if self.msssim_kernel_size is not None else {}
        
        eval_metrics = CustomMetricCollection({
                "psnr": CustomPeakSignalNoiseRatio(data_range=self.data_range),
                "lpips": CustomLearnedPerceptualImagePatchSimilarity(
                    net_type="vgg", normalize=True
                ),
                "ssim": CustomStructuralSimilarityIndexMeasure(kernel_size=11, data_range=self.data_range),
            } | msssim_metric
        )
        
        eval_metrics_perceptual = CustomMetricCollection(
            {
                #"is": CustomInceptionScore(normalize=True),
            }
        )
        train_metrics.requires_grad_(False)
        eval_metrics.requires_grad_(False)
        eval_metrics_perceptual.requires_grad_(False)

        self.train_metrics = train_metrics.clone(prefix="train/")
        self.val_metrics = eval_metrics.clone(prefix="val/")
        self.test_metrics = eval_metrics.clone(prefix="test/")
        self.val_metrics_perceptual = eval_metrics_perceptual.clone(prefix="val/")
        self.test_metrics_perceptual = eval_metrics_perceptual.clone(prefix="test/")
            
    def find_msssim_kernel_size(self):
        for kernel_size, required_image_size in [(11, 160), (9, 128), (7, 96), (5, 64), (3, 32)]:
            if self.hparams.input_dims[1] > required_image_size and self.hparams.input_dims[2] > required_image_size:
                return kernel_size

        return None

    def forward(self, x: torch.Tensor):
        raise NotImplementedError

    def step(self, batch: Any):
        return NotImplementedError

    def generate_csi(self, x: torch.Tensor, stage: str):
        if isinstance(x, list) or isinstance(x, tuple):
            x = x[0]
        
        num_data = x.shape[0]

        csi = self.csi(num_data, x.dtype, x.device, stage=stage)

        return csi

    def training_step(self, batch: Any, batch_idx: int):
        x, _ = batch
        csi = self.generate_csi(x, stage="train")
        loss, preds, targets = self.step((x, csi))

        metrics = self.train_metrics(preds, targets)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int):
        x, _ = batch
        csi = self.generate_csi(x, stage="val")
        loss, preds, targets = self.step((x, csi))

        preds = preds.clip(0, 1)

        metrics = {**self.val_metrics(preds, targets), **self.val_metrics_perceptual(preds)}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_step(self, batch: Any, batch_idx: int):
        x, _ = batch
        csi = self.generate_csi(x, stage="test")
        loss, preds, targets = self.step((x, csi))

        preds = preds.clip(0, 1)

        metrics = {**self.test_metrics(preds, targets), **self.test_metrics_perceptual(preds)}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        """
        if batch_idx < self.hparams.log_test_batches and self.logger and hasattr(self.logger, "experiment") and hasattr(self.logger.experiment, "log_image"):

            for i in range(preds.shape[0]):
                psnr_score = peak_signal_noise_ratio(preds[i].unsqueeze(0), targets[i].unsqueeze(0), data_range=(0.0, 1.0)).item()
                
                if self.msssim_kernel_size:
                    msssim_score = ms_ssim(preds[i].unsqueeze(0), targets[i].unsqueeze(0), data_range=1.0, win_size=self.msssim_kernel_size).item()
                else:
                    msssim_score = -1.0

                ssim_score = structural_similarity_index_measure(preds[i].unsqueeze(0), targets[i].unsqueeze(0), kernel_size=11, data_range=1.0).item()
                
                self.logger.experiment.log_image(preds[i].permute(2,1,0).cpu(), image_minmax=(0,1), name=f"test/preds/{batch_idx}/{i}_psnr{psnr_score:.4f}_ssim{ssim_score:.4f}_msssim{msssim_score:.4f}.png")
                
                self.logger.experiment.log_image(targets[i].permute(2,1,0).cpu(), image_minmax=(0,1), name=f"test/targets/{batch_idx}/{i}.png")
        
        if hasattr(self.hparams, "output_dir"):
            fpath = os.path.join(self.hparams.output_dir, "test_images_snr"+str(self.csi.test_min) + "_" + str(self.csi.test_max))
            os.makedirs(fpath, exist_ok=True)
            for i in range(preds.shape[0]):
                lfpath = os.path.join(fpath, f"img_{str(i)}_{batch_idx}_{self.global_step}.png")
                torchvision.utils.save_image(preds[i].cpu(), lfpath)
        """
            
        return {"loss": loss, "preds": preds, "targets": targets}

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        return {"optimizer": optimizer}

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        for param_name in checkpoint["state_dict"]:
            if param_name.startswith(("loss", "train_metrics", "val_metrics", "val_metrics_perceptual", "test_metrics", "test_metrics_perceptual")):
                del checkpoint["state_dict"][param_name]
        """
        
        return super().on_load_checkpoint(checkpoint)
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        for param_name in checkpoint["state_dict"]:
            if param_name.startswith(("loss", "train_metrics", "val_metrics", "val_metrics_perceptual", "test_metrics", "test_metrics_perceptual")):
                del checkpoint["state_dict"][param_name]
        """
        
        return super().on_save_checkpoint(checkpoint)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = False): # changed strict=false instead of true
        
        return super().load_state_dict(state_dict, strict)