import copy
from typing_extensions import Literal
from pytorch_msssim import ms_ssim
from torch import Tensor
import torch
from torchmetrics import MeanMetric, MetricCollection
from typing import Any, Dict
from torchmetrics.image.inception import InceptionScore, Tuple
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.functional.image.lpips import _LPIPS, _lpips_compute, _lpips_update, _NoTrainLpips
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

class MultiScaleSSIM(MeanMetric):

    def __init__(self, kernel_size=11, data_range=1.0, **kwargs: Any) -> None:
        super().__init__("warn", **kwargs)

        self.kernel_size = kernel_size
        self.data_range = data_range
    
    def update(self, preds, targets) -> None:

        value = ms_ssim(preds, targets, data_range=1.0, size_average=False, win_size=self.kernel_size)

        return super().update(value, 1)

class CustomMultiScaleSSIM(MeanMetric):

    def __init__(self, kernel_size=11, data_range=1.0, **kwargs: Any) -> None:
        super().__init__("warn", **kwargs)

        self.kernel_size = kernel_size
        self.data_range = data_range
        self.add_state("sum_sq", torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds, targets) -> None:

        value = ms_ssim(preds, targets, data_range=1.0, size_average=False, win_size=self.kernel_size)

        self.sum_sq += (value**2).sum()
        
        return super().update(value, 1)
    
    def compute(self) -> Tensor:
        mean = super().compute()
        
        std = torch.sqrt((self.sum_sq / self.weight) - mean**2)

        return mean, std
        
class MeanInceptionScore(InceptionScore):

    def compute(self) -> Tuple[Tensor, Tensor]:
        return super().compute()[0]
    
class CustomInceptionScore(InceptionScore):
    pass

class CustomLearnedPerceptualImagePatchSimilarity(LearnedPerceptualImagePatchSimilarity):
    
    def __init__(self, net_type: Literal['vgg', 'alex', 'squeeze'] = "alex", normalize: bool = False, **kwargs: Any) -> None:
        super().__init__(net_type, normalize=normalize, reduction="mean", **kwargs)
        
        self.add_state("sum_sq", torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, img1: Tensor, img2: Tensor) -> None:
        """Update internal states with lpips score."""
        loss, total = _lpips_update(img1, img2, net=self.net, normalize=self.normalize)
        
        self.sum_scores += loss.sum()
        self.total += total
        
        self.sum_sq += (loss ** 2).sum()

    def compute(self) -> Tensor:
        """Compute final perceptual similarity metric."""
        mean = _lpips_compute(self.sum_scores, self.total, "mean")
        std = torch.sqrt((self.sum_sq / self.total) - mean**2)
        
        return mean, std
    
class CustomPeakSignalNoiseRatio(PeakSignalNoiseRatio):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, reduction=None, dim=[1,2,3], **kwargs)

    def compute(self):
        res_per_sample = super().compute()
        return res_per_sample.mean(), res_per_sample.std()

class CustomStructuralSimilarityIndexMeasure(StructuralSimilarityIndexMeasure):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, reduction=None, **kwargs)

    def compute(self):
        res_per_sample = super().compute()
        
        return res_per_sample.mean(), res_per_sample.std()

class BPP(MeanMetric):

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("warn", **kwargs)

        self.add_state("sum_sq", torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, inp_size, y_num_bits) -> None:

        num_pixels = (inp_size[2] * inp_size[3])
        
        value = y_num_bits.sum() * 8 / (num_pixels*inp_size[0])
                
        self.sum_sq += (value**2).sum()
        
        return super().update(value, 1)
    
    def compute(self) -> Tensor:
        mean = super().compute()
        
        std = torch.sqrt((self.sum_sq / self.weight) - mean**2)

        return mean, std

class CustomMetricCollection(MetricCollection):
    
    def reduce_to_std(self,res) -> Dict[str, Any]:
        
        new_res = {}
        for k, v in res.items():
            if isinstance(v, tuple):
                new_res[k] = v[0]
                new_res[k + "_std"] = v[1]
            else:
                new_res[k] = v
        
        return new_res
        
    def compute(self) -> Dict[str, Any]:
        res = super().compute()
    
        return self.reduce_to_std(res)

    def forward(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        res = self._compute_and_reduce("forward", *args, **kwargs)
        
        return self.reduce_to_std(res)
