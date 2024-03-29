import torch
from base_class.math.statistics import Variance


class BatchNorm(Variance[torch.Tensor]):
    _t = torch.float32  # round-off error: ~1e-7
    _k = {"dim": 0, "keepdim": True, "dtype": _t}

    @classmethod
    @torch.no_grad()
    def calculate(cls, data: torch.Tensor, weight: torch.Tensor = None):
        sumw = cls._t(len(data)) if weight is None else weight.sum(dtype=cls._t)
        for _ in range(len(data.shape) - 1):
            weight = weight.unsqueeze(-1)
        m1 = (
            data.mean(**cls._k)
            if weight is None
            else ((data * weight).sum(**cls._k) / sumw)
        )
        M2 = (
            (data - m1).pow(2).sum(**cls._k)
            if weight is None
            else ((data - m1).pow(2) * weight).sum(**cls._k)
        )
        return sumw, m1, M2
