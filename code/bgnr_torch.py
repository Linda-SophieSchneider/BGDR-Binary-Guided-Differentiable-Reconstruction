"""BGNR on CUDA through torch: a line-by-line port of the MLX implementation.

The original is ``Research/BGNR/eval/bgnr.py``.  Everything that matters for the
numbers is kept identical and in the same order:

*   the objective is mean squared error over all detector elements;
*   the gradient is divided by the sensitivity preconditioner
    ``A^T A 1 / n_proj + 1e-5`` before anything else;
*   the support constraint masks the gradient *and* the AdamW moments, and the
    iterate is projected after the step -- all three, as in the reference
    implementation;
*   AdamW is hand-rolled rather than taken from ``torch.optim``, because the
    moment masking has to happen between the moment update and the step, which
    an optimizer object does not expose;
*   the amplitude box and the soft-support decay of the repaired pipeline apply
    in the same places.

The only intentional difference is the array library.  ``validate_port.py``
checks that this changes no number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class BGNRConfig:
    learning_rate: float = 1e-2
    epochs: int = 500
    weight_decay: float = 1e-2
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    ssim_weight_scale: float = 0.0        # deprecated; non-zero values are rejected
    early_stop_patience: int = 10
    early_stop_after_epoch: int = 200
    early_stop_ratio: float = 0.98
    log_every: int = 25
    snapshot_every: int = 0
    soft_decay: float = 0.0
    value_ceiling: float = 0.0


def _gaussian_kernel(window: int = 11, sigma: float = 1.5) -> np.ndarray:
    coords = np.arange(window, dtype=np.float64) - (window - 1) / 2.0
    kernel = np.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    return (kernel / kernel.sum()).astype(np.float32)


class ProjectionSSIM:
    """Mean SSIM over the views of a ``(views, u, v)`` sinogram, differentiable."""

    def __init__(self, data_range: float, window: int = 11, sigma: float = 1.5,
                 device: str = DEVICE):
        k = torch.as_tensor(_gaussian_kernel(window, sigma), device=device)
        self.kh = k.view(1, 1, window, 1)
        self.kw = k.view(1, 1, 1, window)
        self.c1 = (0.01 * data_range) ** 2
        self.c2 = (0.03 * data_range) ** 2

    def _blur(self, stack: torch.Tensor) -> torch.Tensor:
        return F.conv2d(F.conv2d(stack, self.kh), self.kw)

    def __call__(self, prediction: torch.Tensor, measured: torch.Tensor) -> torch.Tensor:
        x = prediction[:, None]
        y = measured[:, None]
        mu_x = self._blur(x)
        mu_y = self._blur(y)
        xx = self._blur(x * x) - mu_x * mu_x
        yy = self._blur(y * y) - mu_y * mu_y
        xy = self._blur(x * y) - mu_x * mu_y
        num = (2 * mu_x * mu_y + self.c1) * (2 * xy + self.c2)
        den = (mu_x * mu_x + mu_y * mu_y + self.c1) * (xx + yy + self.c2)
        return torch.mean(num / den)


def sensitivity_preconditioner(forward, backward, volume_shape, n_proj: float,
                               device: str = DEVICE) -> torch.Tensor:
    """``A^T A 1 / n_proj + 1e-5`` -- the reference code's gradient normalization."""
    ones = torch.ones(volume_shape, dtype=torch.float32, device=device) / n_proj
    return backward(forward(ones)) + 1e-5


def masked_backprojection_init(measured: torch.Tensor, forward, backward,
                               support_mask: torch.Tensor | None,
                               volume_shape) -> torch.Tensor:
    """``x0 = P_S (A^T y)``, normalized to a unit maximum as in the reference code."""
    n_proj = float(measured.shape[0])
    precond = sensitivity_preconditioner(forward, backward, volume_shape, n_proj,
                                         device=str(measured.device))
    init = backward(measured) / precond
    if support_mask is not None:
        init = init * support_mask
    peak = init.max()
    if float(peak) > 0:
        init = init / peak
    return init.detach()


def reconstruct_bgnr(
    measured: torch.Tensor,
    forward,
    backward,
    *,
    initial_volume: torch.Tensor,
    support_mask: torch.Tensor | None,
    config: BGNRConfig = BGNRConfig(),
    progress_callback=None,
    volume_callback=None,
    soft_mask: torch.Tensor | None = None,
):
    """Run BGNR and return ``(volume, history)``."""
    device = measured.device
    x = initial_volume.clone().detach().to(device=device, dtype=torch.float32)
    n_proj = float(measured.shape[0])
    precond = sensitivity_preconditioner(forward, backward, tuple(x.shape), n_proj,
                                         device=str(device))

    if config.ssim_weight_scale != 0.0:
        raise ValueError(
            "projection SSIM is disabled by the fixed evaluation protocol; "
            "use ssim_weight_scale=0.0"
        )

    beta1, beta2 = config.betas
    m = torch.zeros_like(x)
    v = torch.zeros_like(x)
    history: list[float] = []
    best_loss = float("inf")
    stall = 0

    for epoch in range(config.epochs):
        x_var = x.detach().requires_grad_(True)
        prediction = forward(x_var)
        loss = torch.mean((prediction - measured) ** 2)
        grad, = torch.autograd.grad(loss, x_var)

        with torch.no_grad():
            grad = grad / precond
            if support_mask is not None:
                grad = grad * support_mask

            m = beta1 * m + (1.0 - beta1) * grad
            v = beta2 * v + (1.0 - beta2) * (grad * grad)
            if support_mask is not None:
                m = m * support_mask
                v = v * support_mask
            m_hat = m / (1.0 - beta1 ** (epoch + 1))
            v_hat = v / (1.0 - beta2 ** (epoch + 1))
            x = x_var.detach() - config.learning_rate * (
                m_hat / (torch.sqrt(v_hat) + config.eps)
                + config.weight_decay * x_var.detach()
            )
            x = torch.clamp(x, min=0.0)
            if config.value_ceiling > 0.0:
                x = torch.clamp(x, max=config.value_ceiling)
            if support_mask is not None:
                x = x * support_mask
            if soft_mask is not None and config.soft_decay > 0.0:
                x = x * (1.0 - config.soft_decay * (1.0 - soft_mask))

        value = float(loss.detach())
        history.append(value)
        if progress_callback is not None and epoch % config.log_every == 0:
            progress_callback(epoch, value)
        if (volume_callback is not None and config.snapshot_every
                and epoch % config.snapshot_every == 0):
            volume_callback(epoch, x)

        if epoch > config.early_stop_after_epoch:
            if value > best_loss * config.early_stop_ratio:
                stall += 1
                if stall > config.early_stop_patience:
                    if progress_callback is not None:
                        progress_callback(epoch, value)
                    break
            else:
                stall = 0
                best_loss = value
        else:
            best_loss = min(best_loss, value)

    return x.detach(), history
