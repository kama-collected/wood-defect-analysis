import math
from typing import Iterable, Optional, Tuple, Dict, Type

import torch
from torch.optim import Optimizer


# ============================================================
# Section 1 — Utility Functions
# ============================================================

def _safe_norm(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x.norm().clamp_min(eps)


def _agc_clip_(p: torch.Tensor, g: torch.Tensor, clip: float = 0.01, eps: float = 1e-3) -> torch.Tensor:
    p_norm = torch.norm(p.detach()).clamp_min(eps)
    g_norm = torch.norm(g.detach()).clamp_min(eps)
    max_norm = clip * p_norm
    if g_norm > max_norm:
        g = g * (max_norm / g_norm)
    return g


def _centralize_gradient(g: torch.Tensor) -> torch.Tensor:
    if g.ndim <= 1:
        return g
    dims = tuple(range(1, g.ndim))
    return g - g.mean(dim=dims, keepdim=True)


def _newton_schulz_orthogonalize(g: torch.Tensor, steps: int = 3, eps: float = 1e-7) -> torch.Tensor:
    if g.ndim < 2:
        return g
    orig_shape = g.shape
    x = g.reshape(g.shape[0], -1)
    x = x / (_safe_norm(x) + eps)
    m, n = x.shape
    if m <= n:
        a = x @ x.T
        for _ in range(steps):
            a = 1.5 * a - 0.5 * a @ a @ a
        x = a @ x
    else:
        a = x.T @ x
        for _ in range(steps):
            a = 1.5 * a - 0.5 * a @ a @ a
        x = x @ a
    return x.reshape(orig_shape)


def _spectral_norm_estimate(x: torch.Tensor, power_iters: int = 1, eps: float = 1e-8) -> torch.Tensor:
    if x.ndim < 2:
        return _safe_norm(x, eps)
    w = x.reshape(x.shape[0], -1)
    u = torch.randn(w.shape[0], device=w.device, dtype=w.dtype)
    u = u / (u.norm() + eps)
    for _ in range(max(1, power_iters)):
        v = w.T @ u
        v = v / (v.norm() + eps)
        u = w @ v
        u = u / (u.norm() + eps)
    sigma = u @ (w @ v)
    return sigma.abs().clamp_min(eps)


def _bias_correction(beta: float, step: int) -> float:
    return 1.0 - beta ** step


def _match_key(name: str) -> str:
    return str(name).lower().replace("-", "").replace("_", "")


# ============================================================
# Section 2 — Optimizer Classes
# ============================================================

class OrthoMuonW(Optimizer):
    """Momentum optimizer with orthogonalized updates for 2D+ tensors and decoupled weight decay."""

    def __init__(self, params, lr=1e-3, momentum=0.9, weight_decay=1e-4,
                 muon_ratio=0.5, ns_steps=3, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        muon_ratio=muon_ratio, ns_steps=ns_steps, nesterov=nesterov)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta = group["momentum"]
            wd = group["weight_decay"]
            alpha = group["muon_ratio"]
            ns_steps = group["ns_steps"]
            nesterov = group["nesterov"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                if wd != 0:
                    p.mul_(1 - lr * wd)
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.mul_(beta).add_(g, alpha=1 - beta)
                update = g.add(buf, alpha=beta) if nesterov else buf
                if p.ndim >= 2:
                    ortho = _newton_schulz_orthogonalize(update, steps=ns_steps)
                    final = alpha * ortho + (1 - alpha) * update
                else:
                    final = update
                p.add_(final, alpha=-lr)
        return loss


class LookaheadAdamW_GC_AGC(Optimizer):
    """AdamW + Gradient Centralization + Adaptive Gradient Clipping + Lookahead."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2,
                 agc_clip=0.01, agc_eps=1e-3, lookahead_k=5, lookahead_alpha=0.5, gc=True):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        agc_clip=agc_clip, agc_eps=agc_eps, lookahead_k=lookahead_k,
                        lookahead_alpha=lookahead_alpha, gc=gc)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            clip = group["agc_clip"]
            agc_eps = group["agc_eps"]
            k = group["lookahead_k"]
            la = group["lookahead_alpha"]
            gc = group["gc"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                if gc:
                    g = _centralize_gradient(g)
                if clip is not None and clip > 0:
                    p_norm = _safe_norm(p)
                    g_norm = _safe_norm(g)
                    max_norm = (p_norm + agc_eps) * clip
                    if g_norm > max_norm:
                        g = g * (max_norm / g_norm)
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["slow_param"] = p.detach().clone()
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]
                if wd != 0:
                    p.mul_(1 - lr * wd)
                exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                bias_c1 = _bias_correction(beta1, t)
                bias_c2 = _bias_correction(beta2, t)
                step_size = lr / bias_c1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_c2)).add_(eps)
                p.addcdiv_(exp_avg, denom, value=-step_size)
                if t % k == 0:
                    slow = state["slow_param"]
                    slow.add_(p - slow, alpha=la)
                    p.copy_(slow)
        return loss


class ScaleAwareSGDW(Optimizer):
    """SGD with decoupled weight decay, trust-ratio scaling, and layer-size scaling."""

    def __init__(self, params, lr=1e-2, momentum=0.9, weight_decay=1e-4, nesterov=True,
                 trust_clip=(0.5, 5.0), gamma=0.25, ref_numel=4096, eps=1e-8):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov,
                        trust_clip=trust_clip, gamma=gamma, ref_numel=ref_numel, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta = group["momentum"]
            wd = group["weight_decay"]
            nesterov = group["nesterov"]
            min_t, max_t = group["trust_clip"]
            gamma = group["gamma"]
            ref_numel = group["ref_numel"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                if wd != 0:
                    p.mul_(1 - lr * wd)
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.mul_(beta).add_(g)
                update = g.add(buf, alpha=beta) if nesterov else buf
                p_norm = _safe_norm(p)
                u_norm = _safe_norm(update)
                trust = (p_norm / (u_norm + eps)).clamp(min=min_t, max=max_t)
                layer_scale = (float(p.numel()) / float(ref_numel)) ** gamma
                p.add_(update, alpha=-lr * float(trust) * layer_scale)
        return loss


class CurvatureAwareAdamP(Optimizer):
    """AdamW + projection-style update + optional AGC."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        delta: float = 0.1,
        wd_ratio: float = 0.1,
        agc_clip: Optional[float] = None,
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            delta=delta, wd_ratio=wd_ratio, agc_clip=agc_clip
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            delta = group["delta"]
            wd_ratio = group["wd_ratio"]
            agc_clip = group["agc_clip"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                if g.is_sparse:
                    raise RuntimeError("CurvatureAwareAdamP does not support sparse gradients.")
                if agc_clip is not None:
                    g = _agc_clip_(p, g, agc_clip)
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]
                exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                bias_c1 = _bias_correction(beta1, step)
                bias_c2 = _bias_correction(beta2, step)
                denom = exp_avg_sq.sqrt() / math.sqrt(bias_c2)
                denom.add_(eps)
                update = (exp_avg / bias_c1) / denom
                cur_wd = wd
                if p.ndim > 1:
                    w_flat = p.view(p.shape[0], -1)
                    u_flat = update.view(update.shape[0], -1)
                    cos = torch.nn.functional.cosine_similarity(w_flat, u_flat, dim=1).abs()
                    if cos.mean().item() < delta:
                        w_norm_sq = (w_flat * w_flat).sum(dim=1, keepdim=True).clamp_min(1e-12)
                        proj = ((u_flat * w_flat).sum(dim=1, keepdim=True) / w_norm_sq) * w_flat
                        u_flat = u_flat - proj
                        update = u_flat.view_as(update)
                        cur_wd = wd * wd_ratio
                if cur_wd != 0:
                    p.mul_(1 - lr * cur_wd)
                p.add_(update, alpha=-lr)
        return loss


class FocalMomentumSGD(Optimizer):
    """SGD with momentum + EMA of gradient magnitude for difficulty-aware scaling."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-2,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        gamma: float = 0.5,
        ema_beta: float = 0.95,
        scale_clip: Tuple[float, float] = (0.75, 1.5),
        nesterov: bool = False,
    ):
        defaults = dict(
            lr=lr, momentum=momentum, weight_decay=weight_decay, gamma=gamma,
            ema_beta=ema_beta, scale_clip=scale_clip, nesterov=nesterov
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
            gamma = group["gamma"]
            ema_beta = group["ema_beta"]
            scale_min, scale_max = group["scale_clip"]
            nesterov = group["nesterov"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                if g.is_sparse:
                    raise RuntimeError("FocalMomentumSGD does not support sparse gradients.")
                state = self.state[p]
                if len(state) == 0:
                    state["buf"] = torch.zeros_like(p)
                    state["g_ema"] = torch.zeros(1, device=p.device, dtype=p.dtype)
                if wd != 0:
                    p.mul_(1 - lr * wd)
                g_mag = g.abs().mean()
                g_ema = state["g_ema"]
                g_ema.mul_(ema_beta).add_(g_mag, alpha=1 - ema_beta)
                ratio = (g_mag / g_ema.clamp_min(1e-8)).pow(gamma).item()
                ratio = max(scale_min, min(scale_max, ratio))
                buf = state["buf"]
                buf.mul_(momentum).add_(g)
                update = g.add(buf, alpha=momentum) if nesterov else buf
                p.add_(update, alpha=-lr * ratio)
        return loss


class RangerMuonLite(Optimizer):
    """RAdam + Lookahead + optional orthogonalization for 2D+ tensors."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.95, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        alpha: float = 0.5,
        k: int = 6,
        muon_ratio: float = 0.25,
        ns_steps: int = 3,
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            alpha=alpha, k=k, muon_ratio=muon_ratio, ns_steps=ns_steps
        )
        super().__init__(params, defaults)

    @staticmethod
    def _orthogonalize(t: torch.Tensor, ns_steps: int = 3) -> torch.Tensor:
        return _newton_schulz_orthogonalize(t, steps=ns_steps)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            alpha = group["alpha"]
            k = group["k"]
            muon_ratio = group["muon_ratio"]
            ns_steps = group["ns_steps"]
            rho_inf = 2 / (1 - beta2) - 1
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                if g.is_sparse:
                    raise RuntimeError("RangerMuonLite does not support sparse gradients.")
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["slow_buffer"] = p.detach().clone()
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]
                exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                if wd != 0:
                    p.mul_(1 - lr * wd)
                beta2_t = beta2 ** step
                rho_t = rho_inf - 2 * step * beta2_t / (1 - beta2_t + 1e-12)
                if rho_t > 4:
                    r_t = math.sqrt(
                        ((rho_t - 4) * (rho_t - 2) * rho_inf) /
                        ((rho_inf - 4) * (rho_inf - 2) * rho_t + 1e-12)
                    )
                    denom = exp_avg_sq.sqrt().add_(eps)
                    update = exp_avg / denom * r_t
                else:
                    update = exp_avg
                if muon_ratio > 0 and p.ndim >= 2:
                    ortho = self._orthogonalize(update, ns_steps=ns_steps)
                    update = (1 - muon_ratio) * update + muon_ratio * ortho
                p.add_(update, alpha=-lr)
                if step % k == 0:
                    slow = state["slow_buffer"]
                    slow.add_(p - slow, alpha=alpha)
                    p.copy_(slow)
        return loss


class AdaptiveRangerMuonLitePP(Optimizer):
    """
    Adaptive RangerMuonLite++:
    RAdam + Lookahead + orthogonalized updates with internal adaptation of beta1,
    muon_ratio, lookahead alpha, and lookahead k.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.95, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-4,
        alpha: float = 0.5,
        k: int = 6,
        muon_ratio: float = 0.25,
        ns_steps: int = 3,
        total_steps: int = 10000,
        muon_min_ratio: float = 0.05,
        alpha_max: float = 0.8,
        k_max: int = 12,
        beta1_min: float = 0.85,
        adapt_beta1: bool = True,
        adapt_muon: bool = True,
        adapt_alpha: bool = True,
        adapt_k: bool = True,
        grad_ema_beta: float = 0.95,
        noise_beta: float = 0.9,
        noise_threshold: float = 1.5,
        stability_eps: float = 1e-8,
        muon_boost: float = 1.05,
        muon_decay: float = 0.98,
        agc_clip: Optional[float] = None,
        agc_eps: float = 1e-3,
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            alpha=alpha, k=k, muon_ratio=muon_ratio, ns_steps=ns_steps,
            total_steps=total_steps, muon_min_ratio=muon_min_ratio, alpha_max=alpha_max, k_max=k_max,
            beta1_min=beta1_min, adapt_beta1=adapt_beta1, adapt_muon=adapt_muon,
            adapt_alpha=adapt_alpha, adapt_k=adapt_k, grad_ema_beta=grad_ema_beta,
            noise_beta=noise_beta, noise_threshold=noise_threshold, stability_eps=stability_eps,
            muon_boost=muon_boost, muon_decay=muon_decay, agc_clip=agc_clip, agc_eps=agc_eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            beta1_init, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            alpha_init = group["alpha"]
            k_init = group["k"]
            muon_init = group["muon_ratio"]
            ns_steps = group["ns_steps"]
            total_steps = group["total_steps"]
            muon_min_ratio = group["muon_min_ratio"]
            alpha_max = group["alpha_max"]
            k_max = group["k_max"]
            beta1_min = group["beta1_min"]
            adapt_beta1 = group["adapt_beta1"]
            adapt_muon = group["adapt_muon"]
            adapt_alpha = group["adapt_alpha"]
            adapt_k = group["adapt_k"]
            grad_ema_beta = group["grad_ema_beta"]
            noise_beta = group["noise_beta"]
            noise_threshold = group["noise_threshold"]
            stability_eps = group["stability_eps"]
            muon_boost = group["muon_boost"]
            muon_decay = group["muon_decay"]
            agc_clip = group["agc_clip"]
            agc_eps = group["agc_eps"]
            rho_inf = 2.0 / (1.0 - beta2) - 1.0
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                if g.is_sparse:
                    raise RuntimeError("AdaptiveRangerMuonLitePP does not support sparse gradients.")
                if agc_clip is not None and agc_clip > 0:
                    g = _agc_clip_(p, g, clip=agc_clip, eps=agc_eps)
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["slow_buffer"] = p.detach().clone()
                    state["grad_ema"] = torch.zeros(1, device=p.device, dtype=p.dtype)
                    state["grad_var_ema"] = torch.zeros(1, device=p.device, dtype=p.dtype)
                    state["noise_ema"] = torch.zeros(1, device=p.device, dtype=p.dtype)
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                slow = state["slow_buffer"]
                grad_ema = state["grad_ema"]
                grad_var_ema = state["grad_var_ema"]
                noise_ema = state["noise_ema"]
                state["step"] += 1
                step = state["step"]
                progress = min(1.0, float(step) / float(max(1, total_steps)))
                beta1 = beta1_init - (beta1_init - beta1_min) * progress if adapt_beta1 else beta1_init
                muon_ratio = muon_init * (1.0 - progress) + muon_min_ratio * progress if adapt_muon else muon_init
                alpha = alpha_init + (alpha_max - alpha_init) * progress if adapt_alpha else alpha_init
                k = max(1, int(round(k_init + (k_max - k_init) * progress))) if adapt_k else k_init
                g_mag = g.abs().mean()
                grad_ema.mul_(grad_ema_beta).add_(g_mag, alpha=1.0 - grad_ema_beta)
                centered = g - g.mean()
                g_var = centered.pow(2).mean()
                grad_var_ema.mul_(grad_ema_beta).add_(g_var, alpha=1.0 - grad_ema_beta)
                noise_ratio = (g_var.sqrt() / grad_ema.clamp_min(stability_eps)).clamp_min(0.0)
                noise_ema.mul_(noise_beta).add_(noise_ratio, alpha=1.0 - noise_beta)
                if noise_ema.item() > noise_threshold:
                    muon_ratio = min(1.0, muon_ratio * muon_boost)
                else:
                    muon_ratio = max(muon_min_ratio, muon_ratio * muon_decay)
                exp_avg.mul_(beta1).add_(g, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
                if wd != 0:
                    p.mul_(1.0 - lr * wd)
                beta2_t = beta2 ** step
                rho_t = rho_inf - 2.0 * step * beta2_t / (1.0 - beta2_t + 1e-12)
                if rho_t > 4.0:
                    r_t = math.sqrt(
                        ((rho_t - 4.0) * (rho_t - 2.0) * rho_inf) /
                        ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t + 1e-12)
                    )
                    denom = exp_avg_sq.sqrt().add_(eps)
                    update = (exp_avg / denom) * r_t
                else:
                    update = exp_avg
                if muon_ratio > 0.0 and p.ndim >= 2:
                    ortho = _newton_schulz_orthogonalize(update, steps=ns_steps)
                    update = (1.0 - muon_ratio) * update + muon_ratio * ortho
                p.add_(update, alpha=-lr)
                if step % k == 0:
                    slow.add_(p - slow, alpha=alpha)
                    p.copy_(slow)
        return loss


class LionTrustW(Optimizer):
    """Lion-style sign updates with trust-ratio scaling and decoupled weight decay."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.95, 0.98),
        weight_decay: float = 1e-2,
        trust_clip: Tuple[float, float] = (0.25, 4.0),
        agc_clip: Optional[float] = None,
        eps: float = 1e-8,
    ):
        defaults = dict(
            lr=lr, betas=betas, weight_decay=weight_decay,
            trust_clip=trust_clip, agc_clip=agc_clip, eps=eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]
            trust_lo, trust_hi = group["trust_clip"]
            agc_clip = group["agc_clip"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                if g.is_sparse:
                    raise RuntimeError("LionTrustW does not support sparse gradients.")
                if agc_clip is not None:
                    g = _agc_clip_(p, g, clip=agc_clip)
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]
                if wd != 0:
                    p.mul_(1 - lr * wd)
                update = exp_avg.mul(beta1).add(g, alpha=1 - beta1)
                signed = update.sign()
                p_norm = _safe_norm(p, eps)
                u_norm = _safe_norm(signed, eps)
                trust = (p_norm / u_norm).clamp(trust_lo, trust_hi)
                p.add_(signed, alpha=-lr * float(trust))
                exp_avg.mul_(beta2).add_(g, alpha=1 - beta2)
        return loss


class SpectralSGDW(Optimizer):
    """Momentum SGD with decoupled weight decay and spectral-normalized layer updates."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-2,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        nesterov: bool = True,
        power_iters: int = 1,
        spectral_scale: float = 1.0,
        eps: float = 1e-8,
    ):
        defaults = dict(
            lr=lr, momentum=momentum, weight_decay=weight_decay,
            nesterov=nesterov, power_iters=power_iters,
            spectral_scale=spectral_scale, eps=eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            wd = group["weight_decay"]
            nesterov = group["nesterov"]
            power_iters = group["power_iters"]
            spectral_scale = group["spectral_scale"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                if wd != 0:
                    p.mul_(1 - lr * wd)
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.mul_(mu).add_(g)
                update = g.add(buf, alpha=mu) if nesterov else buf
                if p.ndim >= 2:
                    sigma = _spectral_norm_estimate(update, power_iters=power_iters, eps=eps)
                    update = update / (sigma / spectral_scale + eps)
                else:
                    update = update / (_safe_norm(update, eps) + eps)
                p.add_(update, alpha=-lr)
        return loss


class LambGCW(Optimizer):
    """LAMB-style AdamW with gradient centralization for stable CNN training."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 1e-2,
        trust_clip: Tuple[float, float] = (0.01, 10.0),
        gc: bool = True,
        agc_clip: Optional[float] = None,
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            trust_clip=trust_clip, gc=gc, agc_clip=agc_clip,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            gc = group["gc"]
            agc_clip = group["agc_clip"]
            trust_lo, trust_hi = group["trust_clip"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                if g.is_sparse:
                    raise RuntimeError("LambGCW does not support sparse gradients.")
                if gc:
                    g = _centralize_gradient(g)
                if agc_clip is not None:
                    g = _agc_clip_(p, g, clip=agc_clip)
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]
                exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                bias_c1 = _bias_correction(beta1, step)
                bias_c2 = _bias_correction(beta2, step)
                adam_update = (exp_avg / bias_c1) / (exp_avg_sq.sqrt() / math.sqrt(bias_c2) + eps)
                if wd != 0:
                    adam_update = adam_update.add(p, alpha=wd)
                w_norm = _safe_norm(p)
                u_norm = _safe_norm(adam_update)
                trust = (w_norm / u_norm).clamp(trust_lo, trust_hi)
                p.add_(adam_update, alpha=-lr * float(trust))
        return loss


class GradShiftMuAdam(Optimizer):
    """
    Adam-style optimizer with gradient-shift gating and Muon-style orthogonal refinement.
    A variance-gated blend controls when shifted-gradient information dominates.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-4,
        shift_beta: float = 0.9,
        shift_scale: float = 0.5,
        var_gate: float = 1.0,
        muon_ratio: float = 0.15,
        ns_steps: int = 3,
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            shift_beta=shift_beta, shift_scale=shift_scale, var_gate=var_gate,
            muon_ratio=muon_ratio, ns_steps=ns_steps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            shift_beta = group["shift_beta"]
            shift_scale = group["shift_scale"]
            var_gate = group["var_gate"]
            muon_ratio = group["muon_ratio"]
            ns_steps = group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                if g.is_sparse:
                    raise RuntimeError("GradShiftMuAdam does not support sparse gradients.")
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["prev_grad"] = torch.zeros_like(p)
                    state["shift_ema"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                prev_grad = state["prev_grad"]
                shift_ema = state["shift_ema"]
                state["step"] += 1
                step = state["step"]
                shift = g - prev_grad
                shift_ema.mul_(shift_beta).add_(shift, alpha=1 - shift_beta)
                prev_grad.copy_(g)
                g_var = shift.pow(2).mean()
                gate = float(g_var / (g_var + var_gate + 1e-12))
                blended = g + shift_scale * gate * shift_ema
                exp_avg.mul_(beta1).add_(blended, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(blended, blended, value=1 - beta2)
                if wd != 0:
                    p.mul_(1 - lr * wd)
                bc1 = _bias_correction(beta1, step)
                bc2 = _bias_correction(beta2, step)
                update = (exp_avg / bc1) / (exp_avg_sq.sqrt() / math.sqrt(bc2) + eps)
                if muon_ratio > 0 and p.ndim >= 2:
                    ortho = _newton_schulz_orthogonalize(update, steps=ns_steps)
                    update = (1 - muon_ratio) * update + muon_ratio * ortho
                p.add_(update, alpha=-lr)
        return loss


class TexAdamTrust(Optimizer):
    """
    Texture-aware Adam with trust-ratio scaling and spectral regularization of updates.
    Designed for highly repetitive or texture-dominant visual patterns.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-4,
        trust_clip: Tuple[float, float] = (0.05, 10.0),
        texture_scale: float = 1.0,
        spectral_mix: float = 0.25,
        gc: bool = True,
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            trust_clip=trust_clip, texture_scale=texture_scale,
            spectral_mix=spectral_mix, gc=gc,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            trust_lo, trust_hi = group["trust_clip"]
            texture_scale = group["texture_scale"]
            spectral_mix = group["spectral_mix"]
            gc = group["gc"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                if gc:
                    g = _centralize_gradient(g)
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]
                exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                if wd != 0:
                    p.mul_(1 - lr * wd)
                bc1 = _bias_correction(beta1, step)
                bc2 = _bias_correction(beta2, step)
                update = (exp_avg / bc1) / (exp_avg_sq.sqrt() / math.sqrt(bc2) + eps)
                if p.ndim >= 2:
                    sigma = _spectral_norm_estimate(update, power_iters=1, eps=eps)
                    update = (1 - spectral_mix) * update + spectral_mix * (update / (sigma / texture_scale + eps))
                trust = (_safe_norm(p) / _safe_norm(update)).clamp(trust_lo, trust_hi)
                p.add_(update, alpha=-lr * float(trust))
        return loss


class MedStableAdam(Optimizer):
    """
    Medical-domain stability-focused AdamW variant with GC, AGC, and adaptive variance damping.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 8e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-4,
        gc: bool = True,
        agc_clip: Optional[float] = 0.01,
        variance_floor: float = 1e-6,
        damping: float = 0.1,
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            gc=gc, agc_clip=agc_clip, variance_floor=variance_floor, damping=damping,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            gc = group["gc"]
            agc_clip = group["agc_clip"]
            variance_floor = group["variance_floor"]
            damping = group["damping"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                if gc:
                    g = _centralize_gradient(g)
                if agc_clip is not None:
                    g = _agc_clip_(p, g, clip=agc_clip)
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["var_ema"] = torch.zeros(1, device=p.device, dtype=p.dtype)
                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                var_ema = state["var_ema"]
                grad_var = (g - g.mean()).pow(2).mean()
                var_ema.mul_(0.95).add_(grad_var, alpha=0.05)
                exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                if wd != 0:
                    p.mul_(1 - lr * wd)
                bc1 = _bias_correction(beta1, step)
                bc2 = _bias_correction(beta2, step)
                denom = exp_avg_sq.sqrt() / math.sqrt(bc2)
                denom = denom + eps + damping * var_ema.sqrt().clamp_min(variance_floor)
                update = (exp_avg / bc1) / denom
                p.add_(update, alpha=-lr)
        return loss


class LossSwitchMuAdam(Optimizer):
    """
    Loss-aware optimizer that switches smoothly between Adam-like and Muon-like refinement.
    If closure is available, it uses the current loss; otherwise it falls back to gradient-stagnation signals.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-4,
        switch_beta: float = 0.9,
        loss_threshold: float = 0.02,
        muon_ratio_min: float = 0.05,
        muon_ratio_max: float = 0.4,
        ns_steps: int = 3,
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            switch_beta=switch_beta, loss_threshold=loss_threshold,
            muon_ratio_min=muon_ratio_min, muon_ratio_max=muon_ratio_max, ns_steps=ns_steps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss_tensor = closure() if closure is not None else None
        current_loss = float(loss_tensor.detach().item()) if isinstance(loss_tensor, torch.Tensor) else None
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            switch_beta = group["switch_beta"]
            loss_threshold = group["loss_threshold"]
            mu_lo = group["muon_ratio_min"]
            mu_hi = group["muon_ratio_max"]
            ns_steps = group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["loss_ema"] = torch.tensor(0.0, device=p.device, dtype=p.dtype)
                    state["grad_ema"] = torch.tensor(0.0, device=p.device, dtype=p.dtype)
                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                loss_ema = state["loss_ema"]
                grad_ema = state["grad_ema"]
                gmag = g.abs().mean()
                grad_ema.mul_(switch_beta).add_(gmag, alpha=1 - switch_beta)
                if current_loss is not None:
                    cl = torch.tensor(current_loss, device=p.device, dtype=p.dtype)
                    loss_ema.mul_(switch_beta).add_(cl, alpha=1 - switch_beta)
                    stagnation = (cl - loss_ema).abs().item()
                else:
                    stagnation = abs(float((gmag / grad_ema.clamp_min(1e-8)).item() - 1.0))
                muon_ratio = mu_hi if stagnation < loss_threshold else mu_lo
                exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                if wd != 0:
                    p.mul_(1 - lr * wd)
                bc1 = _bias_correction(beta1, step)
                bc2 = _bias_correction(beta2, step)
                update = (exp_avg / bc1) / (exp_avg_sq.sqrt() / math.sqrt(bc2) + eps)
                if p.ndim >= 2:
                    ortho = _newton_schulz_orthogonalize(update, steps=ns_steps)
                    update = (1 - muon_ratio) * update + muon_ratio * ortho
                p.add_(update, alpha=-lr)
        return loss_tensor


class LayerTrustScaleAdam(Optimizer):
    """
    Layer-wise AdamW with trust-ratio scaling and explicit per-layer size adaptation.
    Useful when feature scales differ substantially across the backbone/head.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-4,
        trust_clip: Tuple[float, float] = (0.05, 10.0),
        gamma: float = 0.5,
        ref_numel: float = 1e6,
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            trust_clip=trust_clip, gamma=gamma, ref_numel=ref_numel,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            trust_lo, trust_hi = group["trust_clip"]
            gamma = group["gamma"]
            ref_numel = group["ref_numel"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                if wd != 0:
                    p.mul_(1 - lr * wd)
                bc1 = _bias_correction(beta1, step)
                bc2 = _bias_correction(beta2, step)
                update = (exp_avg / bc1) / (exp_avg_sq.sqrt() / math.sqrt(bc2) + eps)
                trust = (_safe_norm(p) / _safe_norm(update)).clamp(trust_lo, trust_hi)
                layer_scale = (float(p.numel()) / float(ref_numel)) ** gamma
                p.add_(update, alpha=-lr * float(trust) * layer_scale)
        return loss


class ConsensusDriftAdam(Optimizer):
    """
    AdamW variant with consensus and drift modeling:
    - consensus term = EMA-smoothed gradient direction
    - drift term = discrepancy between current gradient and consensus
    - update blends both adaptively
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-4,
        consensus_beta: float = 0.95,
        drift_scale: float = 0.25,
        drift_gate: float = 1.0,
        gc: bool = True,
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            consensus_beta=consensus_beta, drift_scale=drift_scale,
            drift_gate=drift_gate, gc=gc,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            consensus_beta = group["consensus_beta"]
            drift_scale = group["drift_scale"]
            drift_gate = group["drift_gate"]
            gc = group["gc"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                if gc:
                    g = _centralize_gradient(g)
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["consensus"] = torch.zeros_like(p)
                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                consensus = state["consensus"]
                consensus.mul_(consensus_beta).add_(g, alpha=1 - consensus_beta)
                drift = g - consensus
                drift_energy = drift.pow(2).mean()
                gate = float((drift_energy / (drift_energy + drift_gate + 1e-12)).item())
                mixed = consensus + drift_scale * gate * drift
                exp_avg.mul_(beta1).add_(mixed, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(mixed, mixed, value=1 - beta2)
                if wd != 0:
                    p.mul_(1 - lr * wd)
                bc1 = _bias_correction(beta1, step)
                bc2 = _bias_correction(beta2, step)
                update = (exp_avg / bc1) / (exp_avg_sq.sqrt() / math.sqrt(bc2) + eps)
                p.add_(update, alpha=-lr)
        return loss


# ============================================================
# Section 3 — Optimizer Registry and Factory
# ============================================================

OPTIMIZER_REGISTRY: Dict[str, Type[Optimizer]] = {
    # Existing / earlier optimizers
    "orthomuonw": OrthoMuonW,
    "lookaheadadamw_gc_agc": LookaheadAdamW_GC_AGC,
    "scaleawaresgdw": ScaleAwareSGDW,
    "curvatureawareadamp": CurvatureAwareAdamP,
    "focalmomentumsgd": FocalMomentumSGD,
    "rangermuonlite": RangerMuonLite,
    "adaptiverangermuonlitepp": AdaptiveRangerMuonLitePP,
    "armlpp": AdaptiveRangerMuonLitePP,
    "liontrustw": LionTrustW,
    "spectralsgdw": SpectralSGDW,
    "lambgcw": LambGCW,

    # Newly proposed optimizers
    "gradshift_muadam": GradShiftMuAdam,
    "texadam_trust": TexAdamTrust,
    "medstable_adam": MedStableAdam,
    "lossswitch_muadam": LossSwitchMuAdam,
    "layertrust_scaleadam": LayerTrustScaleAdam,
    "consensusdrift_adam": ConsensusDriftAdam,
}


def available_optimizers():
    return sorted(OPTIMIZER_REGISTRY.keys())


def build_optimizer(name: str, params, **kwargs):
    key = _match_key(name)
    registry = {_match_key(k): v for k, v in OPTIMIZER_REGISTRY.items()}
    if key not in registry:
        raise ValueError(f"Unknown custom optimizer: {name}. Available: {list(OPTIMIZER_REGISTRY)}")
    return registry[key](params, **kwargs)


# ============================================================
# Section 4 — Minimal Smoke Test
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(0)
    model = torch.nn.Linear(10, 1)
    x = torch.randn(4, 10)
    y = model(x).sum()

    # Required example from the prompt
    optimizer = build_optimizer("gradshift_muadam", model.parameters())
    y.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # Broader smoke test across all optimizers
    for name in available_optimizers():
        model = torch.nn.Linear(10, 1)
        optimizer = build_optimizer(name, model.parameters())
        x = torch.randn(4, 10)
        y = model(x).sum()
        y.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    print("Smoke test passed for:", ", ".join(available_optimizers()))
