import math
from typing import Optional

import torch
from sklearn.cluster import KMeans
from torch.optim import Optimizer


class ClusterCoupledAdamW(Optimizer):
    """
    AdamW variant that periodically couples per-row gradients through K-Means clustering.
    Inspired by "Better Embeddings with Coupled Adam" (Stollenwerk & Stollenwerk, 2025).

    For parameter groups marked with ``clustered=True`` (intended for embedding matrices):
      - rows are clustered every ``recluster_interval`` steps (after ``warmup_steps``)
      - gradients are mixed with their cluster mean using weight ``alpha``
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        num_clusters: int = 8,
        alpha: float = 0.3,
        recluster_interval: int = 200,
        warmup_steps: int = 10,
        min_cluster_rows: int = 2,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            num_clusters=num_clusters,
            alpha=alpha,
            recluster_interval=recluster_interval,
            warmup_steps=warmup_steps,
            min_cluster_rows=min_cluster_rows,
            clustered=False,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def _maybe_recluster(
        self,
        p: torch.Tensor,
        state: dict,
        num_clusters: int,
        recluster_interval: int,
        warmup_steps: int,
        min_cluster_rows: int,
    ) -> Optional[torch.Tensor]:
        """
        Update cluster assignments for a 2D parameter tensor when due.
        Returns tensor of assignments or None if clustering is skipped.
        """
        if p.dim() != 2 or p.shape[0] < min_cluster_rows:
            return None

        step = state["step"]
        assignments = state.get("assignments")

        if step < warmup_steps:
            return assignments

        should_recluster = assignments is None or step % recluster_interval == 0
        if not should_recluster:
            return assignments

        weights_2d = p.data.view(p.shape[0], -1).detach().cpu().numpy()
        k = max(1, min(num_clusters, weights_2d.shape[0]))

        if k < 2:
            assignments = torch.zeros(weights_2d.shape[0], dtype=torch.long, device=p.device)
        else:
            kmeans = KMeans(n_clusters=k, n_init=5, random_state=42)
            labels = kmeans.fit_predict(weights_2d)
            assignments = torch.from_numpy(labels).long().to(p.device)

        state["assignments"] = assignments
        return assignments

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            clustered = group.get("clustered", False)
            alpha = group["alpha"]
            num_clusters = group["num_clusters"]
            recluster_interval = group["recluster_interval"]
            warmup_steps = group["warmup_steps"]
            min_cluster_rows = group["min_cluster_rows"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("ClusterCoupledAdamW does not support sparse gradients.")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["assignments"] = None

                state["step"] += 1

                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                if clustered:
                    assignments = self._maybe_recluster(
                        p, state, num_clusters, recluster_interval, warmup_steps, min_cluster_rows
                    )
                    if assignments is not None:
                        g2d = grad.view(p.shape[0], -1)
                        k = int(assignments.max().item()) + 1

                        cluster_grad = torch.zeros(k, g2d.size(1), device=grad.device)
                        expand_idx = assignments.view(-1, 1).expand_as(g2d)
                        cluster_grad.scatter_add_(0, expand_idx, g2d)

                        counts = torch.bincount(assignments, minlength=k).float().to(grad.device)
                        counts = counts.clamp_min(1.0).unsqueeze(1)
                        cluster_grad = cluster_grad / counts

                        mixed = (1.0 - alpha) * g2d + alpha * cluster_grad[assignments]
                        grad = mixed.view_as(grad)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                denom = exp_avg_sq.sqrt().add_(eps)
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
