import math
from typing import Optional

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.optim import Optimizer


class ClusterCoupledAdam(Optimizer):
    """
    Cluster-Coupled AdamW optimizer.
    - 일부 param group에 대해:
      * 주기적으로 K-means로 row 단위 클러스터링
      * 클러스터별 평균 gradient를 계산 후 혼합
      * 혼합된 gradient로 AdamW 업데이트
    - param group 옵션:
      * clustered: True/False
      * num_clusters: K (elbow method로 2~K 중 선택)
      * alpha: 클러스터 공유 강도 (0~1)
      * recluster_interval: 재클러스터링 주기 (step 기준)
      * cluster_source: "param" / "grad" / "ema_grad"
      * cluster_beta: EMA 계수 (cluster_source="ema_grad"일 때 사용)
      * cluster_start_step: 몇 번째 step부터 클러스터링을 켤지 (burn-in 용, 기본 0)
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        alpha=0.5,
        num_clusters=8,
        recluster_interval=100,
        cluster_source="param",
        cluster_beta=0.9,
        cluster_start_step=0,
        min_cluster_rows=2,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            alpha=alpha,
            num_clusters=num_clusters,
            recluster_interval=recluster_interval,
            cluster_source=cluster_source,
            cluster_beta=cluster_beta,
            cluster_start_step=cluster_start_step,
            min_cluster_rows=min_cluster_rows,
        )
        super().__init__(params, defaults)

    def _find_best_k_elbow(self, w2d, k_min=2, k_max=16):
        """
        elbow method로 최적 K를 고르는 함수.
        - w2d: (num_rows, dim) numpy array
        - k_min, k_max: 탐색할 K 범위
        반환: best_k (int)
        """
        Ks = list(range(k_min, k_max + 1))
        inertias = []

        for K in Ks:
            kmeans = KMeans(n_clusters=K, random_state=42)
            kmeans.fit(w2d)
            inertias.append(kmeans.inertia_)

        x1, y1 = Ks[0], inertias[0]
        x2, y2 = Ks[-1], inertias[-1]

        distances = []
        for x, y in zip(Ks, inertias):
            num = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
            den = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            distances.append(num / (den + 1e-12))

        best_idx = int(np.argmax(distances))
        best_k = Ks[best_idx]
        return best_k

    @torch.no_grad()
    def _update_clusters(
        self,
        base_tensor,
        state,
        num_clusters,
        recluster_interval,
        min_cluster_rows,
    ):
        """
        하나의 파라미터 텐서에 대해 K-means 클러스터링을 업데이트하는 함수.

        - base_tensor: 클러스터링 기준 텐서 (예: p.data, grad, EMA-grad)
        - state: 이 파라미터에 해당하는 optimizer state dict
        - num_clusters: elbow 탐색 시 사용할 최대 K (k_max 역할)
        - recluster_interval: 몇 step마다 클러스터링을 다시 할지
        """
        step = state.get("cluster_step", 0)
        assignments = state.get("assignments", None)

        # 아직 클러스터링 안 했거나, 일정 step마다 재클러스터링할 때만 새로 계산
        if assignments is None or step % recluster_interval == 0:
            w2d = base_tensor.detach().view(base_tensor.shape[0], -1).cpu().numpy()

            k_max = min(num_clusters, w2d.shape[0])
            if w2d.shape[0] < min_cluster_rows:
                assignments = torch.zeros(
                    w2d.shape[0], dtype=torch.long, device=base_tensor.device
                )
                state["assignments"] = assignments
                state["cluster_step"] = step + 1
                return assignments

            if k_max < 2:
                assignments = torch.zeros(
                    w2d.shape[0], dtype=torch.long, device=base_tensor.device
                )
            else:
                best_K = state.get("best_K", None)
                if best_K is None:
                    best_K = self._find_best_k_elbow(w2d, k_min=2, k_max=k_max)
                    state["best_K"] = best_K

                    print(
                        f"[ClusterCoupledAdam] step={step} | "
                        f"rows={w2d.shape[0]} | k_max={k_max} | best_K(elbow)={best_K}"
                    )
                else:
                    best_K = min(best_K, k_max)

                if best_K < 2:
                    assignments = torch.zeros(
                        w2d.shape[0], dtype=torch.long, device=base_tensor.device
                    )
                else:
                    kmeans = KMeans(n_clusters=best_K, random_state=42)
                    labels = kmeans.fit_predict(w2d)
                    assignments = torch.from_numpy(labels).long().to(base_tensor.device)

            state["assignments"] = assignments

        state["cluster_step"] = step + 1

        return state["assignments"]

    @torch.no_grad()
    def step(self, closure=None):
        """
        Optimizer의 한 step 수행 함수.
        - 각 param group과 파라미터에 대해:
          1) weight decay 적용 (AdamW 방식)
          2) (clustered=True & burn-in 지나면) gradient를 클러스터 평균과 섞어서 g̃ 계산
          3) Adam 모멘트 업데이트 및 파라미터 갱신
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            # cluster 관련 옵션
            clustered = group.get("clustered", False)
            alpha = group.get("alpha", 0.5)
            num_clusters = group.get("num_clusters", 8)
            recluster_interval = group.get("recluster_interval", 100)
            min_cluster_rows = group.get("min_cluster_rows", 2)

            cluster_source = group.get("cluster_source", "param")   # "param" / "grad" / "ema_grad"
            cluster_beta   = group.get("cluster_beta", 0.9)         # EMA 계수 (0.9 ~ 0.99 정도)
            cluster_start_step = group.get("cluster_start_step", 0) # burn-in 끝나는 step

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("ClusterCoupledAdam does not support sparse gradients.")

                state = self.state[p]
                # state 초기화
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    if clustered:
                        state["cluster_step"] = 0
                        state["assignments"] = None
                        if cluster_source == "ema_grad":
                            state["ema_grad"] = torch.zeros_like(
                                p, memory_format=torch.preserve_format
                            )

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                current_step = state["step"]  ### 🔹 현재 step

                # 1) AdamW 스타일 weight decay
                if weight_decay != 0.0:
                    grad = grad.add(p.data, alpha=weight_decay)

                # 2) Cluster-Coupled gradient 혼합 부분
                #    → burn-in 구간(current_step < cluster_start_step)에서는 **혼합 X**
                if clustered and current_step >= cluster_start_step:   ### 🔹 조건 추가
                    if p.data.dim() != 2:
                        raise ValueError(
                            "Clustered param must be a 2D tensor (e.g., [num_embeddings, dim])."
                        )

                    # EMA-grad 업데이트 (cluster_source == "ema_grad"일 때만)
                    if cluster_source == "ema_grad":
                        ema = state.get("ema_grad", None)
                        if ema is None:
                            ema = torch.zeros_like(grad)
                            state["ema_grad"] = ema
                        ema.mul_(cluster_beta).add_(grad, alpha=1.0 - cluster_beta)
                        base_tensor = ema
                    elif cluster_source == "param":
                        base_tensor = p.data
                    elif cluster_source == "grad":
                        base_tensor = grad
                    else:
                        raise ValueError(f"Unknown cluster_source: {cluster_source}")

                    # 클러스터링 수행
                    assignments = self._update_clusters(
                        base_tensor,
                        state,
                        num_clusters,
                        recluster_interval,
                        min_cluster_rows,
                    )

                    if assignments.dim() != 1 or assignments.size(0) != p.shape[0]:
                        raise ValueError("Cluster assignments must be 1D (num_rows,).")

                    g2d = grad.view(p.shape[0], -1)
                    K = int(assignments.max().item() + 1)

                    gc = torch.zeros(K, g2d.size(1), device=grad.device)
                    idx_expanded = assignments.view(-1, 1).expand(-1, g2d.size(1))
                    gc.scatter_add_(0, idx_expanded, g2d)

                    counts = torch.bincount(
                        assignments, minlength=K
                    ).float().to(grad.device).unsqueeze(1)
                    counts = counts.clamp_min(1.0)
                    gc = gc / counts

                    mixed = (1.0 - alpha) * g2d + alpha * gc[assignments]
                    grad = mixed.view_as(grad)
                # else: burn-in 구간에서는 grad를 그대로 사용 (클러스터링 안 함)

                # 3) Adam 업데이트 (표준 Adam)
                beta1_t, beta2_t = beta1, beta2
                exp_avg.mul_(beta1_t).add_(grad, alpha=1 - beta1_t)
                exp_avg_sq.mul_(beta2_t).addcmul_(grad, grad, value=1 - beta2_t)

                denom = exp_avg_sq.sqrt().add_(eps)

                bias_correction1 = 1 - beta1_t ** current_step
                bias_correction2 = 1 - beta2_t ** current_step
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

class ClusterCoupledSGD(Optimizer):
    """
    Cluster-Coupled SGD
    - 클러스터링된 파라미터 그룹에 대해:
        g̃_i = (1 - α) g_i + α * g_cluster(i)
    - 이후 SGD 업데이트: p <- p - lr * g̃
    """

    def __init__(
        self,
        params,
        lr=1e-2,
        momentum=0.0,
        weight_decay=0.0,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def _update_clusters(self, p, state, num_clusters, recluster_interval):
        step = state.get("cluster_step", 0)
        assignments = state.get("assignments", None)

        if assignments is None or step % recluster_interval == 0:
            w2d = p.data.detach().view(p.shape[0], -1).cpu().numpy()

            K = min(num_clusters, w2d.shape[0])
            if K < 2:
                assignments = torch.zeros(w2d.shape[0], dtype=torch.long, device=p.device)
            else:
                km = KMeans(n_clusters=K, random_state=42)
                labels = km.fit_predict(w2d)
                assignments = torch.from_numpy(labels).long().to(p.device)

            state["assignments"] = assignments

        state["cluster_step"] = step + 1
        return state["assignments"]

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]

            clustered = group.get("clustered", False)
            alpha = group.get("alpha", 0.5)
            num_clusters = group.get("num_clusters", 8)
            recluster_interval = group.get("recluster_interval", 100)

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)

                    if clustered:
                        state["cluster_step"] = 0
                        state["assignments"] = None

                # weight decay 적용
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                # 클러스터 gradient 평균 섞기
                if clustered:
                    if p.data.dim() != 2:
                        raise ValueError("Clustered param must be 2D")

                    assignments = self._update_clusters(
                        p, state, num_clusters, recluster_interval
                    )

                    g2d = grad.view(assignments.numel(), -1)
                    K = int(assignments.max().item()) + 1

                    gc = torch.zeros(K, g2d.size(1), device=grad.device)
                    idx_expanded = assignments.view(-1, 1).expand(-1, g2d.size(1))
                    gc.scatter_add_(0, idx_expanded, g2d)

                    counts = torch.bincount(assignments, minlength=K).float().to(p.device)
                    counts = counts.clamp_min(1).unsqueeze(1)
                    gc = gc / counts

                    mixed = (1 - alpha) * g2d + alpha * gc[assignments]
                    grad = mixed.view_as(grad)

                # SGD with momentum
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)

                p.data.add_(buf, alpha=-lr)

        return loss

class ClusterCoupledRMSProp(Optimizer):
    """
    Cluster-Coupled RMSProp

    g̃_i = (1 - α) g_i + α * g_cluster(i)
    p ← p - lr * g̃_i / (sqrt(E[g̃_i^2]) + eps)
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        alpha_cluster=0.5,
        num_clusters=16,
        recluster_interval=100,
        rho=0.9,
        eps=1e-8,
        weight_decay=0.0,
    ):
        defaults = dict(
            lr=lr,
            alpha_cluster=alpha_cluster,
            num_clusters=num_clusters,
            recluster_interval=recluster_interval,
            rho=rho,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def _update_clusters(self, p, state, num_clusters, recluster_interval):
        step = state.get("cluster_step", 0)
        assignments = state.get("assignments", None)

        if assignments is None or step % recluster_interval == 0:
            w2d = p.data.detach().view(p.shape[0], -1).cpu().numpy()

            K = min(num_clusters, w2d.shape[0])
            if K < 2:
                assignments = torch.zeros(w2d.shape[0], dtype=torch.long, device=p.device)
            else:
                km = KMeans(n_clusters=K, random_state=42)
                labels = km.fit_predict(w2d)
                assignments = torch.from_numpy(labels).long().to(p.device)

            state["assignments"] = assignments

        state["cluster_step"] = step + 1
        return state["assignments"]

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            rho = group["rho"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            alpha_c = group["alpha_cluster"]
            num_clusters = group["num_clusters"]
            recluster_interval = group["recluster_interval"]

            clustered = group.get("clustered", False)

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state["square_avg"] = torch.zeros_like(p)
                    if clustered:
                        state["cluster_step"] = 0
                        state["assignments"] = None

                # weight decay
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                # ---- Cluster Mixing ----
                if clustered:
                    if p.data.dim() != 2:
                        raise ValueError("Clustered param must be 2D")

                    assignments = self._update_clusters(
                        p, state, num_clusters, recluster_interval
                    )

                    g2d = grad.view(assignments.numel(), -1)

                    K = int(assignments.max().item()) + 1
                    gc = torch.zeros(K, g2d.size(1), device=p.device)

                    idx_expanded = assignments.view(-1, 1).expand(-1, g2d.size(1))
                    gc.scatter_add_(0, idx_expanded, g2d)

                    counts = torch.bincount(assignments, minlength=K).float().to(p.device)
                    counts = counts.clamp_min(1).unsqueeze(1)

                    gc = gc / counts
                    mixed = (1 - alpha_c) * g2d + alpha_c * gc[assignments]
                    grad = mixed.view_as(grad)

                # ---- RMSProp 업데이트 ----
                square_avg = state["square_avg"]
                square_avg.mul_(rho).addcmul_(grad, grad, value=1 - rho)

                avg = grad / (square_avg.sqrt() + eps)

                p.data.add_(avg, alpha=-lr)

        return loss

class Lamb(Optimizer):
    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-6,
                 weight_decay=0,
                 clamp_value=10,
                 debias=True):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            clamp_value=clamp_value,
            debias=debias
        )
        super(Lamb, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            clamp_value = group['clamp_value']
            debias = group['debias']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError("LAMB does not support sparse gradients.")

                state = self.state[p]

                # State Initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                state["step"] += 1
                step = state["step"]

                # Adam moment update
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Adam bias correction
                if debias:
                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step
                    m = exp_avg / bias_correction1
                    v = exp_avg_sq / bias_correction2
                else:
                    m, v = exp_avg, exp_avg_sq

                adam_step = m / (v.sqrt() + eps)

                # Decoupled weight decay
                if wd != 0:
                    adam_step = adam_step + wd * p

                # Trust ratio
                w_norm = p.norm(p=2)
                g_norm = adam_step.norm(p=2)

                if w_norm.item() > 0 and g_norm.item() > 0:
                    trust_ratio = w_norm / g_norm
                else:
                    trust_ratio = 1.0

                trust_ratio = min(trust_ratio.item(), clamp_value)

                p.add_(adam_step, alpha=-lr * trust_ratio) # Parameter update

        return loss
