# 25-fall-recommender

LightGCN 모델을 기반으로 한 옵티마이저 개발 프로젝트입니다. 옵티마이저(Cluster-Coupled Adam 외), warm-up, reclustering interval 등을 CLI 옵션으로 제어할 수 있습니다.

## Key Components
- `code/main.py`: 학습/평가 엔트리포인트.
- `code/parse.py`: 인자 정의. 
    - 예) `--dataset`, `--recdim`, `--optimizer`, `--num_clusters`, `--cluster_warmup` 등
- `code/optimizers.py`: ClusterCoupledAdam/SGD/RMSProp 구현 및 클러스터링 로직.

## Run Example
```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python code/main.py \
  --dataset gowalla \
  --model lgn \
  --recdim 32 \
  --layer 1 \
  --epochs 50 \
  --bpr_batch 4096 \
  --testbatch 512 \
  --optimizer cluster \
  --alpha 0.25 \
  --num_clusters 64 \
  --recluster_interval 128 \
  --cluster_warmup 1024
```

구체적인 사용 흐름은 `notebook.ipynb` 파일을 참조하세요.

## CLI Options
- `--recdim`: 임베딩 차원.
- `--optimizer`: `adam`, `sgd`, `rmsprop`, `lamb`, `cluster`
- 클러스터 관련 옵션: `--alpha`, `--num_clusters`, `--recluster_interval`, `--cluster_warmup`(burn-in), `--cluster_log_coherence`.
- 초기화: `--init_std`로 LightGCN 임베딩 초기화 표준편차 설정.
- 평가 주기: `--test_interval`(에포크 단위).

## Metric/Logging
- Recall/Precision/NDCG/HR
- `convergence_speed`: 직전 평가 대비 Recall(@top1) 증가 속도.
- `--cluster_log_coherence` 활성화 시 재클러스터링마다 inertia 및 클러스터 크기 로그 출력.
