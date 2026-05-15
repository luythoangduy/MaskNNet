# Channel Reliability Experiments

## Problem Claim

Synthetic anomalies are useful only when they preserve normal reconstruction outside the corrupted region and induce reliable feature changes inside the corrupted region. A large synthetic delta alone is not enough: it can select channels that react to synthetic artifacts instead of anomaly-relevant directions.

Let `x` be a normal image, `xs` a synthetic anomaly image, `M` the synthetic token mask, `f` the frozen encoder, and `g` the manifold projector.

```text
z  = f(x)
zs = f(xs)
delta = zs - z
```

A good synthetic signal should satisfy:

```text
background preservation: (1 - M) * g(zs) ~= (1 - M) * g(z)
masked sensitivity:       M * |delta| is large on useful channels
background non-leakage:   (1 - M) * |delta| is small on useful channels
cross-view consistency:   delta directions agree across synthetic views
```

## Channel Cases To Prove

| Case | Feature pattern | Expected action | Failure if ignored |
| --- | --- | --- | --- |
| Case 1: useful-sensitive | high masked sensitivity, low background leakage, high cross-view consistency | exploit in projection loss and anomaly score | missing anomaly signal |
| Case 2: wrong-sensitive | high masked sensitivity, high leakage or low consistency | suppress or skip | learns synthetic artifacts, over-generalizes projector |
| Case 3: context/identical | low masked sensitivity, low leakage | keep as context, low anomaly weight | losing context if removed aggressively |

## Metrics

Per channel/component:

```text
sensitivity[c] = mean_M |zs[c] - z[c]|
leakage[c]     = mean_(1-M) |zs[c] - z[c]|
importance[c]  = sensitivity[c] / (leakage[c] + eps)
```

With two synthetic views:

```text
delta1 = f(xs1) - f(x)
delta2 = f(xs2) - f(x)
consistency[c] = 1 - |mean_M(delta1[c]) - mean_M(delta2[c])|
                   / (|mean_M(delta1[c])| + |mean_M(delta2[c])| + eps)
score[c] = sensitivity[c] * consistency[c] / (leakage[c] + eps)
```

With real anomaly available only for analysis:

```text
d_syn  = |PCA(f(xs)) - PCA(f(x))|
d_real = |PCA(f(xa)) - PCA(f(x))|
d_gap  = |PCA(f(xs)) - PCA(f(xa))|
```

Case 1 has high `d_syn`, high `d_real`, low `d_gap`. Case 2 has high `d_syn` and high `d_gap`. Case 3 has low `d_syn` and low `d_real`.

## Tables To Report

### Table 1: Does The Problem Exist?

| Dataset | Class | Synthetic | top-delta case1 % | top-delta case2 % | context case3 % | mean background leakage |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| MVTec | bottle | CutPasteNormal | | | | |
| MVTec | bottle | noise | | | | |
| VisA | chewinggum | CutPasteNormal | | | | |

Purpose: show that top `abs(delta)` selects both useful-sensitive and wrong-sensitive channels.

### Table 2: Synthetic Quality

| Synthetic method | sensitivity | leakage | consistency | case1 ratio | case2 ratio | image AUROC | pixel AUROC | PRO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| neural_mask | | | | | | | | |
| CutPasteNormal | | | | | | | | |
| feature dropout | | | | | | | | |
| feature noise | | | | | | | | |
| feature replacement | | | | | | | | |
| mixed | | | | | | | | |

Purpose: support the claim that CutPasteNormal is the best synthetic on the inspected sample when it gives high case1 ratio and low case2 ratio.

### Table 3: Ablation

| Model | channel score | background preservation | delta contrastive | channel weighted loss | image AUROC | pixel AUROC | PRO |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| FoundAD baseline | none | no | no | no | | | |
| + top abs(delta) | sensitivity only | no | no | yes | | | |
| + reliability gate | sensitivity / leakage | no | no | yes | | | |
| + bg preservation | sensitivity / leakage | yes | no | yes | | | |
| + delta contrastive | sensitivity / leakage / consistency | yes | yes | yes | | | |

Purpose: isolate each part of the proposed method.

### Table 4: PCA Dimension Sweep

| PCA dim | case1 ratio | case2 ratio | retained variance | image AUROC | pixel AUROC | PRO |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | | | | | | |
| 32 | | | | | | |
| 64 | | | | | | |
| 128 | | | | | | |
| 256 | | | | | | |

Purpose: show whether PCA removes anomaly-relevant directions when the dimension is too low.

### Table 5: Parameter/Hyperparameter Sweep

| Parameter | Values | Expected observation |
| --- | --- | --- |
| `channel_topk` | 16, 32, 64, 128 | too small misses Case 1, too large admits Case 2 |
| `lambda_selected` | 0.25, 0.5, 1.0, 2.0 | controls strength on selected channels |
| `lambda_background_preservation` | 0, 0.1, 0.5, 1.0 | should reduce leakage/over-generalization |
| `lambda_delta_contrastive` | 0, 0.05, 0.1, 0.2 | should reduce wrong-sensitive features |
| `delta_contrastive_temperature` | 0.07, 0.1, 0.2, 0.5 | lower temperature enforces stronger consistency |
| `feature_synth_alpha` | 0.5, 1.0, 1.5 | too high may increase leakage |
| `token_mask_mode` | rectangle, random_blob, random_tokens | rectangle/blob should be more local-manifold-like |
| `token_mask_min/max_ratio` | 0.02-0.10, 0.05-0.25, 0.10-0.35 | large masks may break context preservation |

## Claims

1. Synthetic anomaly quality should be measured by masked sensitivity plus background preservation, not by global feature delta alone.
2. Top `abs(delta)` exposes sensitive channels but also selects wrong-sensitive channels.
3. The proposed reliability gate suppresses wrong-sensitive channels by penalizing leakage and enforcing cross-view delta consistency.
4. Background preservation prevents the projector from over-generalizing and expanding the normal manifold.
5. Context/identical channels should not dominate anomaly scoring, but should remain available to the projector as normal context.

## Minimal Configs

Baseline:

```bash
python foundad/main.py mode=train app=train_dinov2 app.meta.use_channel_routing=false
```

Sensitivity-only:

```bash
python foundad/main.py mode=train app=train_dinov2 \
  app.meta.use_channel_routing=true \
  app.meta.use_channel_weighted_loss=true \
  app.meta.use_masked_channel_reliability=false
```

Reliability + background preservation:

```bash
python foundad/main.py mode=train app=train_dinov2 \
  app.meta.use_channel_routing=true \
  app.meta.use_channel_weighted_loss=true \
  app.meta.use_masked_channel_reliability=true \
  app.meta.lambda_background_preservation=0.5
```

Full:

```bash
python foundad/main.py mode=train app=train_dinov2 \
  app.meta.use_channel_routing=true \
  app.meta.use_channel_weighted_loss=true \
  app.meta.use_masked_channel_reliability=true \
  app.meta.lambda_background_preservation=0.5 \
  app.meta.lambda_delta_contrastive=0.1
```
