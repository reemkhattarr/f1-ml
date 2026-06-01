# Experiment results

Auto-generated from `results.jsonl` by `tire_strategy.experiments` — do not edit by hand. Each experiment family shows a comparison table (key metrics) followed by the full parameters and metrics for every run.

**Reading the pit metrics:** lower `*_cost` is better (asymmetric strategic cost, missed pit = 5x a spurious one); `cv_*` come from grouped 5-fold CV by race weekend (the trustworthy estimate); `pit_*` come from a temporal train-then-test holdout. Baselines are *never-pit* (majority) and a simple rule-based strategist.

## season-scope

> Adding 2025 to training can only be judged indirectly (no season exists after it): grouped CV pools it in; forward-chaining tests whether each added season improves next-season prediction.

_Last run: 2026-06-01T21:11:37+00:00 · 7 run(s)._

| run | label | n_rows | n_races | cv_model_cost_mean | cv_model_cost_std | cv_majority_cost_mean | cv_rule_cost_mean | cv_beats_both | cv_ap_mean | cv_compound_accuracy_mean | cv_compound_macro_f1_mean | cv_compound_baseline_accuracy_mean | pit_average_precision | pit_roc_auc | pit_recall | pit_precision | pit_asymmetric_cost | pit_majority_cost | pit_rule_cost | compound_accuracy | compound_baseline_accuracy | compound_n_test | cv_ap_std | cv_compound_n_pits | n_test_rows | pit_n | pit_positives | pit_threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| #1 | CV pool 2022-2024 (no 2025) | 74,235 | 68.0000 | 0.3260 | 0.0393 | 0.4623 | 0.4495 | ✅ | 0.3339 | 0.6646 | 0.6214 | 0.5260 |  |  |  |  |  |  |  |  |  |  | 0.0732 | 2,162 |  |  |  |  |
| #2 | CV pool 2022-2025 (+2025) | 100,448 | 92.0000 | 0.3224 | 0.0095 | 0.4565 | 0.4405 | ✅ | 0.3479 | 0.6492 | 0.6166 | 0.5046 |  |  |  |  |  |  |  |  |  |  | 0.0262 | 2,894 |  |  |  |  |
| #3 | CV pool 2018-2025 (all years) | 168,395 | 155.0000 | 0.3219 | 0.0246 | 0.4334 | 0.4324 | ✅ | 0.3069 | 0.6239 | 0.6082 | 0.4536 |  |  |  |  |  |  |  |  |  |  | 0.0319 | 4,412 |  |  |  |  |
| #4 | train 2022-2023 -> test 2025 | 47,726 | 44.0000 |  |  |  |  |  |  |  |  |  | 0.3048 | 0.7742 | 0.5417 | 0.2499 | 0.3420 | 0.4366 | 0.4138 | 0.5847 | 0.4413 | 732.0000 |  |  | 26,213 | 26,213 | 2,289 | 0.6100 |
| #5 | train 2022-2024 -> test 2025 | 74,235 | 68.0000 |  |  |  |  |  |  |  |  |  | 0.2955 | 0.7755 | 0.5422 | 0.2671 | 0.3298 | 0.4366 | 0.4138 | 0.5902 | 0.4413 | 732.0000 |  |  | 26,213 | 26,213 | 2,289 | 0.6200 |
| #6 | train 2018-2024 -> test 2025 | 142,182 | 131.0000 |  |  |  |  |  |  |  |  |  | 0.3179 | 0.7894 | 0.5273 | 0.2801 | 0.3247 | 0.4366 | 0.4138 | 0.5806 | 0.4413 | 732.0000 |  |  | 26,213 | 26,213 | 2,289 | 0.6300 |
| #7 | train 2022-2023 -> test 2024 (current prod split) | 47,726 | 44.0000 |  |  |  |  |  |  |  |  |  | 0.2684 | 0.8054 | 0.5477 | 0.2620 | 0.3268 | 0.4295 | 0.4224 | 0.6961 | 0.6569 | 714.0000 |  |  | 26,509 | 26,509 | 2,277 | 0.6100 |

<details><summary>Full parameters &amp; metrics</summary>

**#1 — CV pool 2022-2024 (no 2025)**  
`git a3a71d8` · 2026-06-01T21:11:37+00:00

| parameter | value | | metric | value |
| --- | --- | --- | --- | --- |
| n_splits | 5 | | n_rows | 74,235 |
| seed | 42 | | n_races | 68.0000 |
| view | grouped-cv | | cv_beats_both | ✅ |
| years | [2022, 2023, 2024] | | cv_ap_mean | 0.3339 |
|  |  | | cv_model_cost_mean | 0.3260 |
|  |  | | cv_majority_cost_mean | 0.4623 |
|  |  | | cv_rule_cost_mean | 0.4495 |
|  |  | | cv_ap_std | 0.0732 |
|  |  | | cv_model_cost_std | 0.0393 |
|  |  | | cv_compound_accuracy_mean | 0.6646 |
|  |  | | cv_compound_macro_f1_mean | 0.6214 |
|  |  | | cv_compound_baseline_accuracy_mean | 0.5260 |
|  |  | | cv_compound_n_pits | 2,162 |

**#2 — CV pool 2022-2025 (+2025)**  
`git a3a71d8` · 2026-06-01T21:11:37+00:00

| parameter | value | | metric | value |
| --- | --- | --- | --- | --- |
| n_splits | 5 | | n_rows | 100,448 |
| seed | 42 | | n_races | 92.0000 |
| view | grouped-cv | | cv_beats_both | ✅ |
| years | [2022, 2023, 2024, 2025] | | cv_ap_mean | 0.3479 |
|  |  | | cv_model_cost_mean | 0.3224 |
|  |  | | cv_majority_cost_mean | 0.4565 |
|  |  | | cv_rule_cost_mean | 0.4405 |
|  |  | | cv_ap_std | 0.0262 |
|  |  | | cv_model_cost_std | 0.0095 |
|  |  | | cv_compound_accuracy_mean | 0.6492 |
|  |  | | cv_compound_macro_f1_mean | 0.6166 |
|  |  | | cv_compound_baseline_accuracy_mean | 0.5046 |
|  |  | | cv_compound_n_pits | 2,894 |

**#3 — CV pool 2018-2025 (all years)**  
`git a3a71d8` · 2026-06-01T21:11:37+00:00

| parameter | value | | metric | value |
| --- | --- | --- | --- | --- |
| n_splits | 5 | | n_rows | 168,395 |
| seed | 42 | | n_races | 155.0000 |
| view | grouped-cv | | cv_beats_both | ✅ |
| years | [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025] | | cv_ap_mean | 0.3069 |
|  |  | | cv_model_cost_mean | 0.3219 |
|  |  | | cv_majority_cost_mean | 0.4334 |
|  |  | | cv_rule_cost_mean | 0.4324 |
|  |  | | cv_ap_std | 0.0319 |
|  |  | | cv_model_cost_std | 0.0246 |
|  |  | | cv_compound_accuracy_mean | 0.6239 |
|  |  | | cv_compound_macro_f1_mean | 0.6082 |
|  |  | | cv_compound_baseline_accuracy_mean | 0.4536 |
|  |  | | cv_compound_n_pits | 4,412 |

**#4 — train 2022-2023 -> test 2025**  
`git a3a71d8` · 2026-06-01T21:11:37+00:00

| parameter | value | | metric | value |
| --- | --- | --- | --- | --- |
| seed | 42 | | n_rows | 47,726 |
| test_years | [2025] | | n_races | 44.0000 |
| train_years | [2022, 2023] | | n_test_rows | 26,213 |
| view | temporal-holdout | | pit_average_precision | 0.3048 |
|  |  | | pit_roc_auc | 0.7742 |
|  |  | | pit_recall | 0.5417 |
|  |  | | pit_precision | 0.2499 |
|  |  | | pit_asymmetric_cost | 0.3420 |
|  |  | | pit_threshold | 0.6100 |
|  |  | | pit_n | 26,213 |
|  |  | | pit_positives | 2,289 |
|  |  | | pit_majority_cost | 0.4366 |
|  |  | | pit_rule_cost | 0.4138 |
|  |  | | compound_accuracy | 0.5847 |
|  |  | | compound_baseline_accuracy | 0.4413 |
|  |  | | compound_n_test | 732.0000 |

**#5 — train 2022-2024 -> test 2025**  
`git a3a71d8` · 2026-06-01T21:11:37+00:00

| parameter | value | | metric | value |
| --- | --- | --- | --- | --- |
| seed | 42 | | n_rows | 74,235 |
| test_years | [2025] | | n_races | 68.0000 |
| train_years | [2022, 2023, 2024] | | n_test_rows | 26,213 |
| view | temporal-holdout | | pit_average_precision | 0.2955 |
|  |  | | pit_roc_auc | 0.7755 |
|  |  | | pit_recall | 0.5422 |
|  |  | | pit_precision | 0.2671 |
|  |  | | pit_asymmetric_cost | 0.3298 |
|  |  | | pit_threshold | 0.6200 |
|  |  | | pit_n | 26,213 |
|  |  | | pit_positives | 2,289 |
|  |  | | pit_majority_cost | 0.4366 |
|  |  | | pit_rule_cost | 0.4138 |
|  |  | | compound_accuracy | 0.5902 |
|  |  | | compound_baseline_accuracy | 0.4413 |
|  |  | | compound_n_test | 732.0000 |

**#6 — train 2018-2024 -> test 2025**  
`git a3a71d8` · 2026-06-01T21:11:37+00:00

| parameter | value | | metric | value |
| --- | --- | --- | --- | --- |
| seed | 42 | | n_rows | 142,182 |
| test_years | [2025] | | n_races | 131.0000 |
| train_years | [2018, 2019, 2020, 2021, 2022, 2023, 2024] | | n_test_rows | 26,213 |
| view | temporal-holdout | | pit_average_precision | 0.3179 |
|  |  | | pit_roc_auc | 0.7894 |
|  |  | | pit_recall | 0.5273 |
|  |  | | pit_precision | 0.2801 |
|  |  | | pit_asymmetric_cost | 0.3247 |
|  |  | | pit_threshold | 0.6300 |
|  |  | | pit_n | 26,213 |
|  |  | | pit_positives | 2,289 |
|  |  | | pit_majority_cost | 0.4366 |
|  |  | | pit_rule_cost | 0.4138 |
|  |  | | compound_accuracy | 0.5806 |
|  |  | | compound_baseline_accuracy | 0.4413 |
|  |  | | compound_n_test | 732.0000 |

**#7 — train 2022-2023 -> test 2024 (current prod split)**  
`git a3a71d8` · 2026-06-01T21:11:37+00:00

| parameter | value | | metric | value |
| --- | --- | --- | --- | --- |
| seed | 42 | | n_rows | 47,726 |
| test_years | [2024] | | n_races | 44.0000 |
| train_years | [2022, 2023] | | n_test_rows | 26,509 |
| view | temporal-holdout | | pit_average_precision | 0.2684 |
|  |  | | pit_roc_auc | 0.8054 |
|  |  | | pit_recall | 0.5477 |
|  |  | | pit_precision | 0.2620 |
|  |  | | pit_asymmetric_cost | 0.3268 |
|  |  | | pit_threshold | 0.6100 |
|  |  | | pit_n | 26,509 |
|  |  | | pit_positives | 2,277 |
|  |  | | pit_majority_cost | 0.4295 |
|  |  | | pit_rule_cost | 0.4224 |
|  |  | | compound_accuracy | 0.6961 |
|  |  | | compound_baseline_accuracy | 0.6569 |
|  |  | | compound_n_test | 714.0000 |

</details>
