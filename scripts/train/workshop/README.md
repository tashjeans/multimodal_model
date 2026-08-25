# Workshop training & analysis scripts

Scripts that produce the NeurIPS workshop results.

## Training (validation-selected checkpoints)

| Script | Model |
|--|--|
| `train_onehot_vicreg_workshop.py` | One-hot + VICReg (+ composition baseline) |
| `train_esm_vicreg_raw_workshop.py` | Raw ESMC + VICReg |
| `train_esm_vicreg_workshop.py` | LoRA ESMC + VICReg |
| `esm_vicreg_common.py` | Shared ESM loaders / evaluate |
| `export_esm_vicreg_workshop_latents.py` | Re-export latents from `best.pt` |
| `run_workshop_experiments.sh` | Launch helper |

## Paper analysis (no retraining)

| Script | Output folder under `paper_analysis/` |
|--|--|
| `analyse_workshop_paper_results.py` | tables, crossreactivity, effective_rank, … |
| `collect_workshop_metrics.py` | `tables/workshop_metrics_*.csv` |
| `analyse_peptide_balanced_geometry.py` | `refined_geometry/` |
| `plot_geometry_multipanel.py` | `refined_geometry/` figure |
| `analyse_negative_set_difficulty.py` | `negative_set_difficulty/` (nearest-positive) |
| `analyse_negative_difficulty_matched.py` | `negative_set_difficulty/` (matched *m*) |
| `analyse_immrep_transfer_stage_diagnostic.py` | `immrep_transfer_stage_diagnostic/` |
| `analyse_immrep_stage_unnormalised_mse.py` | same (Euclidean extension) |
| `analyse_immrep_failure_diagnostics.py` | `immrep_failure_analysis/` |
| `analyse_auc_by_training_distance.py` | `distance_to_training/` |
| `analyse_category_breakdown.py` | `category_breakdown/` |

See `models/outputs/workshop/paper_analysis/README.md` for the output layout,
`detail/` conventions, and GitHub packaging notes.

Do not enable `--cache-merged-reps` for submission runs; it recreates a large
`paper_analysis/reps/` tree that is regenerable from per-seed latents.
