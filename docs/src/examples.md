# Examples

Runnable examples live in the [`example/`](https://github.com/una-auxme/GraphNetSim.jl/tree/main/example)
directory of the repository. The two smallest — [BallisticSmall](#BallisticSmall) and
[DamBreakSmall](#DamBreakSmall) — are self-contained: on first run they generate their dataset into
`data/` via the generators in `test/generators.jl`, so no external download is required. Run them from
the repository root with the package's own project environment:

```bash
julia --project example/BallisticSmall/BallisticSmall.jl
```

## BallisticSmall

A tiny ballistic dataset (10 particles, no boundary nodes, linear drag physics) — the simplest
end-to-end example, and a good first run to confirm your setup works.

[`example/BallisticSmall/BallisticSmall.jl`](https://github.com/una-auxme/GraphNetSim.jl/tree/main/example/BallisticSmall/BallisticSmall.jl)
walks through the recommended multi-phase workflow:

1. **DerivativeTraining** — fast initial training against precomputed derivatives (no ODE solve per step).
2. **[`BatchingStrategy`](@ref GraphNetSim.BatchingStrategy)** fine-tuning — ODE-based loss over the trajectory.
3. **[`MultipleShooting`](@ref GraphNetSim.MultipleShooting)** fine-tuning — trajectory split into intervals with a continuity penalty.
4. **[`eval_network`](@ref GraphNetSim.eval_network)** — long-horizon rollout on the test split, then `visualize_eval` to export VTK HDF5 for ParaView.

Because there are no boundary particles, `types_updated = [1]` predicts every particle.

## DamBreakSmall

A tiny 2D weakly-compressible SPH dam break (9 fluid + 9 boundary particles). Like
[BallisticSmall](#BallisticSmall), but with boundary nodes — so it is the reference environment for
both a complete training run and a hyperparameter search.

### Full training pipeline

[`example/DamBreakSmall/DamBreakSmall.jl`](https://github.com/una-auxme/GraphNetSim.jl/tree/main/example/DamBreakSmall/DamBreakSmall.jl)
runs the same four-step pipeline as BallisticSmall, updating only the fluid particles
(`types_updated = [2]`). Offline normalization statistics are precomputed once with
[`data_minmax`](@ref GraphNetSim.data_minmax) and [`data_meanstd`](@ref GraphNetSim.data_meanstd) so
training can run with `norm_steps=0`.

### Hyperparameter optimization with Optuna

[`example/DamBreakSmall/DamBreakSmall_optuna.jl`](https://github.com/una-auxme/GraphNetSim.jl/tree/main/example/DamBreakSmall/DamBreakSmall_optuna.jl)
runs an automated hyperparameter search over the same dataset using
[Optuna.jl](https://github.com/una-auxme/Optuna.jl). It uses the ask/tell interface: each trial trains
a GNN with [`DerivativeTraining`](@ref GraphNetSim.DerivativeTraining) for a fixed number of steps and
reports the best validation loss returned by [`train_network`](@ref GraphNetSim.train_network).

Optuna is an extra dependency, provided by this example's own `Project.toml`, so run it with that
environment activated:

```bash
julia --project=example/DamBreakSmall example/DamBreakSmall/DamBreakSmall_optuna.jl
```

Searched hyperparameters:

| Group | Parameters |
| --- | --- |
| Architecture | `mps`, `layer_size`, `hidden_layers` |
| Optimiser | `optimizer` (Adam / AdamW / RAdam), `lr`, `lr_decay_ratio`, `weight_decay` (AdamW only) |
| Regularisation | `noise_std` |
| Normalisation | `norm_type` (`:minmax` / `:meanstd`) |
| Training | `random_sampling`, `window_size` |

Key properties:

- **Sampler / pruner** — a TPE sampler with a median pruner drops unpromising trials early.
- **Both normalization statistics are precomputed** with [`update_meta!`](@ref GraphNetSim.update_meta!)
  (once for `:minmax`, once for `:meanstd`), so a trial only selects between them via `norm_type`.
- **Resumable** — the study is persisted in a SQLite database and trial artifacts on disk, so re-running
  the script continues from where it left off until the target trial count is reached.

When the run finishes, the best trial, its parameters, and its validation loss are printed. Adjust
`n_trials` and the per-trial `n_steps` at the top of the script to trade search breadth against
wall-clock time.

## Further scripts

The remaining subfolders of [`example/`](https://github.com/una-auxme/GraphNetSim.jl/tree/main/example)
— `Ballistics`, `DamBreak`, `Duese`, `GradientDiagnostics`, `RuntimeBenchmark`, and `WaterRamps` —
hold research and benchmarking material: training variants, ablations, evaluation/visualization
utilities, SLURM (`.sbatch`) cluster job scripts, and comparison harnesses. They target larger
datasets that are **not** bundled with the repository and often assume specific hardware, so treat
them as references rather than turnkey tutorials. Notably, `WaterRamps/WaterRamps_optuna.jl` mirrors
the DamBreakSmall Optuna search for the (external) WaterRamps dataset.
