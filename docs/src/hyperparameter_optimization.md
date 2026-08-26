# Hyperparameter Optimization

GraphNetSim integrates with [Optuna.jl](https://github.com/una-auxme/Optuna.jl) to automate the search
for good GNN-simulator hyperparameters. This page documents the workflow implemented by the runnable
example
[`example/DamBreakSmall/DamBreakSmall_optuna.jl`](https://github.com/una-auxme/GraphNetSim.jl/tree/main/example/DamBreakSmall/DamBreakSmall_optuna.jl);
the same pattern applies to any dataset.

## Requirements

Optuna is an extra dependency, so activate an environment that provides it. The DamBreakSmall example
ships its own `Project.toml` with Optuna, OrdinaryDiffEq, and Optimisers, so run the script with that
environment from the repository root:

```bash
julia --project=example/DamBreakSmall example/DamBreakSmall/DamBreakSmall_optuna.jl
```

## Workflow overview

The workflow has four parts: a **persistent study**, an **objective** that trains one model per trial,
an **ask/tell loop** that samples the search space, and **result inspection**. A single trial trains a
GNN with sampled hyperparameters and reports the best validation loss returned by
[`train_network`](@ref GraphNetSim.train_network) — that is, the loss of the periodic ODE rollout on
the validation split. Minimizing that value across trials is the optimization objective.

### 1. A persistent, resumable study

The study is backed by a SQLite database and an on-disk artifact store, so re-running the script
continues an existing study rather than starting over:

```julia
storage_url = create_sqlite_url(database_url, database_name)
storage = RDBStorage(storage_url)
artifact_store = FileSystemArtifactStore(artifact_path)

study = Study(
    study_name,
    artifact_store,
    storage;
    sampler=TPESampler(),          # Tree-structured Parzen Estimator
    pruner=MedianPruner(5, 1),     # stop trials worse than the running median
    direction="minimize",
    load_if_exists=true,           # resume an existing study of the same name
)
```

- **Sampler** — `TPESampler` models the relationship between hyperparameters and loss and proposes
  promising configurations; swap in another sampler to change the search strategy.
- **Pruner** — `MedianPruner` terminates unpromising trials early (after a startup grace period) by
  comparing a trial's reported loss against previous trials.
- **`load_if_exists=true`** — combined with the SQLite storage, this is what makes the run resumable.

### 2. The objective — one training run per trial

The objective converts sampled parameters into a training configuration, runs
[`train_network`](@ref GraphNetSim.train_network), and returns the best validation loss. Each trial
trains into a fresh temporary checkpoint directory so trials do not interfere:

```julia
function objective(trial::Trial; params)
    cp_path = mktempdir()

    opt = if params[:optimizer] == "Adam"
        Adam(params[:lr])
    elseif params[:optimizer] == "AdamW"
        AdamW(; eta=params[:lr], lambda=params[:weight_decay])
    else
        RAdam(params[:lr])
    end

    min_val_loss = train_network(
        opt, ds_path, cp_path;
        training_strategy=DerivativeTraining(;
            random=params[:random_sampling], window_size=params[:window_size]
        ),
        steps=n_steps, checkpoint=cp_interval,
        mps=params[:mps], layer_size=params[:layer_size], hidden_layers=params[:hidden_layers],
        noise_stddevs=[params[:noise_std]],
        norm_steps=0, norm_type=params[:norm_type],
        optimizer_learning_rate_start=params[:lr],
        optimizer_learning_rate_stop=params[:lr] * params[:lr_decay_ratio],
        # ... fixed args: types_updated, types_noisy, solver_valid, use_cuda, ...
    )

    report(trial, Float64(min_val_loss), 1)   # feed the pruner
    should_prune(trial) && return nothing

    upload_artifact(study, trial, Dict(String(k) => v for (k, v) in pairs(params)))
    return Float64(min_val_loss)
end
```

`report` hands the trial's loss to the pruner; `should_prune` then decides whether to abandon it;
`upload_artifact` records the trial's hyperparameters for later inspection.

### 3. The ask/tell loop — sampling the search space

Each iteration `ask`s the study for a trial, draws hyperparameters with the `suggest_*` family, runs
the objective, and `tell`s the study the result (or that it was pruned):

```julia
trial = ask(study)

mps           = suggest_int(trial, "mps", 3, 10)
layer_size    = suggest_categorical(trial, "layer_size", [32, 64, 128])
lr            = suggest_float(trial, "lr", 1.0e-5, 1.0e-3; log=true)   # log-scale
norm_type     = Symbol(suggest_categorical(trial, "norm_type", ["minmax", "meanstd"]))
# ... remaining suggestions ...

params = (; mps, layer_size, lr=Float32(lr), norm_type, #= ... =#)
score = objective(trial; params)

if isnothing(score)
    tell(study, trial; prune=true)
else
    tell(study, trial, score)
end
```

Use `suggest_int` / `suggest_categorical` / `suggest_float` (with `log=true` for scale-free
quantities like learning rates) to declare each dimension. The example searches architecture
(`mps`, `layer_size`, `hidden_layers`), optimiser (`optimizer`, `lr`, `lr_decay_ratio`,
`weight_decay`), regularisation (`noise_std`), normalisation (`norm_type`), and training strategy
(`random_sampling`, `window_size`).

### 4. Resume-awareness and results

Because the study is persistent, the loop counts already-completed trials so restarts converge on a
fixed total instead of adding a fresh batch each time:

```julia
n_completed = length(study.study.trials)
n_remaining = max(0, n_trials - n_completed)
```

When the run finishes, inspect the outcome with `best_trial(study)`, `best_params(study)`, and
`best_value(study)`.

## Adapting it to your own dataset

1. **Point at your data** — set `ds_path`, and generate or provide `train.h5` / `valid.h5` /
   `test.h5` + `meta.json`.
2. **Precompute normalization** — if you search `norm_type`, run [`update_meta!`](@ref GraphNetSim.update_meta!)
   once per statistic you want available (`:minmax` and/or `:meanstd`); a trial then only selects
   between them.
3. **Set the fixed budget** — `n_steps` per trial, `cp_interval` (validation cadence), `n_trials`,
   and the simulation interval (`dt`, `tstop`). These trade search breadth against wall-clock time.
4. **Edit the search space** — add or remove `suggest_*` calls, mirror them in the `params`
   NamedTuple, and forward them to `train_network`.

!!! warning "Only tune parameters the API actually exposes"
    Every keyword forwarded to `train_network` must be an [`Args`](@ref GraphNetSim.Args) field, and
    every keyword to a strategy constructor must exist on that strategy (for example,
    [`DerivativeTraining`](@ref GraphNetSim.DerivativeTraining) accepts only `window_size` and
    `random`). Passing an unknown keyword errors when the trial builds its configuration.

## Another instance

[`example/WaterRamps/WaterRamps_optuna.jl`](https://github.com/una-auxme/GraphNetSim.jl/tree/main/example/WaterRamps/WaterRamps_optuna.jl)
applies this same workflow to the (external) WaterRamps dataset, which is useful as a larger-scale
reference — see [Examples](@ref) for why those research scripts are not turnkey.
