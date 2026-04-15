#
# Copyright (c) 2026 Josef Kircher
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
# Hyperparameter optimization for the DamBreakSmall dataset using Optuna.jl.
#
# Uses the ask/tell interface for single-step trials: each trial trains a
# GNN with sampled hyperparameters and reports the best validation loss.
#
# Searched hyperparameters:
#   Architecture:  mps, layer_size, hidden_layers
#   Optimiser:     optimizer, lr, lr_decay_ratio, weight_decay (AdamW only)
#   Regularisation: noise_std
#   Normalisation: norm_type
#   Training:      random_sampling, window_size, loss_function
#   Physics:       connectivity_radius
#
# Usage:
#   julia --project example/DamBreakSmall/DamBreakSmall_optuna.jl
#
# The study is persisted in a SQLite database, so re-running the script
# continues from where it left off.
#

using GraphNetSim
using Optuna
import OrdinaryDiffEq: Euler
import Optimisers: Adam, AdamW, RAdam

# ── Dataset setup ────────────────────────────────────────────────────────

include(joinpath(dirname(dirname(@__DIR__)), "test", "generators.jl"))
let _gen_dir = joinpath(dirname(dirname(@__DIR__)), "data", "dam_break_small")
    if _needs_generation(_gen_dir)
        @info "Generating dataset: dam_break_small"
        _GenDamBreak.generate(_gen_dir)
        @info "  Done."
    end
end

ds_path = "data/dam_break_small"

# Precompute both normalization statistics (reusable across trials)
update_meta!(ds_path, :all)

# ── Fixed training parameters ────────────────────────────────────────────

types_updated = [2]
types_noisy = [2]
cuda = true
tstart = 0.0f0
dt = 0.001f0
tstop = 0.079f0
n_steps = 1500         # derivative training steps per trial
cp_interval = 500      # checkpoint (and validate) every N steps

# ── Optuna study setup ──────────────────────────────────────────────────

database_url = "data/dam_break_small/optuna"
database_name = "hpo_db"
study_name = "dam-break-small-hpo"
artifact_path = "data/dam_break_small/optuna/artifacts"

storage_url = create_sqlite_url(database_url, database_name)
storage = RDBStorage(storage_url)
artifact_store = FileSystemArtifactStore(artifact_path)

study = Study(
    study_name,
    artifact_store,
    storage;
    sampler=TPESampler(),
    pruner=MedianPruner(5, 1),
    direction="minimize",
    load_if_exists=true,
)

# ── Objective function ──────────────────────────────────────────────────

function objective(trial::Trial; params)
    cp_path = mktempdir()

    # Build optimizer
    opt = if params[:optimizer] == "Adam"
        Adam(params[:lr])
    elseif params[:optimizer] == "AdamW"
        AdamW(; eta=params[:lr], lambda=params[:weight_decay])
    else  # RAdam
        RAdam(params[:lr])
    end

    lr_stop = params[:lr] * params[:lr_decay_ratio]

    min_val_loss = train_network(
        opt,
        ds_path,
        cp_path;
        training_strategy=DerivativeTraining(;
            random=params[:random_sampling],
            window_size=params[:window_size],
            loss_function=params[:loss_function],
        ),
        steps=n_steps,
        checkpoint=cp_interval,
        types_updated=types_updated,
        types_noisy=types_noisy,
        noise_stddevs=[params[:noise_std]],
        mps=params[:mps],
        layer_size=params[:layer_size],
        hidden_layers=params[:hidden_layers],
        norm_steps=0,
        norm_type=params[:norm_type],
        use_cuda=cuda,
        connectivity_radius=params[:connectivity_radius],
        solver_valid=Euler(),
        solver_valid_dt=dt,
        optimizer_learning_rate_start=params[:lr],
        optimizer_learning_rate_stop=lr_stop,
        show_progress_bars=false,
    )

    # Report validation loss for pruning
    report(trial, Float64(min_val_loss), 1)

    if should_prune(trial)
        return nothing
    end

    # Upload trial hyperparameters and result as artifact
    upload_artifact(study, trial, Dict(String(k) => v for (k, v) in pairs(params)))

    return Float64(min_val_loss)
end

# ── Optimization loop ───────────────────────────────────────────────────

n_trials = 50

# Resume-aware: count already-completed trials so restarts reach the
# target total rather than adding n_trials on top of previous runs.
n_completed = length(study.study.trials)
n_remaining = max(0, n_trials - n_completed)
if n_completed > 0
    println(
        "Resuming study: $n_completed trials already completed, $n_remaining remaining."
    )
end

for i in 1:n_remaining
    trial_num = n_completed + i
    println("\n" * "="^60)
    println("  Trial $trial_num / $n_trials")
    println("="^60)

    trial = ask(study)

    # --- Architecture ---
    mps = suggest_int(trial, "mps", 3, 10)
    layer_size = suggest_categorical(trial, "layer_size", [32, 64, 128])
    hidden_layers = suggest_int(trial, "hidden_layers", 1, 3)

    # --- Optimizer ---
    optimizer = suggest_categorical(trial, "optimizer", ["Adam", "AdamW", "RAdam"])
    lr = suggest_float(trial, "lr", 1.0e-5, 1.0e-3; log=true)
    lr_decay_ratio = suggest_float(trial, "lr_decay_ratio", 0.001, 0.1; log=true)
    weight_decay = if optimizer == "AdamW"
        suggest_float(trial, "weight_decay", 1.0e-6, 1.0e-2; log=true)
    else
        0.0
    end

    # --- Regularisation ---
    noise_std = suggest_float(trial, "noise_std", 1.0e-6, 1.0e-3; log=true)

    # --- Normalisation ---
    norm_type = Symbol(suggest_categorical(trial, "norm_type", ["minmax", "meanstd"]))

    # --- Training strategy ---
    random_sampling = suggest_categorical(trial, "random_sampling", [true, false])
    window_size = suggest_int(trial, "window_size", 0, 5)
    loss_function = Symbol(suggest_categorical(trial, "loss_function", ["mse", "mae"]))

    # --- Physics ---
    # Base connectivity radius from meta.json is 0.072 for dam_break_small
    connectivity_radius = suggest_float(trial, "connectivity_radius", 0.05, 0.15)

    params = (;
        mps,
        layer_size,
        hidden_layers,
        optimizer,
        lr=Float32(lr),
        lr_decay_ratio=Float32(lr_decay_ratio),
        weight_decay=Float32(weight_decay),
        noise_std=Float32(noise_std),
        norm_type,
        random_sampling,
        window_size,
        loss_function,
        connectivity_radius=Float64(connectivity_radius),
    )

    println(
        "  Architecture: mps=$mps, layer_size=$layer_size, hidden_layers=$hidden_layers"
    )
    println(
        "  Optimizer:    $optimizer, lr=$(round(lr; sigdigits=3)), decay_ratio=$(round(lr_decay_ratio; sigdigits=2))",
    )
    if optimizer == "AdamW"
        println("  Weight decay: $(round(weight_decay; sigdigits=3))")
    end
    println("  Noise std:    $(round(Float64(noise_std); sigdigits=3))")
    println("  Norm type:    $norm_type, loss=$loss_function")
    println("  Training:     random=$random_sampling, window=$window_size")
    println("  Connectivity: $(round(connectivity_radius; sigdigits=3))")

    score = objective(trial; params)

    if isnothing(score)
        tell(study, trial; prune=true)
        println("  -> PRUNED")
    else
        tell(study, trial, score)
        println("  -> val_loss = $score")
    end
end

# ── Results ─────────────────────────────────────────────────────────────

println("\n" * "="^60)
println("  OPTIMIZATION COMPLETE ($n_trials trials)")
println("="^60)
println("Best trial:  ", best_trial(study))
println("Best params: ", best_params(study))
println("Best value:  ", best_value(study))
