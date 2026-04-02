#
# Copyright (c) 2026 Josef Kircher
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
# Small ballistic dataset: 10 particles, no boundary nodes, linear drag physics.
# Two-phase training: DerivativeTraining first, then BatchingStrategy fine-tuning.
#

using GraphNetSim

import OrdinaryDiffEq: Euler, Tsit5
import Optimisers: Adam

# Generate dataset into data/ballistic_small if not already present
include(joinpath(dirname(dirname(@__DIR__)), "test", "generators.jl"))
let _gen_dir = joinpath(dirname(dirname(@__DIR__)), "data", "ballistic_small")
    if _needs_generation(_gen_dir)
        @info "Generating dataset: ballistic_small"
        _GenBallistic.generate(_gen_dir)
        @info "  Done."
    end
end

######################
# Network parameters #
######################

message_steps   = 5
layer_size      = 64
hidden_layers   = 2
batch           = 1
epo             = 1
nder            = 40000   # derivative training steps
ns              = 500 + nder  # total steps incl. batching fine-tune
norm_steps      = 20      # small: only 5 train trajectories × ~67 steps each
cuda            = true
cp_derivative   = 2000
cp_solver       = 100

########################
# Node type parameters #
########################

noise_stddevs = [3.0f-7]
types_updated = [1]   # only fluid particles (no boundary in this dataset)
types_noisy   = [1]

########################
# Optimiser parameters #
########################

learning_rate_start  = 1.0f-4
learning_rate_finish = 1.0f-6
opt = Adam(learning_rate_start)

#######################
# Simulation interval #
#######################

tstart = 0.0f0
dt     = 0.002f0
tstop  = 0.132f0   # (T_LENGTH - 1) * dt = 66 * 0.002

stepsPerTrajectory = 66   # covers the full trajectory in one batch

# timesteps at which MSE is calculated during evaluation
mse_steps = tstart:dt:tstop

#########################
# Paths to data folders #
#########################

ds_path  = "data/ballistic_small"
chk_path = "data/ballistic_small/checkpoints"
eval_path = "data/ballistic_small/eval"

data_minmax(ds_path; types_updated=types_updated, types_noisy=types_noisy, noise_stddevs=noise_stddevs)
data_meanstd(ds_path; types_updated=types_updated, types_noisy=types_noisy, noise_stddevs=noise_stddevs)

###########
# Solvers #
###########

solver_train = Tsit5()
solver_eval  = Tsit5()

#################
# Train network #
#################

# Phase 1: Derivative-based training (fast, no ODE solve per step)

train_network(
    opt,
    ds_path,
    chk_path;
    mps=message_steps,
    layer_size=layer_size,
    hidden_layers=hidden_layers,
    batchsize=batch,
    epochs=epo,
    steps=Int(nder),
    use_cuda=cuda,
    checkpoint=cp_derivative,
    norm_steps=norm_steps,
    types_updated=types_updated,
    types_noisy=types_noisy,
    noise_stddevs=noise_stddevs,
    training_strategy=DerivativeTraining(),
    solver_valid=solver_eval,
    solver_valid_dt=dt,
    optimizer_learning_rate_start=learning_rate_start,
    optimizer_learning_rate_stop=learning_rate_finish,
    show_progress_bars=true,
    save_step=false,
)

# Phase 2: BatchingStrategy fine-tuning (ODE-based loss, lower learning rate)

learning_rate_start  = 1.0f-6
learning_rate_finish = nothing
opt = Adam(learning_rate_start)

train_network(
    opt,
    ds_path,
    chk_path;
    mps=message_steps,
    layer_size=layer_size,
    hidden_layers=hidden_layers,
    batchsize=batch,
    epochs=epo,
    steps=Int(ns),
    use_cuda=cuda,
    checkpoint=cp_solver,
    norm_steps=norm_steps,
    types_updated=types_updated,
    types_noisy=types_noisy,
    noise_stddevs=noise_stddevs,
    training_strategy=BatchingStrategy(
        tstart,
        tstop,          # intervall = full trajectory (66 steps × dt)
        solver_train,
        stepsPerTrajectory;
        loss_function=:mae,
        adaptive=false,
    ),
    solver_valid=solver_eval,
    optimizer_learning_rate_start=learning_rate_start,
    optimizer_learning_rate_stop=learning_rate_finish,
    show_progress_bars=true,
)

####################
# Evaluate network #
####################

eval_network(
    ds_path,
    chk_path,
    eval_path,
    solver_eval;
    start=tstart,
    stop=tstop,
    saves=tstart:dt:tstop,
    dt=dt,
    mse_steps=collect(mse_steps),
    mps=message_steps,
    types_updated=types_updated,
    layer_size=layer_size,
    hidden_layers=hidden_layers,
    use_cuda=cuda,
)

visualize(
    eval_path * "/tsit5/trajectories.h5",
    eval_path * "/vtkhdf/",
    "pos",
    "gt",
    ["vel", "acc"],
)

visualize(
    eval_path * "/tsit5/trajectories.h5",
    eval_path * "/vtkhdf/",
    "pos",
    "prediction",
    ["vel", "acc", "err"],
)
