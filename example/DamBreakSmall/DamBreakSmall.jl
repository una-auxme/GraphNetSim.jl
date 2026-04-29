#
# Copyright (c) 2026 Josef Kircher
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
# Small WCSPH dam break dataset: 18 particles (9 fluid + 9 boundary), 2D.
# Two-phase training: DerivativeTraining first, then BatchingStrategy fine-tuning.
#
# Physics: weakly compressible SPH with cubic spline kernel.
#   Fluid particles (type 2) move under pressure + viscosity + gravity.
#   Boundary particles (type 1) are stationary wall/floor nodes.
#

using GraphNetSim

import OrdinaryDiffEq: Euler, Tsit5
import Optimisers: Adam

# Generate dataset into data/dam_break_small if not already present
include(joinpath(dirname(dirname(@__DIR__)), "test", "generators.jl"))
let _gen_dir = joinpath(dirname(dirname(@__DIR__)), "data", "dam_break_small")
    if _needs_generation(_gen_dir)
        @info "Generating dataset: dam_break_small"
        _GenDamBreak.generate(_gen_dir)
        @info "  Done."
    end
end

######################
# Network parameters #
######################

message_steps = 5
layer_size = 64
hidden_layers = 2
batch = 1
epo = 1
nder = 5000    # derivative training steps
ns = 500 + nder  # total steps incl. batching fine-tune
nms = 500 + ns   # total steps incl. multiple-shooting fine-tune
norm_steps = 0       # offline stats available in meta.json
cuda = true
cp_derivative = 1000
cp_solver = 100
cp_ms = 100

########################
# Node type parameters #
########################

noise_stddevs = [3.0f-5]    # ≈ 0.1 % of particle spacing — position noise
types_updated = [2]          # only update fluid particles
types_noisy = [2]          # add noise to fluid particles

########################
# Optimiser parameters #
########################

learning_rate_start = 1.0f-4
learning_rate_finish = 1.0f-6
opt = Adam(learning_rate_start)

#######################
# Simulation interval #
#######################

tstart = 0.0f0
dt = 0.001f0
tstop = 0.079f0   # (T_LENGTH - 1) * dt = 79 * 0.001

stepsPerTrajectory = 20   # ≈ 4 batches per trajectory

# Timesteps at which MSE is calculated during evaluation
mse_steps = tstart:dt:tstop

#########################
# Paths to data folders #
#########################

ds_path = "data/dam_break_small"
chk_path = "data/dam_break_small/checkpoints"
eval_path = "data/dam_break_small/eval"

data_minmax(
    ds_path;
    types_updated=types_updated,
    types_noisy=types_noisy,
    noise_stddevs=noise_stddevs,
)
data_meanstd(
    ds_path;
    types_updated=types_updated,
    types_noisy=types_noisy,
    noise_stddevs=noise_stddevs,
)

###########
# Solvers #
###########

solver_train = Tsit5()
solver_eval = Euler()

#################
# Train network #
#################

# Phase 1: Derivative-based training (fast — no ODE solve per step)

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

learning_rate_start = 1.0f-6
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
        tstart, tstop, solver_train, stepsPerTrajectory; loss_function=:mae, adaptive=false
    ),
    solver_valid=solver_eval,
    solver_valid_dt=dt,
    optimizer_learning_rate_start=learning_rate_start,
    optimizer_learning_rate_stop=learning_rate_finish,
    show_progress_bars=true,
)

# Phase 3: MultipleShooting fine-tuning (splits trajectory into intervals
# with continuity penalty — helps escape local minima from SingleShooting)

learning_rate_start = 1.0f-6
learning_rate_finish = nothing
opt = Adam(learning_rate_start)

ms_interval_size = 20   # 4 intervals across the 80-step trajectory

train_network(
    opt,
    ds_path,
    chk_path;
    mps=message_steps,
    layer_size=layer_size,
    hidden_layers=hidden_layers,
    batchsize=batch,
    epochs=epo,
    steps=Int(nms),
    use_cuda=cuda,
    checkpoint=cp_ms,
    norm_steps=norm_steps,
    types_updated=types_updated,
    types_noisy=types_noisy,
    noise_stddevs=noise_stddevs,
    training_strategy=MultipleShooting(
        tstart, dt, tstop, solver_train, ms_interval_size, 100
    ),
    solver_valid=solver_eval,
    solver_valid_dt=dt,
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

visualize_eval(eval_path * "/euler/trajectories.h5", eval_path * "/vtkhdf/")
