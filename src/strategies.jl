#
# Copyright (c) 2026 Josef Jouaux, Julian Trommer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
# This file contains work derived from DeepMind's "learning_to_simulate"
# (https://github.com/google-deepmind/deepmind-research), modified from the original:
#
#   Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# See THIRD_PARTY_NOTICES.md for details.
#
# This file contains work derived from DeepMind's "learning_to_simulate"
# (https://github.com/google-deepmind/deepmind-research), modified from the original:
#
#   Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# See THIRD_PARTY_NOTICES.md for details.
#

import SciMLBase: AbstractSensitivityAlgorithm, ODEFunction, ReturnCode
import SciMLSensitivity: InterpolatingAdjoint, ZygoteVJP, STACKTRACE_WITH_VJPWARN
import Zygote: pullback
using RecursiveArrayTools, CUDA
using JLD2

#######################################################
# Abstract type and functions for training strategies #
#######################################################

abstract type TrainingStrategy end

include("schedulers.jl")

"""
    prepare_training(strategy)

Function that is executed once before training. Can be overwritten by training strategies if necessary.

## Arguments
- `strategy`: Used training strategy.

## Returns
- Tuple containing the results of the function.
"""
function prepare_training(::TrainingStrategy)
    return (nothing,)
end

"""
    get_delta(strategy, trajectory_length)

Returns the delta between samples in the training data.

## Arguments
- `strategy`: Used training strategy.
- Trajectory length (used for Derivative strategies).

## Returns
- Delta between samples in the training data.
"""
function get_delta(strategy::TrainingStrategy, ::Integer)
    throw(
        ArgumentError(
            "Unknown training strategy: $strategy. See [documentation](https://una-auxme.github.io/MeshGraphNets.jl/dev/strategies/) for available strategies.",
        ),
    )
end

"""
    init_train_step(strategy, t)

Function that is executed before each training sample.

## Arguments
- `strategy`: Used training strategy.
- `t`: Tuple containing the variables necessary for initializing training.
- `ta`: Tuple with additional variables that is returned from [prepare_training](@ref).

## Returns
- Tuple containing variables needed for [train_step](@ref).
"""
function init_train_step(strategy::TrainingStrategy, ::Tuple)
    throw(
        ArgumentError(
            "Unknown training strategy: $strategy. See [documentation](https://una-auxme.github.io/MeshGraphNets.jl/dev/strategies/) for available strategies.",
        ),
    )
end

"""
    train_step(strategy, t)

Performs a single training step and return the resulting gradients and loss.

## Arguments
- `strategy`: Solver strategy that is used for training.
- `t`: Tuple that is returned from [`init_train_step`](@ref).

## Returns
- Gradients for optimization step.
- Loss for optimization step.
"""
function train_step(strategy::TrainingStrategy, ::Tuple)
    throw(
        ArgumentError(
            "Unknown training strategy: $strategy. See [documentation](https://una-auxme.github.io/MeshGraphNets.jl/dev/strategies/) for available strategies.",
        ),
    )
end

"""
    validation_step(strategy, t)

Performs validation of a single trajectory. Should be overwritten by training strategies to determine simulation and data interval before calling the inner function [`_validation_step`](@ref).

## Arguments
- `strategy`: Type of training strategy (used for dispatch).
- `t`: Tuple containing the variables necessary for validation.

## Returns
- See [`_validation_step`](@ref).
"""
function validation_step(strategy::TrainingStrategy, ::Tuple)
    throw(
        ArgumentError(
            "Unknown training strategy: $strategy. See [documentation](https://una-auxme.github.io/MeshGraphNets.jl/dev/strategies/) for available strategies.",
        ),
    )
end

"""
    _validation_step(t, sim_interval, data_interval)

Inner function for validation of a single trajectory.

## Arguments
- `t`: Tuple containing the variables necessary for validation.
- `sim_interval`: Interval that determines the simulated time for the validation.
- `data_interval`: Interval that determines the indices of the timesteps in ground truth and prediction data.

## Returns
- Loss calculated on the difference between ground truth and prediction (via mse).
- Ground truth data with `data_interval` as timesteps.
- Prediction data with `data_interval` as timesteps.
"""
function _validation_step(t::Tuple, sim_interval, data_interval)
    gns, data, meta, _, solver, solver_dt, node_type, pr = t

    initial_state = Dict(
        "position" => data["position"][:, :, 1],
        "velocity" => data["velocity"][:, :, 1],
    )

    target_dict = Dict{String,Int32}()
    for tf in meta["solver_target_features"]
        target_dict[tf] = meta["features"][tf]["dim"]
    end

    gt = vcat([data[tf] for tf in meta["solver_target_features"]]...)[
        :, data["mask"], data_interval
    ]
    sol = rollout(
        solver,
        gns,
        initial_state,
        meta["output_features"],
        meta,
        meta["solver_target_features"],
        node_type,
        data["mask"],
        data["val_mask"],
        Float32(sim_interval[1]),
        Float32(sim_interval[end]),
        solver_dt,
        sim_interval,
        meta["device"],
        pr,
    )
    GC.gc()         # Run Julia's garbage collector first
    if CUDA.functional()
        CUDA.reclaim()  # Force garbage collection and free unused memory
    end
    sol_pos = [u.x for u in sol.u]
    # `sol.u[k]` is the predicted state at frame `k`; the comparison takes the first
    # `length(data_interval)` saved entries.
    prediction = cat(sol_pos...; dims=3)[:, data["mask"], 1:length(data_interval)]

    error = mean((prediction - gt) .^ 2; dims=3)

    return mean(error)
end

####################################################################
# Abstract type and functions for solver based training strategies #
####################################################################

"""
    SolverStrategy <: TrainingStrategy

Abstract base type for ODE solver-based training strategies.

Training strategies that use differential equation solvers (e.g., adaptive Runge-Kutta)
to simulate system dynamics and compute gradients via automatic differentiation.
Includes `SingleShooting`, `MultipleShooting`, and `BatchingStrategy`.

All solver-based strategies solve an ODE problem and compute loss by comparing
predicted and ground truth trajectories at specified timesteps, using sensitivity
algorithms for gradient computation.
"""
abstract type SolverStrategy <: TrainingStrategy end

"""
    _default_solver_sense(solver)

Default sensitivity algorithm for solver-based strategies, chosen from the solver's
adaptivity.

`InterpolatingAdjoint(checkpointing=true)` re-solves the forward problem between
checkpoints during the reverse pass, but that internal re-solve forwards neither `dt`
nor `tstops` (see SciMLSensitivity `interpolating_adjoint.jl`), so a fixed-timestep
solver such as `Euler()` throws `ArgumentError: Fixed timestep methods require a choice
of dt or choosing the tstops`. Disabling checkpointing skips that re-solve and lets the
forward solve's `dt` reach the backward adjoint solve, so fixed-step solvers work (at the
cost of storing the full forward trajectory). Adaptive solvers keep `checkpointing=true`.
"""
function _default_solver_sense(solver)
    return InterpolatingAdjoint(; autojacvec=ZygoteVJP(), checkpointing=isadaptive(solver))
end

"""
    get_delta(::SolverStrategy, ::Integer)

Returns the delta (step size) for solver-based training strategies.

For most solver-based strategies, returns 1 (advancing by single timestep).
Can be overridden by specific strategies (e.g., `BatchingStrategy`).

## Arguments
- `strategy::SolverStrategy`: Solver-based training strategy.
- `trajectory_length::Integer`: Length of the trajectory (unused in base implementation).

## Returns
- Integer delta between training samples.
"""
function get_delta(::SolverStrategy, ::Integer)
    return 1
end

"""
    init_train_step(strategy::SolverStrategy, t::Tuple)

Initializes a training step for solver-based strategies.

Extracts initial conditions, packs state into ComponentArray format, and
prepares ground truth data for ODE problem setup.

## Arguments
- `strategy::SolverStrategy`: Solver-based training strategy.
- `t::Tuple`: Input tuple containing (gns, data, position, velocity, meta, output_fields, target_fields, node_type, mask, device, ...).

## Returns
- `Tuple`: Initialized data for training step.
"""
function init_train_step(strategy::SolverStrategy, t::Tuple)
    gns,
    data, meta, output_fields, target_fields, node_type, mask, val_mask, device, _, _,
    show_progress_bars = t

    initial_state = Dict(
        "position" => data["position"][:, :, 1], "velocity" => data["velocity"][:, :, 1]
    )

    x0 = initial_state["position"]
    dx0 = initial_state["velocity"]

    u0 = device(ComponentArray(; x=initial_state["position"], dx=initial_state["velocity"]))
    gt = vcat([data[tf][:, mask, :] for tf in target_fields]...)

    return (
        gns,
        meta,
        output_fields,
        target_fields,
        node_type,
        mask,
        val_mask,
        u0,
        gt,
        device,
        show_progress_bars,
    )
end

"""
    train_step(strategy::SolverStrategy, t::Tuple)

Performs one training step for solver-based strategies.

Constructs an ODE problem from the GNS model, solves it using the strategy's solver,
and computes gradients via sensitivity analysis (adjoint method).

## Arguments
- `strategy::SolverStrategy`: Solver-based training strategy.
- `t::Tuple`: Initialized data from `init_train_step()`.

## Returns
- `Tuple`: (gradients, loss) - Gradients for optimization and scalar training loss.

## Algorithm
1. Create ODE right-hand side function using `ode_func_train()`.
2. Setup ODE problem with initial conditions and parameters.
3. Compute loss via `train_loss()`.
4. Backpropagate through ODE solver using sensitivity algorithm.
5. Return gradients and loss.
"""
function train_step(strategy::SolverStrategy, t::Tuple)
    gns,
    meta, output_fields, target_fields, node_type, mask, val_mask, u0, gt, device,
    show_progress_bars = t

    pr = ProgressUnknown(;
        desc="Solver progress: ", showspeed=true, enabled=show_progress_bars
    )
    if show_progress_bars
        print("\n\n\n\n\n\n\n\n") # display solver progress after main progress
    end

    ff = ODEFunction{false}(
        (x, ps, t) -> ode_func_train(
            x,
            (
                gns,
                ps,
                output_fields,
                meta,
                target_fields,
                node_type,
                pr,
                mask,
                val_mask,
                device,
            ),
            t,
        ),
    )
    prob = ODEProblem(ff, u0, (strategy.tstart, strategy.tstop), gns.train_state.parameters)
    shoot_loss, shoot_gs = Zygote.withgradient(
        ps -> train_loss(
            strategy,
            (
                prob,
                ps,
                u0,
                nothing,
                gt,
                mask,
                gns.n_norm,
                target_fields,
                [meta["features"][tf]["dim"] for tf in target_fields],
            ),
        ),
        gns.train_state.parameters,
    )
    return shoot_gs, shoot_loss
end

"""
    train_loss(strategy, t)

Inner function for a solver based training step that calculates the loss based on the difference between the ground truth and the predicted solution.

## Arguments
- `strategy`: Solver strategy that is used for training.
- `t`: Tuple containing all variables necessary for loss calculation.

## Returns
- Calculated loss.
"""
function train_loss(strategy::SolverStrategy, ::Tuple)
    throw(
        ArgumentError(
            "Unknown solver based training strategy: $strategy. See [documentation](https://una-auxme.github.io/MeshGraphNets.jl/dev/strategies/) for available solver strategies.",
        ),
    )
end

"""
    validation_step(strategy::SolverStrategy, t::Tuple)

Validation step for solver-based strategies.

Computes validation loss by rolling out the GNS model over the full validation
trajectory and comparing predicted outputs with ground truth.

## Arguments
- `strategy::SolverStrategy`: Solver-based training strategy.
- `t::Tuple`: Validation data tuple containing (gns, data, meta, ...).

## Returns
- `Float32`: Validation loss (mean squared error).
"""
function validation_step(strategy::SolverStrategy, t::Tuple)
    _, data, _, _, _, _, _, _ = t
    dt = data["dt"]
    tstop = dt * (data["trajectory_length"]-1)
    if typeof(dt) <: AbstractArray
        sim_interval = dt[1]:(dt[2] - dt[1]):dt[end]
    else
        # sim_interval = 0.0: dt: dt * t[2]["trajectory_length"]
        sim_interval = 0.0:dt:tstop
    end
    # data_interval = 1:t[2]["trajectory_length"]
    data_interval = 1:(length(sim_interval) - 1)
    return _validation_step(t, sim_interval, data_interval)
end

"""
    struct BatchingStrategy <: SolverStrategy

Solver-based training strategy that batches long trajectories into segments.

Divides long trajectories into time intervals and solves/trains on each segment
independently. Useful for memory-efficient training on long sequences.
"""
struct BatchingStrategy <: SolverStrategy
    tstart::Float32
    intervall::Float32
    solver::OrdinaryDiffEqAlgorithm
    sense::AbstractSensitivityAlgorithm
    steps::Int64
    loss_function::Symbol
    solargs::Any
    scheduler::Scheduler
end

get_scheduler(s::BatchingStrategy) = s.scheduler
outer_iters(s::BatchingStrategy, ::Scheduler, ::Integer) = s.steps
outer_iters(s::BatchingStrategy, ::Nothing, ::Integer) = s.steps

struct Batch
    batchStart::Float32
    batchStop::Float32
end

"""
    BatchingStrategy(tstart, intervall, solver, steps; loss_function=:mae, sense=..., scheduler=WorstLoss(0), solargs...)

Constructor for BatchingStrategy solver-based training.

## Arguments
- `tstart::Float32`: Start time for first interval.
- `intervall::Float32`: Duration of each batch interval.
- `solver::OrdinaryDiffEqAlgorithm`: ODE solver algorithm.
- `steps::Int64`: Number of training iterations per trajectory pass
  (total outer-loop budget; the scheduler decides which batch each iteration
  trains on).

## Keyword Arguments
- `loss_function::Symbol=:mae`: Loss function type.
- `sense::AbstractSensitivityAlgorithm`: Gradient algorithm. Defaults to
  [`_default_solver_sense`](@ref)`(solver)`: `InterpolatingAdjoint` with
  `checkpointing` enabled for adaptive solvers and disabled for fixed-timestep
  solvers (e.g. `Euler()`), which are otherwise incompatible with the
  checkpointed reverse pass.
- `scheduler::Scheduler=WorstLoss(0)`: Per-pass batch-selection rule. The
  default + `Inf32`-initialised losses buffer reproduces the historical
  "uncomputed-first-else-highest-loss" curriculum. Any `Scheduler` works
  (`Sequential`, `Shuffled`, `WorstLoss`, `UniqueWorst`, `WeightedWorst`,
  `TopKWorst`, `EpsilonGreedy`, `PercentileWorst`, `UCB`, `LearningProgress`,
  `Staleness`, `ProgressiveHorizon`, `AnnealedWeighted`); for loss-driven ones
  the buffer fills during the implicit base pass. The scheduler's own
  `rerun_steps` field (if any) is ignored — total outer iterations always equal
  `strategy.steps`.
- `solargs...`: Additional ODE solver keywords.
"""
function BatchingStrategy(
    tstart::Float32,
    intervall,
    solver::OrdinaryDiffEqAlgorithm,
    steps;
    loss_function=:mae,
    sense::AbstractSensitivityAlgorithm=InterpolatingAdjoint(
        autojacvec=ZygoteVJP(), checkpointing=true
    ),
    scheduler::Scheduler=WorstLoss(0),
    solargs...,
)
    loss_function in (:mse, :mae) ||
        throw(ArgumentError("loss_function must be :mse or :mae, got :$loss_function"))
    BatchingStrategy(
        tstart, intervall, solver, sense, steps, loss_function, solargs, scheduler
    )
end

"""
    get_delta(strategy::BatchingStrategy, ::Integer)

Returns the number of steps per batch.

## Arguments
- `strategy::BatchingStrategy`: BatchingStrategy instance.
- Unused trajectory length parameter.

## Returns
- `Integer`: Number of steps per batch (strategy.steps).
"""
function get_delta(strategy::BatchingStrategy, ::Integer)
    return strategy.steps
end

"""
    batchTrajectory(strategy::BatchingStrategy, data::Dict)

Partitions a trajectory into time intervals (batches) for sequential training.

Divides the full trajectory duration into equal-sized time intervals, creating
one Batch object per interval. Used for memory-efficient training on long sequences.

## Arguments
- `strategy::BatchingStrategy`: Batching strategy with interval specifications.
- `data::Dict`: Data dictionary containing `"dt"` (timestep) and `"trajectory_length"`.

## Returns
- `Vector{Batch}`: Array of Batch objects partitioning the trajectory.
"""
function batchTrajectory(strategy::BatchingStrategy, data)
    batches = Array{Batch}(undef, 0)
    i = strategy.tstart
    tstop = data["dt"] * (data["trajectory_length"] - 1)
    while round(i; digits=6) < round(tstop; digits=6)
        b = Batch(round(i; digits=5), round(min(i+strategy.intervall, tstop); digits=6))
        push!(batches, b)
        i += strategy.intervall
    end
    return batches
end

"""
    init_train_step(strategy::BatchingStrategy, t::Tuple)

Initializes a training step for the BatchingStrategy.

Receives the scheduler-selected batch index from the caller, extracts initial
conditions for that time window, packs state into ComponentArray, and
prepares ground truth data.

## Arguments
- `strategy::BatchingStrategy`: BatchingStrategy instance.
- `t::Tuple`: Input tuple (gns, data, meta, output_fields, target_fields, node_type, mask, val_mask, device, batch_index, batches, show_progress_bars).

## Returns
- `Tuple`: Initialized batch data for training step.
"""
function init_train_step(strategy::BatchingStrategy, t::Tuple)
    gns,
    data,
    meta,
    output_fields,
    target_fields,
    node_type,
    mask,
    val_mask,
    device,
    b,
    batches,
    show_progress_bars = t

    tstart = round(Int, (batches[b].batchStart/data["dt"]) + 1)
    tstop = round(Int, (batches[b].batchStop/data["dt"]) + 1)

    initial_state = Dict(
        "position" => data["position"][:, :, tstart],
        "velocity" => data["velocity"][:, :, tstart],
    )

    x0 = initial_state["position"]
    dx0 = initial_state["velocity"]
    # println(mask) hat index von Fluid als Wert

    u0 = device(ComponentArray(; x=x0, dx=dx0))
    gt = vcat([data[tf][:, mask, tstart:tstop] for tf in target_fields]...)
    # Extra targets for env-toggled Phase-2 loss experiments (see `train_loss`):
    #   P2_LOSS=posvel  -> adds a velocity term (velocity depends on the network
    #                      acceleration even for a 1-step Euler batch)
    #   P2_LOSS=posnorm -> normalises the position residual by per-dim std
    gt_vel = data["velocity"][:, mask, tstart:tstop]
    pos_std = reshape(Float32.(meta["features"]["position"]["data_std"]), :, 1, 1)
    return (
        gns,
        data,
        meta,
        output_fields,
        target_fields,
        node_type,
        mask,
        val_mask,
        u0,
        gt,
        device,
        batches,
        b,
        show_progress_bars,
        gt_vel,
        pos_std,
    )
end

"""
    train_step(strategy::BatchingStrategy, t::Tuple)

Performs one training step for the BatchingStrategy.

Constructs an ODE problem for the selected batch, solves it using the strategy's solver,
and computes gradients via sensitivity analysis. Updates batch loss.

## Arguments
- `strategy::BatchingStrategy`: BatchingStrategy instance.
- `t::Tuple`: Data tuple from init_train_step().
"""
function train_step(strategy::BatchingStrategy, t::Tuple)
    gns,
    data,
    meta,
    output_fields,
    target_fields,
    node_type,
    mask,
    val_mask,
    u0,
    gt,
    device,
    batches,
    b,
    show_progress_bars,
    gt_vel,
    pos_std = t

    pr = ProgressUnknown(;
        desc="Solver progress: ", showspeed=true, enabled=show_progress_bars
    )
    if show_progress_bars
        print("\n\n\n\n\n\n\n\n") # display solver progress after main progress
    end

    ff = ODEFunction{false}(
        (x, ps, t) -> ode_func_train(
            x,
            (
                gns,
                ps,
                output_fields,
                meta,
                target_fields,
                node_type,
                pr,
                mask,
                val_mask,
                device,
            ),
            t,
        ),
    )
    prob = ODEProblem(
        ff,
        u0,
        (round(batches[b].batchStart; digits=4), round(batches[b].batchStop; digits=4)),
        gns.train_state.parameters,
    )
    # Read the loss-variant selector OUTSIDE the differentiated region: an ENV
    # lookup is a `ccall` that Zygote cannot differentiate through.
    p2mode = get(ENV, "P2_LOSS", "pos")
    shoot_loss, shoot_gs = Zygote.withgradient(
        ps ->
            train_loss(strategy, (prob, ps, u0, nothing, gt, mask, data["dt"], batches[b])),
        gns.train_state.parameters,
    )
    return shoot_gs, shoot_loss
end

"""
    train_loss(strategy::BatchingStrategy, t::Tuple)

Computes loss for a batch during BatchingStrategy training.

Solves the ODE problem for the batch time interval and compares prediction
with ground truth using the configured loss function.

## Arguments
- `strategy::BatchingStrategy`: BatchingStrategy instance.
- `t::Tuple`: Tuple containing problem, parameters, and ground truth data.
"""
function train_loss(strategy::BatchingStrategy, t::Tuple)
    prob, ps, u0, callback_solve, gt, mask, dt, batch, gt_vel, pos_std, mode = t
    sol = solve(
        remake(prob; p=ps),
        strategy.solver;
        u0=u0,
        dt=dt,
        sensealg=strategy.sense,
        callback=callback_solve,
        saveat=(batch.batchStart:dt:batch.batchStop),
        strategy.solargs...,
    )
    sol_pos = [u.x for u in sol.u]
    pred = cat(sol_pos...; dims=3)

    # Env-toggled Phase-2 loss variants (experiment). Default "pos" reproduces the
    # original position-only loss exactly. See the gradient-suppression analysis:
    # with a 1-step forward-Euler batch the position is independent of the network
    # acceleration, so "pos" has ~zero gradient; "posvel" (velocity term) and a
    # longer `intervall` restore it. `mode` is read outside the AD region in
    # `train_step` (an ENV `ccall` is not differentiable by Zygote).
    if mode == "posnorm"
        res = cpu_device()(gt[:, :, 1:size(pred, 3)] .- pred[:, mask, :]) ./ pos_std
        loss = strategy.loss_function == :mse ? mean(res .^ 2) : mean(abs.(res))
    elseif mode == "posvel"
        sol_vel = [u.dx for u in sol.u]
        predv = cat(sol_vel...; dims=3)
        if strategy.loss_function == :mse
            ep = cpu_device()((gt[:, :, 1:size(pred, 3)] .- pred[:, mask, :]) .^ 2)
            ev = cpu_device()((gt_vel[:, :, 1:size(predv, 3)] .- predv[:, mask, :]) .^ 2)
        else
            ep = cpu_device()(abs.(gt[:, :, 1:size(pred, 3)] .- pred[:, mask, :]))
            ev = cpu_device()(abs.(gt_vel[:, :, 1:size(predv, 3)] .- predv[:, mask, :]))
        end
        loss = mean(ep) + mean(ev)
    else  # "pos" — original behaviour
        if strategy.loss_function == :mse
            error = cpu_device()((gt[:, :, 1:size(pred, 3)] .- pred[:, mask, :]) .^ 2)
        elseif strategy.loss_function == :mae
            error = cpu_device()(abs.(gt[:, :, 1:size(pred, 3)] .- pred[:, mask, :]))
        end
        loss = mean(error)
    end
    return loss
end

"""
    SingleShooting(tstart, dt, tstop, solver; sense = InterpolatingAdjoint(autojacvec = ZygoteVJP()), solargs...)

The default solver based training that is normally used for NeuralODEs.
Simulates the system from `tstart` to `tstop` and calculates the loss based on the difference between the prediction and the ground truth at the timesteps `tstart:dt:tstop`.

## Arguments
- `tstart`: Start time of the simulation.
- `dt`: Interval at which the simulation is saved.
- `tstop`: Stop time of the simulation.
- `solver`: Solver that is used for simulating the system.

## Keyword Arguments
- `sense`: The sensitivity algorithm used for calculating the sensitivities. Defaults to
  [`_default_solver_sense`](@ref)`(solver)` (`InterpolatingAdjoint` with `checkpointing`
  on for adaptive solvers, off for fixed-timestep solvers such as `Euler()`).
- `solargs`: Keyword arguments that are passed on to the solver.
"""
struct SingleShooting <: SolverStrategy
    tstart::Float32
    dt::Float32
    tstop::Float32
    solver::OrdinaryDiffEqAlgorithm
    sense::AbstractSensitivityAlgorithm
    loss_function::Any
    solargs::Any
end

"""
    SingleShooting(tstart, dt, tstop, solver; sense=..., solargs...)

Constructor for SingleShooting with default sensitivity algorithm.

## Arguments
- `tstart::Float32`: Start time of simulation.
- `dt::Float32`: Timestep size.
- `tstop::Float32`: Stop time of simulation.
- `solver::OrdinaryDiffEqAlgorithm`: ODE solver algorithm.

## Keyword Arguments
- `sense::AbstractSensitivityAlgorithm`: Gradient computation algorithm.
- `solargs...`: Additional ODE solver keywords.
"""
function SingleShooting(
    tstart::Float32,
    dt::Float32,
    tstop::Float32,
    solver::OrdinaryDiffEqAlgorithm;
    sense::AbstractSensitivityAlgorithm=InterpolatingAdjoint(
        autojacvec=ZygoteVJP(), checkpointing=true
    ),
    loss_function=:mae,
    solargs...,
)
    SingleShooting(tstart, dt, tstop, solver, sense, loss_function, solargs)
end

"""
    train_loss(strategy::SingleShooting, t::Tuple)

Computes loss for SingleShooting strategy by solving the full trajectory.

Solves the ODE problem over the entire time interval and compares the predicted
trajectory with ground truth data.

## Arguments
- `strategy::SingleShooting`: SingleShooting instance.
- `t::Tuple`: Tuple containing problem, parameters, and ground truth data.
"""
function train_loss(strategy::SingleShooting, t::Tuple)
    prob, ps, u0, callback_solve, gt, mask, n_norm, target_fields, target_dims = t # TODO add du0 TODO vary ps
    tstop_data = strategy.tstart + (size(gt, 3) - 1) * strategy.dt
    sol = solve(
        remake(prob; p=ps),
        strategy.solver;
        u0=u0,
        dt=strategy.dt,
        sensealg=strategy.sense,
        callback=callback_solve,
        saveat=(strategy.tstart:strategy.dt:tstop_data),
        strategy.solargs...,
    ) # TODO remake du0=du0 currently not implemented Chris what are u doing?
    # sol = solve(remake(prob; p = ps), strategy.solver; u0 = u0, dt = strategy.dt)
    # pred = typeof(gt) <: CuArray ? CuArray(sol) : Array(sol)
    # # println("solution aquired")
    # local gt_n
    # local pred_n
    # println(sol.t)
    sol_pos = [u.x for u in sol.u]
    # println(sol_pos)
    pred = cat(sol_pos...; dims=3)
    # for i in eachindex(target_fields)
    #     gt_n = vcat(
    #         [cat(
    #              [n_norm[target_fields[i]](gt[
    #                   (sum(target_dims[1:(i - 1)]) + 1):sum(target_dims[1:i]), :, ts])
    #               for ts in axes(pred, 3)]...; dims = 3
    #          ) for i in eachindex(target_fields)]...
    #     ) |> cpu_device()
    #     pred_n = vcat(
    #         [cat(
    #              [n_norm[target_fields[i]](pred[
    #                   (sum(target_dims[1:(i - 1)]) + 1):sum(target_dims[1:i]), :, ts])
    #               for ts in axes(pred, 3)]...; dims = 3
    #          ) for i in eachindex(target_fields)]...
    #     ) |> cpu_device()
    #     # gt_n = vcat([gt[sum(target_dims[1:i-1])+1:sum(target_dims[1:i]), :, 1:size(pred, 3)] for i in eachindex(target_fields)]...)
    #     # pred_n = cat(sol.u..., dims = 3)
    # end

    # println(typeof(gt_n))
    # println(typeof(pred_n))

    if strategy.loss_function == :mse
        error = cpu_device()((gt[:, :, 1:size(pred, 3)] .- pred[:, mask, :]) .^ 2)
    elseif strategy.loss_function == :mae
        error = cpu_device()(abs.(gt[:, :, 1:size(pred, 3)] .- pred[:, mask, :]))
    end
    loss = mean(error)
    return loss

    # err_buf = Zygote.Buffer(error)

    # err_buf[:, :, :] = error
    # for i in axes(err_buf, 3)
    #     err_buf[:, mask, i] = err_buf[:, :, i] .* vm
    # end
end

"""
    MultipleShooting(tstart, dt, tstop, solver, interval_size, continuity_term = 100; sense = InterpolatingAdjoint(autojacvec = ZygoteVJP(), checkpointing = true), solargs...)

Similar to SingleShooting, but splits the trajectory into intervals that are solved independently and then combines them for loss calculation.
Useful if the network tends to get stuck in a local minimum if SingleShooting is used.

## Arguments
- `tstart`: Start time of the simulation.
- `dt`: Interval at which the simulation is saved.
- `tstop`: Stop time of the simulation.
- `solver`: Solver that is used for simulating the system.
- `interval_size`: Size of the intervals (i.e. number of datapoints in one interval).
- `continuity_term = 100`: Factor by which the error between points of concurrent intervals is multiplied.

## Keyword Arguments
- `sense`: The sensitivity algorithm used for calculating the sensitivities. Defaults to
  [`_default_solver_sense`](@ref)`(solver)` (`InterpolatingAdjoint` with `checkpointing`
  on for adaptive solvers, off for fixed-timestep solvers such as `Euler()`).
- `solargs`: Keyword arguments that are passed on to the solver.
"""
struct MultipleShooting <: SolverStrategy
    tstart::Float32
    dt::Float32
    tstop::Float32
    solver::OrdinaryDiffEqAlgorithm
    sense::AbstractSensitivityAlgorithm
    interval_size::Integer                  # Number of observations in one interval
    continuity_term::Integer
    solargs::Any
end

"""
    MultipleShooting(tstart, dt, tstop, solver, interval_size, continuity_term=100; sense=..., solargs...)

Constructor for MultipleShooting strategy with interval-based solving.

## Arguments
- `tstart::Float32`: Start time of simulation.
- `dt::Float32`: Timestep size.
- `tstop::Float32`: Stop time of simulation.
- `solver::OrdinaryDiffEqAlgorithm`: ODE solver algorithm.
- `interval_size::Integer`: Number of timesteps per interval.
- `continuity_term::Integer=100`: Weight for continuity loss between intervals.

## Keyword Arguments
- `sense::AbstractSensitivityAlgorithm`: Gradient computation algorithm.
- `solargs...`: Additional ODE solver keywords.
"""
function MultipleShooting(
    tstart::Float32,
    dt::Float32,
    tstop::Float32,
    solver::OrdinaryDiffEqAlgorithm,
    interval_size,
    continuity_term=100;
    # checkpointing=false for MultipleShooting: its short sub-interval solves can produce
    # single-point checkpoint segments, and the checkpointed reverse pass then evaluates
    # `abs(cpsol_t[end] - cpsol_t[end-1])` -> BoundsError at index 0 (SciMLSensitivity's
    # interpolating_adjoint.jl). Storing the full forward trajectory avoids that re-solve.
    sense::AbstractSensitivityAlgorithm=InterpolatingAdjoint(
        autojacvec=ZygoteVJP(), checkpointing=false
    ),
    solargs...,
)
    MultipleShooting(
        tstart, dt, tstop, solver, sense, interval_size, continuity_term, solargs
    )
end

function init_train_step(::MultipleShooting, t::Tuple)
    gns,
    data, meta, output_fields, target_fields, node_type, mask, val_mask, device, _, _,
    show_progress_bars = t

    u0 = device(ComponentArray(; x=data["position"][:, :, 1], dx=data["velocity"][:, :, 1]))
    gt = vcat([data[tf][:, mask, :] for tf in target_fields]...)

    return (
        gns,
        data,
        meta,
        output_fields,
        target_fields,
        node_type,
        mask,
        val_mask,
        u0,
        gt,
        device,
        show_progress_bars,
    )
end

function train_step(strategy::MultipleShooting, t::Tuple)
    gns,
    data, meta, output_fields, target_fields, node_type, mask, val_mask, u0, gt, device,
    show_progress_bars = t

    pr = ProgressUnknown(;
        desc="Solver progress: ", showspeed=true, enabled=show_progress_bars
    )
    if show_progress_bars
        print("\n\n\n\n\n\n\n\n")
    end

    ff = ODEFunction{false}(
        (x, ps, t) -> ode_func_train(
            x,
            (
                gns,
                ps,
                output_fields,
                meta,
                target_fields,
                node_type,
                pr,
                mask,
                val_mask,
                device,
            ),
            t,
        ),
    )
    prob = ODEProblem(ff, u0, (strategy.tstart, strategy.tstop), gns.train_state.parameters)
    shoot_loss, shoot_gs = Zygote.withgradient(
        ps -> train_loss(strategy, (prob, ps, data, gt, mask, device)),
        gns.train_state.parameters,
    )
    return shoot_gs, shoot_loss
end

"""
    train_loss(strategy::MultipleShooting, t::Tuple)

Computes loss for MultipleShooting strategy with continuity constraints.

Solves multiple independent intervals and combines losses, adding a penalty
for mismatches between interval endpoints.

## Arguments
- `strategy::MultipleShooting`: MultipleShooting instance.
- `t::Tuple`: Tuple containing problem, parameters, data, ground truth, mask, and device.
"""
function train_loss(strategy::MultipleShooting, t::Tuple)
    prob, ps, data, gt, mask, device = t

    tstop_data = strategy.tstart + (size(gt, 3) - 1) * strategy.dt
    tsteps = strategy.tstart:strategy.dt:tstop_data
    ranges = [
        i:min(length(tsteps), i + strategy.interval_size - 1) for
        i in 1:(strategy.interval_size - 1):(length(tsteps) - 1)
    ]
    sols = [
        solve(
            remake(
                prob;
                p=ps,
                tspan=(tsteps[first(rg)], tsteps[last(rg)]),
                u0=device(
                    ComponentArray(;
                        x=data["position"][:, :, first(rg)],
                        dx=data["velocity"][:, :, first(rg)],
                    ),
                ),
            ),
            strategy.solver;
            # Pass dt only for fixed-timestep solvers (e.g. Euler): they require
            # it, and it must flow through to the InterpolatingAdjoint reverse
            # pass via kwargs. Adaptive solvers choose their own initial step, so
            # we omit it to keep their trajectories byte-identical to the historical
            # MultipleShooting behavior (which passed no dt at all).
            (isadaptive(strategy.solver) ? (;) : (; dt=strategy.dt))...,
            saveat=tsteps[rg],
            sensealg=strategy.sense,
            strategy.solargs...,
        ) for rg in ranges
    ]

    retcodes = [sol.retcode for sol in sols]
    if any(retcodes .!= ReturnCode.Success)
        return Inf
    end

    group_predictions = [cat([u.x for u in sol.u]...; dims=3)[:, mask, :] for sol in sols]

    loss = 0
    for (i, rg) in enumerate(ranges)
        error = cpu_device()((gt[:, :, rg] .- group_predictions[i]) .^ 2)
        loss += mean(error)

        if i > 1
            loss +=
                strategy.continuity_term *
                sum(abs, group_predictions[i - 1][:, :, end] .- gt[:, :, first(rg)])
        end
    end

    return loss
end

#########################################################################
# Abstract type and functions for Derivative based training strategies #
#########################################################################

"""
    abstract type DerivativeStrategy <: TrainingStrategy

Abstract base for derivative-based training strategies.

Training strategies comparing network predictions with finite-difference derivatives
from data, rather than full ODE trajectories. Useful for initial, fast model training.
Includes DerivativeTraining with temporal windowing and optional shuffling.
"""
abstract type DerivativeStrategy <: TrainingStrategy end

"""
    get_delta(strategy::DerivativeStrategy, trajectory_length)

Returns the effective trajectory length for derivative training.

If strategy window_size > 0 and smaller than trajectory_length, returns window_size.
Otherwise returns the full trajectory_length.

## Arguments
- `strategy::DerivativeStrategy`: Derivative-based strategy.
- `trajectory_length::Integer`: Length of the trajectory.
"""
function get_delta(strategy::DerivativeStrategy, trajectory_length::Integer)
    return if strategy.window_size > 0 && strategy.window_size < trajectory_length
        strategy.window_size
    else
        trajectory_length
    end
end

"""
    init_train_step(strategy::DerivativeStrategy, t::Tuple)

Initializes a training step for derivative-based strategies.

Extracts target derivatives at a single datapoint and normalizes using network
feature normalizers.

## Arguments
- `strategy::DerivativeStrategy`: Derivative-based strategy.
- `t::Tuple`: Input tuple with network, data, and sampling information.
"""
function init_train_step(::DerivativeStrategy, t::Tuple)
    gns, data, meta, _, target_fields, node_type, mask, _, device, datapoint, _, _ = t
    target_quantities_change = vcat(
        [
            gns.o_norm[field](data["target|" * field][:, mask, datapoint]) for
            field in target_fields
        ]...,
    )

    return (gns, data, meta, target_quantities_change, node_type, mask, device, datapoint)
end

"""
    train_step(strategy::DerivativeStrategy, t::Tuple)

Performs one training step for derivative-based strategies.

Evaluates network on graph at a single timepoint and computes loss against target
derivatives. Computes gradients via backpropagation.

## Arguments
- `strategy::DerivativeStrategy`: Derivative-based strategy.
- `t::Tuple`: Data tuple from init_train_step().
"""
function train_step(strategy::DerivativeStrategy, t::Tuple)
    gns, data, meta, target_quantities_change, node_type, mask, device, datapoint = t # TODO here own function
    loss, gs = Zygote.withgradient(
        ps -> train_loss(
            strategy,
            (
                ps,
                gns,
                data["position"][:, :, datapoint],
                data["velocity"][:, :, datapoint],
                meta,
                target_quantities_change,
                node_type,
                mask,
                device,
            ),
        ),
        gns.train_state.parameters,
    )

    return gs, loss
end

"""
    train_loss(strategy::DerivativeStrategy, t::Tuple)

Computes loss by comparing predicted and target derivatives.

Builds graph from current state, evaluates network, and computes MSE between
output and target derivatives.

## Arguments
- `strategy::DerivativeStrategy`: Derivative-based strategy.
- `t::Tuple`: Tuple with network parameters, data, and target derivatives.
"""
function train_loss(strategy::DerivativeStrategy, t::Tuple)
    ps, gns, position, velocity, meta, target, node_type, mask, device = t
    graph = build_graph(gns, position, velocity, meta, node_type, mask, device)
    # GraphNetCore >= 0.4: model/state live inside the TrainState; layers are
    # stateless under a forward pass so no state write-back is needed.
    output, _ = gns.train_state.model(graph, ps, gns.train_state.states)

    if strategy.loss_function == :mse
        error = (target .- output[:, mask]) .^ 2
    elseif strategy.loss_function == :mae
        error = abs.(target .- output[:, mask])
    end

    loss = mean(error)

    return loss
end

"""
    mse_red(target, output)

Computes squared error between target and output.

## Arguments
- `target`: Target values.
- `output`: Model output values.

## Returns
- Squared error array.
"""
mse_red(target, output) = begin
    error = (target .- output) .^ 2
end

"""
    mae_red(target, output)

Computes absolute error between target and output.

## Arguments
- `target`: Target values.
- `output`: Model output values.

## Returns
- Absolute error array.
"""
mae_red(target, output) = begin
    error = abs.(target .- output)
end

"""
    validation_step(strategy::DerivativeStrategy, t::Tuple)

Validation step for derivative-based strategies.

Computes validation loss by rolling out GNS model over trajectory window
and comparing derivatives with ground truth.

## Arguments
- `strategy::DerivativeStrategy`: Derivative-based strategy.
- `t::Tuple`: Validation data tuple.
"""
function validation_step(::DerivativeStrategy, t::Tuple)
    _, data, _, delta, _, _, _, _ = t
    dt = data["dt"]
    if typeof(dt) <: AbstractArray
        sim_interval = dt[1]:(dt[2] - dt[1]):dt[delta]
    else
        # sim_interval = 0.0: dt: dt * t[2]["trajectory_length"]
        sim_interval = 0.0:dt:(dt * delta)
    end
    # data_interval = 1:t[2]["trajectory_length"]
    data_interval = 1:(length(sim_interval) - 1)
    return _validation_step(t, sim_interval, data_interval)
end

"""
    struct DerivativeTraining <: DerivativeStrategy

Derivative-based training strategy using finite-difference ground truth.

Compares network output with finite-difference derivatives from data. Faster than
solver-based training, useful for initial model training. Supports temporal windowing,
optional random shuffling, and a pluggable per-pass `Scheduler`.
"""
struct DerivativeTraining <: DerivativeStrategy
    window_size::Integer
    random::Bool
    scheduler::Scheduler
    loss_function::Symbol
end

"""
    DerivativeTraining(; window_size=0, random=true, scheduler=Sequential(), loss_function=:mse)

Constructor for DerivativeTraining strategy.

## Keyword Arguments
- `window_size::Integer=0`: Number of timesteps per trajectory (0 means use all).
- `random::Bool=true`: Whether to shuffle timesteps within the window.
- `scheduler::Scheduler=Sequential()`: Per-pass selection rule. Options:
  `Sequential()` (linear), `Shuffled()` (random permutation per cycle),
  `WorstLoss(n)` (revisit `argmax`-worst step `n` times, repeats allowed),
  `UniqueWorst(n)` (revisit each of the `n` worst steps at most once),
  `WeightedWorst(n; temperature)` (loss-proportional rerun sampling),
  `TopKWorst(n, k)` (round-robin over the `k` worst),
  `EpsilonGreedy(n; epsilon)` (greedy-worst with `epsilon` random exploration),
  `PercentileWorst(n; q)` (uniform among the top-`q`-quantile worst),
  `UCB(n; c)` (upper-confidence-bound bandit), `LearningProgress(n)` (prioritise
  by `|Δloss|`), `Staleness(n; weight)` (loss + recency anti-starvation),
  `ProgressiveHorizon(n; warmup_frac, growth_steps)` (growing time-horizon
  curriculum), or `AnnealedWeighted(n; t0, t1, decay_steps)` (temperature-decayed
  `WeightedWorst`).
- `loss_function::Symbol=:mse`: Loss function type (`:mse` or `:mae`).
"""
function DerivativeTraining(;
    window_size::Integer=0,
    random=true,
    scheduler::Scheduler=Sequential(),
    loss_function::Symbol=:mse,
)
    loss_function in (:mse, :mae) ||
        throw(ArgumentError("loss_function must be :mse or :mae, got :$loss_function"))
    DerivativeTraining(window_size, random, scheduler, loss_function)
end

get_scheduler(s::DerivativeTraining) = s.scheduler
