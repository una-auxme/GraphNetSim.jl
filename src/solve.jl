#
# Copyright (c) 2026 Josef Kircher, Julian Trommer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

import ProgressMeter: ProgressUnknown

import ChainRulesCore: @ignore_derivatives
using ComponentArrays

"""
    rollout(solver, gns::GraphNetwork, initial_state, output_fields, meta, target_fields, node_type, mask, val_mask, start, stop, dt, saves, device; pr=nothing)

Solves the ODE problem for a Graph Neural Network simulator using the given solver and computes solution at specified timesteps.

Solves the ODEProblem of the GNS model over the specified time interval. The function handles both fixed and adaptive timestep solvers, with optional progress reporting.

## Arguments
- `solver`: ODE solver algorithm (e.g., `Tsit5()`, `RK4()`) from OrdinaryDiffEq.jl.
- `gns::GraphNetwork`: Graph neural network model to evaluate for dynamics.
- `initial_state::Dict`: Dictionary with `"position"` and `"velocity"` arrays for initial conditions.
- `output_fields::Vector{String}`: Names of output features predicted by the network.
- `meta::Dict`: Dataset metadata containing feature dimensions and specifications.
- `target_fields::Vector{String}`: Names of target (output) features.
- `node_type::Vector`: One-hot encoded node type indicators.
- `mask::Vector`: Boolean mask for valid nodes in graph.
- `val_mask::Vector`: Validation/evaluation mask for output features.
- `start::Float32`: Start time of ODE integration.
- `stop::Float32`: Stop time of ODE integration.
- `dt::Union{Nothing,Float32}`: Fixed timestep (if `nothing`, uses adaptive timestepping).
- `saves::Vector`: Timesteps where solution should be saved.
- `device::Function`: Device placement function (cpu_device or gpu_device).

## Keyword Arguments
- `pr::Union{Nothing,ProgressBar}=nothing`: Progress bar for tracking ODE solve.

## Returns
- `sol`: Solution object containing state trajectories at specified `saves` timesteps.

## Notes
- Uses `ode_step_eval()` as the right-hand side function for the ODE.
- State is packed as `ComponentArray` with `x` (position) and `dx` (velocity) fields.
- Output features are denormalized using stored normalizers before return.
"""
function rollout(
    solver,
    gns::GraphNetwork,
    initial_state,
    output_fields,
    meta,
    target_fields,
    node_type,
    mask,
    val_mask,
    start,
    stop,
    dt,
    saves,
    device,
    pr=nothing,
)
    interval = (start, stop)
    x0 = initial_state["position"]
    dx0 = initial_state["velocity"]

    u0 = device(ComponentArray(; x=x0, dx=dx0))

    prob = ODEProblem{false}(
        ode_step_eval,
        u0,
        interval,
        (
            gns,
            gns.ps,
            output_fields,
            meta,
            target_fields,
            node_type,
            pr,
            mask,
            val_mask,
            device,
        ),
    )
    if isnothing(dt)
        sol = solve(prob, solver; saveat=saves)
        # sol = solve(prob, solver; saveat = saves, tstops = saves)
    else
        sol = solve(prob, solver; saveat=saves, dt=dt)
        # sol = solve(prob, solver; saveat = saves, tstops = saves)
    end

    if !isnothing(pr)
        finish!(pr)
    end

    return sol
end

"""
    ode_func_train(x, (gns, ps, output_fields, meta, target_fields, node_type, pr, mask, val_mask, device), t)

ODE right-hand side function wrapper for training mode.

Simplifies the interface by wrapping the full parameter tuple, delegating to `ode_step()` for the actual dynamics computation.

## Arguments
- `x::ComponentArray`: Current state with fields `x` (position) and `dx` (velocity).
- `params::Tuple`: Parameter tuple containing:
  - `gns::GraphNetwork`: GNS model to evaluate.
  - `ps`: Network parameters.
  - `output_fields::Vector{String}`: Output feature names.
  - `meta::Dict`: Dataset metadata.
  - `target_fields::Vector{String}`: Target feature names.
  - `node_type::Vector`: One-hot node type indicators.
  - `pr::Union{Nothing,ProgressBar}`: Optional progress bar.
  - `mask::Vector`: Node validity mask.
  - `val_mask::Vector`: Validation/output mask.
  - `device::Function`: Device placement function.
- `t::Float32`: Current timestep.

## Returns
- `ComponentArray`: Time derivatives with `x` = velocity and `dx` = accelerations.

## Notes
- Used as RHS function for ODE solver during training phase.
- Delegates to `ode_step()` for graph building and network evaluation.
"""
function ode_func_train(
    x,
    (gns, ps, output_fields, meta, target_fields, node_type, pr, mask, val_mask, device),
    t,
) #TODO add dx
    return ode_step(
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
    ) # TODO for test made ODE instead of SOODE
end

"""
    ode_func_eval(x, (gns, ps, output_fields, meta, target_fields, node_type, pr, mask, val_mask, device), t)

ODE right-hand side function wrapper for evaluation/inference mode.

Simplifies the interface by wrapping the full parameter tuple, delegating to `ode_step_eval()` for the actual dynamics computation.

## Arguments
- `x::ComponentArray`: Current state with fields `x` (position) and `dx` (velocity).
- `params::Tuple`: Parameter tuple containing:
  - `gns::GraphNetwork`: GNS model to evaluate.
  - `ps`: Network parameters.
  - `output_fields::Vector{String}`: Output feature names.
  - `meta::Dict`: Dataset metadata.
  - `target_fields::Vector{String}`: Target feature names.
  - `node_type::Vector`: One-hot node type indicators.
  - `pr::Union{Nothing,ProgressBar}`: Optional progress bar.
  - `mask::Vector`: Node validity mask.
  - `val_mask::Vector`: Validation/output mask.
  - `device::Function`: Device placement function.
- `t::Float32`: Current timestep.

## Returns
- `ComponentArray`: Time derivatives with `x` = velocity and `dx` = accelerations.

## Notes
- Used as RHS function for ODE solver during inference/evaluation phase.
- Delegates to `ode_step_eval()` for graph building and network evaluation.
"""
function ode_func_eval(
    x,
    (gns, ps, output_fields, meta, target_fields, node_type, pr, mask, val_mask, device),
    t,
)
    return ode_step_eval(
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
    )
end

"""
    ode_step(x, (gns, ps, output_fields, meta, target_fields, node_type, pr, mask, val_mask, device), t)

Computes one integration step of ODE dynamics for training mode.

Builds the computation graph, evaluates the GNS model, denormalizes outputs, applies masking, and
returns time derivatives (velocities and accelerations). Includes optional progress reporting.

## Arguments
- `x::ComponentArray`: Current state with fields `x` (position) and `dx` (velocity).
- `params::Tuple`: Parameter tuple containing:
  - `gns::GraphNetwork`: GNS model to evaluate.
  - `ps`: Network parameters.
  - `output_fields::Vector{String}`: Output feature names.
  - `meta::Dict`: Dataset metadata.
  - `target_fields::Vector{String}`: Target feature names.
  - `node_type::Vector`: One-hot node type indicators.
  - `pr::Union{Nothing,ProgressBar}`: Optional progress bar.
  - `mask::Vector`: Node validity mask.
  - `val_mask::Vector`: Validation/output mask.
  - `device::Function`: Device placement function.
- `t::Float32`: Current timestep (for progress reporting).

## Returns
- `ComponentArray`: Time derivatives with fields:
  - `x`: Velocity (from current state)
  - `dx`: Accelerations (denormalized network output masked by `val_mask`)

## Algorithm
1. Build graph from positions, velocities, metadata, and node types.
2. Evaluate GNS model to get predicted output features.
3. Denormalize outputs using stored normalizers.
4. Apply output mask (for selective feature learning).
5. Update progress bar if provided.
6. Return as ComponentArray on specified device.

## Notes
- Used during training phase with gradient computation enabled.
- Network state `gns.st` is updated in-place.
- Denormalization uses normalizers from `gns.o_norm` dictionary.
"""
function ode_step(
    x,
    (gns, ps, output_fields, meta, target_fields, node_type, pr, mask, val_mask, device),
    t,
)
    # graph = nothing
    # @ignore_derivatives begin
    graph = build_graph(gns, x.x, x.dx, meta, node_type, mask, device)
    # end

    output, st = gns.model(graph, ps, gns.st)
    gns.st = st

    indices = [meta["features"][tf]["dim"] for tf in target_fields]
    buf = Zygote.Buffer(output)
    for i in 1:length(output_fields)
        buf[(sum(indices[1:(i - 1)]) + 1):sum(indices[1:i]), :] = inverse_data(
            gns.o_norm[output_fields[i]],
            output[(sum(indices[1:(i - 1)]) + 1):sum(indices[1:i]), :],
        )
    end

    @ignore_derivatives begin
        if !isnothing(pr)
            next!(pr, showvalues=[(:t, "$(t)")])
        end
    end
    return device(ComponentArray(; x=x.dx, dx=copy(buf) .* val_mask))
end

"""
    ode_step_eval(x, (gns, ps, output_fields, meta, target_fields, node_type, pr, mask, val_mask, device), t)

Computes one integration step of ODE dynamics for evaluation/inference mode.

Builds the computation graph, evaluates the GNS model, denormalizes outputs, applies masking, and
returns time derivatives (velocities and accelerations). Identical to `ode_step()` but separated for
potential future inference optimizations.

## Arguments
- `x::ComponentArray`: Current state with fields `x` (position) and `dx` (velocity).
- `params::Tuple`: Parameter tuple containing:
  - `gns::GraphNetwork`: GNS model to evaluate.
  - `ps`: Network parameters.
  - `output_fields::Vector{String}`: Output feature names.
  - `meta::Dict`: Dataset metadata.
  - `target_fields::Vector{String}`: Target feature names.
  - `node_type::Vector`: One-hot node type indicators.
  - `pr::Union{Nothing,ProgressBar}`: Optional progress bar.
  - `mask::Vector`: Node validity mask.
  - `val_mask::Vector`: Validation/output mask.
  - `device::Function`: Device placement function.
- `t::Float32`: Current timestep (for progress reporting).

## Returns
- `ComponentArray`: Time derivatives with fields:
  - `x`: Velocity (from current state)
  - `dx`: Accelerations (denormalized network output masked by `val_mask`)

## Algorithm
1. Build graph from positions, velocities, metadata, and node types.
2. Evaluate GNS model to get predicted output features.
3. Denormalize outputs using stored normalizers.
4. Apply output mask (for selective feature evaluation).
5. Update progress bar if provided.
6. Return as ComponentArray on specified device.

## Notes
- Used during evaluation/inference phase without gradient computation.
- Network state `gns.st` is updated in-place.
- Denormalization uses normalizers from `gns.o_norm` dictionary.
- Future optimization target: could implement checkpointing or reduced-precision compute here.
"""
function ode_step_eval(
    x,
    (gns, ps, output_fields, meta, target_fields, node_type, pr, mask, val_mask, device),
    t,
) # TODO if rework is finished only one ode_step function is needed
    graph = build_graph(gns, x.x, x.dx, meta, node_type, mask, device)

    output, st = gns.model(graph, ps, gns.st)
    gns.st = st
    indices = [meta["features"][tf]["dim"] for tf in target_fields]
    buf = Zygote.Buffer(output)
    for i in 1:length(output_fields)
        buf[(sum(indices[1:(i - 1)]) + 1):sum(indices[1:i]), :] = inverse_data(
            gns.o_norm[output_fields[i]],
            output[(sum(indices[1:(i - 1)]) + 1):sum(indices[1:i]), :],
        )
    end

    @ignore_derivatives begin
        if !isnothing(pr)
            next!(pr, showvalues=[(:t, "$(t)")])
        end
    end

    return device(ComponentArray(; x=x.dx, dx=copy(buf) .* val_mask)) # TODO check why output is used here directly
end
