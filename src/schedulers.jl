#
# Copyright (c) 2026 Josef Kircher, Julian Trommer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

"""
    abstract type Scheduler

Per-pass selection rule shared across training strategies that iterate over a
pool of discrete training units (timesteps for `DerivativeTraining`, batches
for `BatchingStrategy`). Each scheduler implements:

- `total_iters(sched, n_units)`: outer-loop iterations per trajectory pass.
  The training loop may still override this via strategy-level `outer_iters`
  (e.g. `BatchingStrategy` uses its `strategy.steps` field).
- `next_index(sched, i, n_units, losses, norm_active)`: unit index for
  iteration `i` (1-based).
- `update_state!(sched, i, actual, loss, n_units, losses)`: hook after the
  training step completes; may mutate the per-unit losses buffer.
- `tracks_losses(sched)`: whether the training loop must allocate the
  per-unit losses buffer before the trajectory pass begins.

See `Sequential`, `WorstLoss`, and `UniqueWorst` for the built-in schedulers.
"""
abstract type Scheduler end

"""
    Sequential()

Linear-cycle scheduler: picks unit `((i-1) % n_units) + 1`. For strategies
whose outer-iteration budget equals `n_units` (e.g. `DerivativeTraining`
without reruns) this is a single ordered pass; for strategies whose budget
exceeds `n_units` it wraps around cyclically. No reruns, no loss tracking.
"""
struct Sequential <: Scheduler end

total_iters(::Sequential, n_units::Integer) = n_units
function next_index(
    ::Sequential, i::Integer, n_units::Integer, ::Union{Vector{Float32},Nothing}, ::Bool
)
    ((i - 1) % n_units) + 1
end
function update_state!(
    ::Sequential, ::Integer, ::Integer, ::Real, ::Integer, ::Union{Vector{Float32},Nothing}
)
    nothing
end
tracks_losses(::Sequential) = false

"""
    WorstLoss(rerun_steps)

After the base pass, take `rerun_steps` extra gradient steps on
`argmax(losses_per_unit)`. The same unit can be selected repeatedly as long
as it remains the worst; each visit overwrites the recorded loss.

When paired with an `Inf32`-initialised losses buffer, the base pass reduces
to `argmax` too — this reproduces `BatchingStrategy`'s historical
"uncomputed-first-else-highest-loss" curriculum exactly.
"""
struct WorstLoss <: Scheduler
    rerun_steps::Integer
end

total_iters(sched::WorstLoss, n_units::Integer) = n_units + sched.rerun_steps

"""
    UniqueWorst(rerun_steps)

Like `WorstLoss`, but each rerun-phase pick sets the selected entry to
`-Inf32` so it cannot be picked again in the same trajectory pass. Total
rerun budget is clamped to `min(rerun_steps, n_units)` so candidates are
never exhausted mid-pass.
"""
struct UniqueWorst <: Scheduler
    rerun_steps::Integer
end

function total_iters(sched::UniqueWorst, n_units::Integer)
    return n_units + min(sched.rerun_steps, n_units)
end

# Shared rerun-phase selection rule for loss-driven schedulers. During the
# norm-steps window we cycle through `1:n_units` instead of calling `argmax`
# because the losses buffer has not been fully populated with real values.
function _rerun_next_index(
    i::Integer, n_units::Integer, losses::Vector{Float32}, norm_active::Bool
)
    if i <= n_units
        return i
    elseif norm_active
        return ((i - 1) % n_units) + 1
    else
        return argmax(losses)
    end
end

function next_index(
    ::Union{WorstLoss,UniqueWorst},
    i::Integer,
    n_units::Integer,
    losses::Vector{Float32},
    norm_active::Bool,
)
    _rerun_next_index(i, n_units, losses, norm_active)
end

function update_state!(
    ::WorstLoss, ::Integer, actual::Integer, loss::Real, ::Integer, losses::Vector{Float32}
)
    losses[actual] = Float32(loss)
    return nothing
end

function update_state!(
    ::UniqueWorst,
    i::Integer,
    actual::Integer,
    loss::Real,
    n_units::Integer,
    losses::Vector{Float32},
)
    losses[actual] = Float32(loss)
    if i > n_units
        losses[actual] = -Inf32
    end
    return nothing
end

tracks_losses(::Union{WorstLoss,UniqueWorst}) = true

"""
    get_scheduler(strategy)

Return the `Scheduler` attached to `strategy`, or `nothing` for strategies
that do not iterate over a unit pool (`SingleShooting`, `MultipleShooting`).
Dispatched on the strategy type; specific methods are defined alongside each
scheduler-aware strategy's struct.
"""
get_scheduler(::TrainingStrategy) = nothing

"""
    outer_iters(strategy, sched, n_units)

Training-loop budget per trajectory pass. Defaults to `total_iters(sched,
n_units)` when a scheduler is present and to `n_units` when absent. Strategies
that own their own budget (e.g. `BatchingStrategy.steps`) override this.
"""
function outer_iters(::TrainingStrategy, sched::Scheduler, n_units::Integer)
    total_iters(sched, n_units)
end
outer_iters(::TrainingStrategy, ::Nothing, n_units::Integer) = n_units
