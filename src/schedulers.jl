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

# Shared base-pass / norm-phase selection rule for loss-driven schedulers.
#
# All loss-driven schedulers share the same first two phases and differ only in
# how they pick a unit once the base pass is complete and normalization is no
# longer accumulating ("exploit" phase):
#
#   1. Base pass (`i <= n_units`): return the linear index `i`. This fills the
#      losses buffer one entry per unit and — combined with the `Inf32` init —
#      reproduces the historical "uncomputed-first" curriculum.
#   2. Norm window (`norm_active`): cycle through `1:n_units` instead of
#      consulting `losses`, which has not been fully populated with real values.
#   3. Exploit phase: defer to `exploit(losses, i, n_units)`.
#
# `exploit` receives the current losses buffer, the (1-based) global iteration
# index `i`, and `n_units`; the latter two let stateless schedulers derive a
# rerun position (e.g. round-robin over a top-K set) without mutable state.
function _explore_exploit_next_index(
    exploit, i::Integer, n_units::Integer, losses::Vector{Float32}, norm_active::Bool
)
    if i <= n_units
        return i
    elseif norm_active
        return ((i - 1) % n_units) + 1
    else
        return exploit(losses, i, n_units)
    end
end

# `WorstLoss`/`UniqueWorst` exploit rule: the single highest-loss unit. With an
# `Inf32`-initialised buffer this also drives the base pass linearly, which is
# why those schedulers can route their whole selection through this helper.
function _rerun_next_index(
    i::Integer, n_units::Integer, losses::Vector{Float32}, norm_active::Bool
)
    return _explore_exploit_next_index(
        (l, _i, _n) -> argmax(l), i, n_units, losses, norm_active
    )
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
    Shuffled()

Random-permutation scheduler: each pass over the unit pool visits every unit
exactly once, but in a fresh random order. A new permutation is drawn at the
start of every cycle (whenever the cyclic position resets to `1`), so a
strategy whose budget exceeds `n_units` re-shuffles on each wrap-around.

Like `Sequential` it performs no reruns and tracks no losses; it differs only
in ordering, which decorrelates consecutive gradient steps without biasing any
unit's visit count. Safe for every training strategy and unaffected by the
norm-accumulation window (the training loop bypasses the scheduler there).
"""
mutable struct Shuffled <: Scheduler
    perm::Vector{Int}
end
Shuffled() = Shuffled(Int[])

total_iters(::Shuffled, n_units::Integer) = n_units
function next_index(
    s::Shuffled, i::Integer, n_units::Integer, ::Union{Vector{Float32},Nothing}, ::Bool
)
    pos = ((i - 1) % n_units) + 1
    if pos == 1
        # Draw a fresh permutation at the start of each cycle. `randperm`
        # allocates a new vector sized to the current pool, so this also
        # adapts when `n_units` changes between trajectories.
        s.perm = randperm(n_units)
    end
    return s.perm[pos]
end
function update_state!(
    ::Shuffled, ::Integer, ::Integer, ::Real, ::Integer, ::Union{Vector{Float32},Nothing}
)
    return nothing
end
tracks_losses(::Shuffled) = false

"""
    WeightedWorst(rerun_steps; temperature=1.0f0)

Stochastic loss-aware scheduler. After the base pass, take `rerun_steps` extra
gradient steps, each on a unit sampled with probability proportional to its
loss raised to `1/temperature`:

- `temperature → 0` concentrates on the single worst unit (≈ `WorstLoss`).
- `temperature == 1` samples in direct proportion to loss.
- `temperature → ∞` approaches uniform sampling.

Unlike `WorstLoss`, attention is spread across several high-loss units instead
of fixating on the current `argmax`, which can avoid over-fitting a single hard
unit. Non-finite (uncomputed) entries are always selected first via `argmax`,
preserving the `Inf32`-init "uncomputed-first" invariant.
"""
struct WeightedWorst <: Scheduler
    rerun_steps::Integer
    temperature::Float32
end
function WeightedWorst(rerun_steps::Integer; temperature::Real=1.0f0)
    WeightedWorst(rerun_steps, Float32(temperature))
end

total_iters(sched::WeightedWorst, n_units::Integer) = n_units + sched.rerun_steps

# Sample an index with probability ∝ loss^(1/temperature). Falls back to the
# first non-finite entry (uncomputed-first) and to uniform sampling when all
# weights vanish, so it degrades gracefully regardless of loss scale.
function _weighted_pick(losses::Vector{Float32}, temperature::Float32)
    @inbounds for k in eachindex(losses)
        if !isfinite(losses[k])
            return argmax(losses)
        end
    end
    inv_t = 1.0f0 / max(temperature, eps(Float32))
    total = 0.0
    @inbounds for k in eachindex(losses)
        total += Float64(max(losses[k], 0.0f0))^inv_t
    end
    if !(total > 0.0)
        return rand(1:length(losses))
    end
    r = rand() * total
    acc = 0.0
    @inbounds for k in eachindex(losses)
        acc += Float64(max(losses[k], 0.0f0))^inv_t
        if r <= acc
            return k
        end
    end
    return length(losses)
end

function next_index(
    sched::WeightedWorst,
    i::Integer,
    n_units::Integer,
    losses::Vector{Float32},
    norm_active::Bool,
)
    return _explore_exploit_next_index(
        (l, _i, _n) -> _weighted_pick(l, sched.temperature), i, n_units, losses, norm_active
    )
end
function update_state!(
    ::WeightedWorst,
    ::Integer,
    actual::Integer,
    loss::Real,
    ::Integer,
    losses::Vector{Float32},
)
    losses[actual] = Float32(loss)
    return nothing
end
tracks_losses(::WeightedWorst) = true

"""
    TopKWorst(rerun_steps, k)

After the base pass, distribute `rerun_steps` extra gradient steps round-robin
across the current `k` worst-loss units rather than repeatedly hammering the
single `argmax` (`WorstLoss`) or visiting each worst unit only once
(`UniqueWorst`). The top-`k` set is recomputed from the live losses buffer on
every rerun pick, so it tracks the worst units as their losses change.

`k` is clamped to `1:n_units`. With `k == 1` this reduces to `WorstLoss`; with
`k == n_units` and `rerun_steps == n_units` it sweeps every unit once more in
worst-first order. Non-finite (uncomputed) entries sort first, preserving the
`Inf32`-init "uncomputed-first" invariant.
"""
struct TopKWorst <: Scheduler
    rerun_steps::Integer
    k::Integer
end

total_iters(sched::TopKWorst, n_units::Integer) = n_units + sched.rerun_steps

function _topk_pick(losses::Vector{Float32}, i::Integer, n_units::Integer, k::Integer)
    k_eff = clamp(k, 1, n_units)
    ranked = partialsortperm(losses, 1:k_eff; rev=true)
    pos = ((i - n_units - 1) % k_eff) + 1
    return ranked[pos]
end

function next_index(
    sched::TopKWorst,
    i::Integer,
    n_units::Integer,
    losses::Vector{Float32},
    norm_active::Bool,
)
    return _explore_exploit_next_index(
        (l, ii, nn) -> _topk_pick(l, ii, nn, sched.k), i, n_units, losses, norm_active
    )
end
function update_state!(
    ::TopKWorst, ::Integer, actual::Integer, loss::Real, ::Integer, losses::Vector{Float32}
)
    losses[actual] = Float32(loss)
    return nothing
end
tracks_losses(::TopKWorst) = true

"""
    EpsilonGreedy(rerun_steps; epsilon=0.1f0)

ε-greedy loss-aware scheduler. After the base pass, take `rerun_steps` extra
gradient steps; each picks the current worst unit (`argmax`) with probability
`1 - epsilon` ("exploit") and a uniformly random unit with probability
`epsilon` ("explore"). This keeps `WorstLoss`'s focus on the hardest unit while
periodically sampling elsewhere to avoid starving units that the greedy rule
never revisits.

`epsilon` is clamped to `[0, 1]`; `epsilon == 0` recovers `WorstLoss`.
Non-finite (uncomputed) entries are selected first via `argmax`, preserving the
`Inf32`-init "uncomputed-first" invariant.
"""
struct EpsilonGreedy <: Scheduler
    rerun_steps::Integer
    epsilon::Float32
end
function EpsilonGreedy(rerun_steps::Integer; epsilon::Real=0.1f0)
    EpsilonGreedy(rerun_steps, Float32(epsilon))
end

total_iters(sched::EpsilonGreedy, n_units::Integer) = n_units + sched.rerun_steps

function _epsilon_pick(losses::Vector{Float32}, epsilon::Float32)
    @inbounds for k in eachindex(losses)
        if !isfinite(losses[k])
            return argmax(losses)
        end
    end
    if rand() < clamp(epsilon, 0.0f0, 1.0f0)
        return rand(1:length(losses))
    end
    return argmax(losses)
end

function next_index(
    sched::EpsilonGreedy,
    i::Integer,
    n_units::Integer,
    losses::Vector{Float32},
    norm_active::Bool,
)
    return _explore_exploit_next_index(
        (l, _i, _n) -> _epsilon_pick(l, sched.epsilon), i, n_units, losses, norm_active
    )
end
function update_state!(
    ::EpsilonGreedy,
    ::Integer,
    actual::Integer,
    loss::Real,
    ::Integer,
    losses::Vector{Float32},
)
    losses[actual] = Float32(loss)
    return nothing
end
tracks_losses(::EpsilonGreedy) = true

# ─────────────────────────────────────────────────────────────────────────────
# Stateful schedulers
#
# The base interface passes only `(i, n_units, losses, norm_active)` — no global
# step counter and no cross-pass history. Schedulers that need either keep it in
# mutable struct fields (like `Shuffled.perm`). Two flavours of state appear
# below:
#
#   * Per-pass state (`UCB`, `LearningProgress`, `Staleness`) — reset at the
#     start of every trajectory pass (`i == 1`), mirroring the loop's per-pass
#     `losses_per_dp` buffer. The base pass visits each unit once and seeds the
#     state; the exploit phase then reads it.
#   * Global state (`ProgressiveHorizon`, `AnnealedWeighted`) — a monotonic
#     `step` counter incremented in `update_state!` that never resets, so these
#     can anneal a quantity over the whole training run. The counter only
#     advances once the scheduler is active (the norm-accumulation window
#     bypasses the scheduler entirely, so warm-up steps are not counted).
# ─────────────────────────────────────────────────────────────────────────────

"""
    ProgressiveHorizon(rerun_steps=0; warmup_frac=0.25f0, growth_steps=1000)

Time-horizon curriculum. Restricts every selection to a growing prefix
`1:h` of the unit pool, where `h = ceil(n_units · frac)` and `frac` ramps
linearly from `warmup_frac` to `1` over the first `growth_steps` active
gradient steps. Within the prefix it cycles linearly (`((i-1) % h) + 1`).

For autoregressive simulators this learns near-term dynamics before long
horizons — the same intuition behind multiple-shooting, applied as a sampling
curriculum. Most natural for `BatchingStrategy`, where units are consecutive
time intervals, but valid for any strategy. No loss tracking; stateful in the
global `step` counter only.
"""
mutable struct ProgressiveHorizon <: Scheduler
    rerun_steps::Integer
    warmup_frac::Float32
    growth_steps::Integer
    step::Int
end
function ProgressiveHorizon(
    rerun_steps::Integer=0; warmup_frac::Real=0.25f0, growth_steps::Integer=1000
)
    return ProgressiveHorizon(rerun_steps, Float32(warmup_frac), growth_steps, 0)
end

total_iters(sched::ProgressiveHorizon, n_units::Integer) = n_units + sched.rerun_steps

function _horizon(sched::ProgressiveHorizon, n_units::Integer)
    frac = clamp(
        sched.warmup_frac +
        (1.0f0 - sched.warmup_frac) *
        (Float32(sched.step) / Float32(max(sched.growth_steps, 1))),
        sched.warmup_frac,
        1.0f0,
    )
    return clamp(ceil(Int, n_units * frac), 1, n_units)
end

function next_index(
    sched::ProgressiveHorizon,
    i::Integer,
    n_units::Integer,
    ::Union{Vector{Float32},Nothing},
    ::Bool,
)
    h = _horizon(sched, n_units)
    return ((i - 1) % h) + 1
end
function update_state!(
    sched::ProgressiveHorizon,
    ::Integer,
    ::Integer,
    ::Real,
    ::Integer,
    ::Union{Vector{Float32},Nothing},
)
    sched.step += 1
    return nothing
end
tracks_losses(::ProgressiveHorizon) = false

"""
    UCB(rerun_steps=0; c=1.0f0)

Upper-confidence-bound bandit scheduler. After the base pass (which visits
each unit once and seeds its running mean loss), each rerun picks the unit
maximising `mean_loss + c·sqrt(ln t / visits)`. The exploration bonus grows for
rarely-visited units, so — unlike `WorstLoss` — no unit is starved
indefinitely; `c` trades exploitation (`0` ≈ greedy mean loss) against
exploration. Per-pass state: visit counts and loss sums, reset each trajectory.
"""
mutable struct UCB <: Scheduler
    rerun_steps::Integer
    c::Float32
    counts::Vector{Int}
    sums::Vector{Float64}
    t::Int
end
function UCB(rerun_steps::Integer=0; c::Real=1.0f0)
    UCB(rerun_steps, Float32(c), Int[], Float64[], 0)
end

total_iters(sched::UCB, n_units::Integer) = n_units + sched.rerun_steps

function _ucb_reset!(sched::UCB, n_units::Integer)
    sched.counts = zeros(Int, n_units)
    sched.sums = zeros(Float64, n_units)
    sched.t = 0
    return nothing
end

function _ucb_pick(sched::UCB)
    # Any never-visited unit wins first (uncomputed-first invariant).
    @inbounds for k in eachindex(sched.counts)
        if sched.counts[k] == 0
            return k
        end
    end
    best, best_val = 1, -Inf
    logt = log(max(sched.t, 1))
    @inbounds for k in eachindex(sched.counts)
        mean_k = sched.sums[k] / sched.counts[k]
        val = mean_k + sched.c * sqrt(logt / sched.counts[k])
        if val > best_val
            best_val, best = val, k
        end
    end
    return best
end

function next_index(
    sched::UCB, i::Integer, n_units::Integer, losses::Vector{Float32}, norm_active::Bool
)
    if i == 1 || length(sched.counts) != n_units
        _ucb_reset!(sched, n_units)
    end
    return _explore_exploit_next_index(
        (l, _i, _n) -> _ucb_pick(sched), i, n_units, losses, norm_active
    )
end
function update_state!(
    sched::UCB, ::Integer, actual::Integer, loss::Real, ::Integer, losses::Vector{Float32}
)
    losses[actual] = Float32(loss)
    sched.counts[actual] += 1
    sched.sums[actual] += Float64(loss)
    sched.t += 1
    return nothing
end
tracks_losses(::UCB) = true

"""
    LearningProgress(rerun_steps=0)

Prioritise units by *learning progress* — the absolute change in loss between
consecutive visits, `|loss - prev_loss|` — rather than by loss level. A unit
whose loss is high but stuck is deprioritised relative to one that is actively
moving, focusing reruns where gradient steps still pay off. Units with only one
sample so far carry `Inf32` progress, so each is revisited at least once to
establish a delta (preserving the uncomputed-first invariant). Per-pass state:
previous-loss and progress buffers, reset each trajectory.
"""
mutable struct LearningProgress <: Scheduler
    rerun_steps::Integer
    prevloss::Vector{Float32}
    progress::Vector{Float32}
end
function LearningProgress(rerun_steps::Integer=0)
    return LearningProgress(rerun_steps, Float32[], Float32[])
end

total_iters(sched::LearningProgress, n_units::Integer) = n_units + sched.rerun_steps

function _lp_reset!(sched::LearningProgress, n_units::Integer)
    sched.prevloss = fill(Inf32, n_units)
    sched.progress = fill(Inf32, n_units)
    return nothing
end

function next_index(
    sched::LearningProgress,
    i::Integer,
    n_units::Integer,
    losses::Vector{Float32},
    norm_active::Bool,
)
    if i == 1 || length(sched.progress) != n_units
        _lp_reset!(sched, n_units)
    end
    return _explore_exploit_next_index(
        (l, _i, _n) -> argmax(sched.progress), i, n_units, losses, norm_active
    )
end
function update_state!(
    sched::LearningProgress,
    ::Integer,
    actual::Integer,
    loss::Real,
    ::Integer,
    losses::Vector{Float32},
)
    losses[actual] = Float32(loss)
    if isfinite(sched.prevloss[actual])
        sched.progress[actual] = abs(Float32(loss) - sched.prevloss[actual])
    else
        sched.progress[actual] = Inf32
    end
    sched.prevloss[actual] = Float32(loss)
    return nothing
end
tracks_losses(::LearningProgress) = true

"""
    PercentileWorst(rerun_steps; q=0.9f0)

Robust top-quantile scheduler. Each rerun samples uniformly among the units
whose loss is at or above the `q`-quantile of the current losses buffer. Unlike
`TopKWorst`'s fixed-`k` set, the candidate pool adapts to the loss distribution
— widening when many units are equally bad and narrowing to the true tail when
one dominates — which is less sensitive to a single outlier than `argmax`.
Stateless; tracks losses. Non-finite (uncomputed) entries are selected first.
"""
struct PercentileWorst <: Scheduler
    rerun_steps::Integer
    q::Float32
end
function PercentileWorst(rerun_steps::Integer; q::Real=0.9f0)
    PercentileWorst(rerun_steps, Float32(q))
end

total_iters(sched::PercentileWorst, n_units::Integer) = n_units + sched.rerun_steps

function _percentile_pick(losses::Vector{Float32}, q::Float32)
    @inbounds for k in eachindex(losses)
        if !isfinite(losses[k])
            return argmax(losses)
        end
    end
    thr = Float32(quantile(losses, q))
    n_cand = 0
    @inbounds for k in eachindex(losses)
        n_cand += losses[k] >= thr
    end
    n_cand == 0 && return argmax(losses)
    pick = rand(1:n_cand)
    c = 0
    @inbounds for k in eachindex(losses)
        if losses[k] >= thr
            c += 1
            c == pick && return k
        end
    end
    return argmax(losses)
end

function next_index(
    sched::PercentileWorst,
    i::Integer,
    n_units::Integer,
    losses::Vector{Float32},
    norm_active::Bool,
)
    return _explore_exploit_next_index(
        (l, _i, _n) -> _percentile_pick(l, sched.q), i, n_units, losses, norm_active
    )
end
function update_state!(
    ::PercentileWorst,
    ::Integer,
    actual::Integer,
    loss::Real,
    ::Integer,
    losses::Vector{Float32},
)
    losses[actual] = Float32(loss)
    return nothing
end
tracks_losses(::PercentileWorst) = true

"""
    Staleness(rerun_steps=0; weight=1.0f0)

Recency-aware anti-starvation scheduler. Each rerun picks
`argmax(loss + weight · steps_since_last_visit)`, so high-loss units are still
favoured but any unit left untouched long enough eventually wins regardless of
its loss. `weight` tunes the trade-off: `0` recovers `WorstLoss`, large values
approach round-robin coverage. A simpler, single-knob alternative to `UCB`.
Per-pass state: per-unit last-visit timestamps, reset each trajectory.
"""
mutable struct Staleness <: Scheduler
    rerun_steps::Integer
    weight::Float32
    lastvisit::Vector{Int}
    t::Int
end
function Staleness(rerun_steps::Integer=0; weight::Real=1.0f0)
    return Staleness(rerun_steps, Float32(weight), Int[], 0)
end

total_iters(sched::Staleness, n_units::Integer) = n_units + sched.rerun_steps

function _stale_reset!(sched::Staleness, n_units::Integer)
    sched.lastvisit = zeros(Int, n_units)
    sched.t = 0
    return nothing
end

function _stale_pick(sched::Staleness, losses::Vector{Float32})
    best, best_val = 1, -Inf32
    @inbounds for k in eachindex(losses)
        val = losses[k] + sched.weight * Float32(sched.t - sched.lastvisit[k])
        if val > best_val
            best_val, best = val, k
        end
    end
    return best
end

function next_index(
    sched::Staleness,
    i::Integer,
    n_units::Integer,
    losses::Vector{Float32},
    norm_active::Bool,
)
    if i == 1 || length(sched.lastvisit) != n_units
        _stale_reset!(sched, n_units)
    end
    return _explore_exploit_next_index(
        (l, _i, _n) -> _stale_pick(sched, l), i, n_units, losses, norm_active
    )
end
function update_state!(
    sched::Staleness,
    ::Integer,
    actual::Integer,
    loss::Real,
    ::Integer,
    losses::Vector{Float32},
)
    losses[actual] = Float32(loss)
    sched.t += 1
    sched.lastvisit[actual] = sched.t
    return nothing
end
tracks_losses(::Staleness) = true

"""
    AnnealedWeighted(rerun_steps=0; t0=2.0f0, t1=0.1f0, decay_steps=1000)

Temperature-annealed variant of `WeightedWorst`. Reruns sample a unit with
probability ∝ `loss^(1/T)`, where the temperature `T` decays linearly from `t0`
(exploratory, near-uniform) to `t1` (greedy, near-`argmax`) over the first
`decay_steps` active gradient steps. Mirrors simulated annealing: spread
attention early, sharpen onto the worst units late. Reuses `WeightedWorst`'s
sampler; stateful in the global `step` counter only.
"""
mutable struct AnnealedWeighted <: Scheduler
    rerun_steps::Integer
    t0::Float32
    t1::Float32
    decay_steps::Integer
    step::Int
end
function AnnealedWeighted(
    rerun_steps::Integer=0; t0::Real=2.0f0, t1::Real=0.1f0, decay_steps::Integer=1000
)
    return AnnealedWeighted(rerun_steps, Float32(t0), Float32(t1), decay_steps, 0)
end

total_iters(sched::AnnealedWeighted, n_units::Integer) = n_units + sched.rerun_steps

function _annealed_temp(sched::AnnealedWeighted)
    frac = clamp(Float32(sched.step) / Float32(max(sched.decay_steps, 1)), 0.0f0, 1.0f0)
    return sched.t0 + (sched.t1 - sched.t0) * frac
end

function next_index(
    sched::AnnealedWeighted,
    i::Integer,
    n_units::Integer,
    losses::Vector{Float32},
    norm_active::Bool,
)
    temp = _annealed_temp(sched)
    return _explore_exploit_next_index(
        (l, _i, _n) -> _weighted_pick(l, temp), i, n_units, losses, norm_active
    )
end
function update_state!(
    sched::AnnealedWeighted,
    ::Integer,
    actual::Integer,
    loss::Real,
    ::Integer,
    losses::Vector{Float32},
)
    losses[actual] = Float32(loss)
    sched.step += 1
    return nothing
end
tracks_losses(::AnnealedWeighted) = true

"""
    TemporalWindow(rerun_steps; radius=2)

Rerun-phase scheduler that pools the loss over the `±radius` temporal
neighbours of the worst timestep. Each rerun iteration runs `2*radius + 1` GNN
forward passes and **one** pooled backward pass — a single gradient update
informed by the whole hard region rather than a single timestep.

The scalar pick is identical to [`WorstLoss`](@ref): a linear base pass (which,
with the `Inf32`-initialised losses buffer, preserves the "uncomputed-first"
invariant) followed by `argmax(losses)` reruns. The *pooling* happens through
the [`window_indices`](@ref) hook, which the training loop expands into a
multi-forward window only during the rerun phase — the base pass and the
norm-accumulation window stay length-1, so initial-pass throughput is unchanged.

!!! note "Naming vs. `DerivativeTraining.window_size`"
    `window_size` is the trajectory-slicing field on `DerivativeTraining` (how
    many timesteps per trajectory). `TemporalWindow`'s `radius` is the
    neighbourhood half-width around the `argmax` pick during a rerun. They are
    unrelated concepts and must not be confused.

!!! warning "Buffer semantics"
    After a windowed rerun, `losses[centre]` holds the *window-pooled mean*, not
    the single-step loss at `centre`; neighbour entries are untouched. Writing
    only the centre index is the cheapest way to advance the `argmax` state
    without a second no-grad forward per member.

!!! warning "Combination guards"
    Requires sequential timesteps, so only supported on
    `DerivativeTraining(; random=false, …)` — with `random=true` the trajectory
    arrays are shuffled in-place and index neighbours no longer correspond to
    adjacent physics steps (the constructor throws via `requires_sequential`).
    Not supported on `BatchingStrategy` (rejected via `pools_in_inner_loop`);
    that strategy already pools temporally through its ODE interval.
"""
struct TemporalWindow <: Scheduler
    rerun_steps::Integer
    radius::Integer
end

function TemporalWindow(rerun_steps::Integer; radius::Integer=2)
    rerun_steps >= 0 || throw(ArgumentError("rerun_steps must be >= 0"))
    radius >= 0 || throw(ArgumentError("radius must be >= 0"))
    return TemporalWindow(rerun_steps, radius)
end

total_iters(sched::TemporalWindow, n_units::Integer) = n_units + sched.rerun_steps

# Scalar pick is identical to WorstLoss: linear base pass / norm-window cycling,
# `argmax` otherwise. The temporal batching is applied separately via
# `window_indices` at the training-loop call site.
function next_index(
    ::TemporalWindow,
    i::Integer,
    n_units::Integer,
    losses::Vector{Float32},
    norm_active::Bool,
)
    return _rerun_next_index(i, n_units, losses, norm_active)
end

function update_state!(
    ::TemporalWindow,
    ::Integer,
    actual::Integer,
    loss::Real,
    ::Integer,
    losses::Vector{Float32},
)
    # `loss` is the window-pooled mean during the rerun phase (length-1 windows
    # on the base pass collapse to the single-step loss). Writing the centre
    # index is enough to advance the argmax state.
    losses[actual] = Float32(loss)
    return nothing
end

tracks_losses(::TemporalWindow) = true
requires_sequential(::TemporalWindow) = true
pools_in_inner_loop(::TemporalWindow) = true

# Temporal neighbourhood `[centre-radius, centre+radius]` clamped to `1:n_units`.
# Boundary clamping silently shrinks the window at trajectory edges.
function window_indices(sched::TemporalWindow, centre::Integer, n_units::Integer)
    lo = max(1, centre - sched.radius)
    hi = min(n_units, centre + sched.radius)
    return Tuple(lo:hi)
end

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

"""
    requires_sequential(sched)

Whether `sched` needs the training units visited in their natural (unshuffled)
order — e.g. schedulers that pool over temporal neighbours. `DerivativeTraining`
rejects such schedulers when `random=true`. Defaults to `false`.
"""
requires_sequential(::Scheduler) = false

"""
    pools_in_inner_loop(sched)

Whether `sched` expands a selected unit into a multi-member [`window_indices`](@ref)
window that is pooled into a single gradient step. `BatchingStrategy` rejects
such schedulers (it already pools temporally via its ODE interval). Defaults to
`false`.
"""
pools_in_inner_loop(::Scheduler) = false

"""
    window_indices(sched, centre, n_units)

Training units pooled into one gradient step when `sched` selects `centre`.
Defaults to the singleton `(centre,)`; only pooling schedulers (e.g.
[`TemporalWindow`](@ref)) override it to return a wider neighbourhood.
"""
window_indices(::Scheduler, centre::Integer, ::Integer) = (centre,)
