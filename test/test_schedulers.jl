#
# Pure-unit tests for the Scheduler abstraction (src/schedulers.jl).
#
# These complement the per-scheduler behaviour tests in `test_datasets.jl`
# (groups D1b-D1s), which pin down each concrete's *specific* selection
# pattern. What lives here instead is the shared contract every scheduler
# must satisfy, checked uniformly across all fourteen, plus the edge cases
# that no single-scheduler test covers: degenerate pools, out-of-range
# constructor arguments, and the numeric limits of the weighted samplers.
#
# Everything here is fixture-independent, so unlike the D-groups it runs
# once rather than once per dataset.
#
# Run in isolation:
#   julia --project test/test_schedulers.jl
#

using Test
using GraphNetSim
using Random

const G = GraphNetSim

# Every concrete scheduler, constructed with a non-trivial rerun budget so
# both the base pass and the rerun phase are exercised.
function all_scheds()
    return Any[
        Sequential(),
        Shuffled(),
        WorstLoss(3),
        UniqueWorst(3),
        WeightedWorst(3),
        TopKWorst(3, 2),
        EpsilonGreedy(3),
        PercentileWorst(3),
        UCB(3),
        LearningProgress(3),
        Staleness(3),
        ProgressiveHorizon(3),
        AnnealedWeighted(3),
        TemporalWindow(3),
    ]
end

# The losses buffer the training loop would hand a scheduler: `nothing` when
# the scheduler does not track losses, otherwise a Float32 vector.
buffer_for(sched, v::Vector{Float32}) = G.tracks_losses(sched) ? copy(v) : nothing

sname(sched) = string(nameof(typeof(sched)))

@testset "schedulers" begin
    println("Running: schedulers (unit)")

    # ─────────────────────────────────────────────────────────────────────
    # S1: universal `next_index` contract
    #
    # Whatever the scheduler, the selected unit must be a valid index into
    # the pool. This is the invariant the training loop relies on when it
    # uses the return value to index the trajectory; an out-of-range pick
    # would be a BoundsError deep inside a training run.
    # ─────────────────────────────────────────────────────────────────────
    @testset "S1: next_index always in 1:n_units" begin
        Random.seed!(20250901)
        for sched in all_scheds(), n_units in (1, 2, 5, 17)
            for losses in (
                fill(Inf32, n_units),                        # nothing computed yet
                zeros(Float32, n_units),                     # degenerate: all equal, zero
                fill(2.5f0, n_units),                        # degenerate: all equal, +ve
                Float32.(collect(1:n_units)),                # strictly increasing
                Float32.(reverse(collect(1:n_units))),       # strictly decreasing
            )
                buf = buffer_for(sched, losses)
                for i in 1:(2 * n_units + 3), norm in (false, true)
                    idx = G.next_index(sched, i, n_units, buf, norm)
                    @test idx isa Integer
                    @test 1 <= idx <= n_units
                end
            end
        end
    end

    # ─────────────────────────────────────────────────────────────────────
    # S2: degenerate single-unit pool
    #
    # A trajectory short enough to yield one training unit must not send any
    # scheduler out of range, into a modulo-by-zero, or into an empty
    # `argmax`. Every scheduler has exactly one legal answer here.
    # ─────────────────────────────────────────────────────────────────────
    @testset "S2: n_units == 1 always selects unit 1" begin
        Random.seed!(11)
        for sched in all_scheds()
            buf = buffer_for(sched, Float32[Inf32])
            for i in 1:6
                @test G.next_index(sched, i, 1, buf, false) == 1
            end
            @test G.window_indices(sched, 1, 1) == (1,)
        end
    end

    # ─────────────────────────────────────────────────────────────────────
    # S3: optional traits default to the non-pooling behaviour
    #
    # CLAUDE.md documents that only TemporalWindow overrides these three.
    # If a new scheduler silently picks up `pools_in_inner_loop`, the
    # training loop would hand BatchingStrategy a multi-element window and
    # `only(window)` would throw — so pin the defaults.
    # ─────────────────────────────────────────────────────────────────────
    @testset "S3: pooling traits default false (TemporalWindow excepted)" begin
        for sched in all_scheds()
            pooling = sched isa TemporalWindow
            @test G.requires_sequential(sched) == pooling
            @test G.pools_in_inner_loop(sched) == pooling
            if !pooling
                # Non-pooling schedulers train exactly the centre unit.
                @test G.window_indices(sched, 4, 10) == (4,)
                @test G.window_indices(sched, 1, 10) == (1,)
                @test G.window_indices(sched, 10, 10) == (10,)
            end
        end
    end

    # ─────────────────────────────────────────────────────────────────────
    # S4: `tracks_losses` agrees with what `next_index` actually accepts
    #
    # The training loop allocates the Inf32 buffer only when `tracks_losses`
    # is true and passes `nothing` otherwise. A scheduler that reports
    # `false` but dereferences the buffer would crash only at runtime.
    # ─────────────────────────────────────────────────────────────────────
    @testset "S4: tracks_losses matches buffer usage" begin
        Random.seed!(5)
        for sched in all_scheds()
            @test G.tracks_losses(sched) isa Bool
            if !G.tracks_losses(sched)
                @test G.next_index(sched, 1, 6, nothing, false) in 1:6
                @test G.update_state!(sched, 1, 1, 0.0f0, 6, nothing) === nothing
            end
        end
    end

    # ─────────────────────────────────────────────────────────────────────
    # S5: uncomputed-first (Inf32) invariant, across every loss-driven
    #     scheduler
    #
    # CLAUDE.md makes the Inf32 buffer init load-bearing: a non-finite entry
    # means "never trained", and every loss-driven scheduler must route it
    # through argmax/Inf-priority so no unit is skipped. D1f covers only
    # WorstLoss; this covers all of them.
    # ─────────────────────────────────────────────────────────────────────
    @testset "S5: non-finite entries are selected before finite ones" begin
        Random.seed!(99)
        n_units = 6
        for sched in all_scheds()
            G.tracks_losses(sched) || continue
            sched isa TemporalWindow && continue   # window pick verified in D1r
            # UCB and LearningProgress express the same "never starve a unit"
            # guarantee through their own per-pass visit/progress state rather
            # than by reading the shared buffer (CLAUDE.md: "Inf32-priority for
            # the stateful ones"), so the buffer-driven form below does not
            # apply to them - D1m and D1n cover their variant.
            (sched isa UCB || sched isa LearningProgress) && continue
            # Units 1..4 trained, 5 and 6 never touched.
            losses = Float32[0.1, 0.2, 0.3, 0.4, Inf32, Inf32]
            # Drive the rerun phase (i > n_units) and confirm an untrained
            # unit wins over any finite loss.
            picks = [
                G.next_index(sched, n_units + 1, n_units, copy(losses), false) for _ in 1:40
            ]
            @test all(p -> p in (5, 6), picks)
        end
    end

    # ─────────────────────────────────────────────────────────────────────
    # S6: norm-accumulation window never consults the losses buffer
    #
    # During the online-normaliser window the loop wants plain cycling so a
    # loss-driven scheduler cannot revisit one timestep and bias
    # NormaliserOnline's running stats (the D1g hazard). With `norm=true`
    # the pick must be positional, i.e. identical for any loss buffer.
    # ─────────────────────────────────────────────────────────────────────
    @testset "S6: norm phase ignores loss values" begin
        n_units = 5
        flat = Float32[1, 1, 1, 1, 1]
        spiky = Float32[1, 1, 1, 1, 1000]
        for sched in all_scheds()
            G.tracks_losses(sched) || continue
            for i in (n_units + 1):(n_units + 4)
                a = G.next_index(sched, i, n_units, copy(flat), true)
                b = G.next_index(sched, i, n_units, copy(spiky), true)
                @test a == b
            end
        end
    end

    # ─────────────────────────────────────────────────────────────────────
    # S7: iteration budget
    # ─────────────────────────────────────────────────────────────────────
    @testset "S7: total_iters covers the base pass" begin
        for sched in all_scheds(), n_units in (1, 4, 9)
            @test G.total_iters(sched, n_units) >= n_units
        end
        # UniqueWorst cannot rerun a unit twice, so its budget saturates.
        @test G.total_iters(UniqueWorst(50), 6) == 12
        @test G.total_iters(WorstLoss(50), 6) == 56
    end

    # ─────────────────────────────────────────────────────────────────────
    # S8: weighted samplers converge on argmax as temperature falls
    #
    # Regression test for a real defect: `_weighted_pick` raises each loss to
    # 1/T, and for small T that overflows Float64 to Inf. The cumulative
    # scan then matched `Inf <= Inf` at the *first* overflowing index and
    # returned it — which equals argmax only by accident of ordering. The
    # loss vector below is ordered so the first overflowing entry (index 1)
    # is NOT the argmax (index 4), which is exactly the case the old code
    # got wrong.
    # ─────────────────────────────────────────────────────────────────────
    @testset "S8: low temperature converges on WorstLoss" begin
        Random.seed!(4)
        losses = Float32[3, 1, 2, 5, 4]      # argmax == 4, index 1 overflows first
        @test argmax(losses) == 4
        for T in Float32[0.01, 0.002, 0.001, 0.0005, 0.0]
            picks = [G._weighted_pick(copy(losses), T) for _ in 1:60]
            @test all(==(4), picks)
        end
        # AnnealedWeighted reaches the same regime once it has annealed to
        # its floor temperature, and must agree.
        sched = AnnealedWeighted(5; t0=2.0f0, t1=1.0f-4, decay_steps=1)
        sched.step = 1
        @test all(G.next_index(sched, 6, 5, copy(losses), false) == 4 for _ in 1:60)
    end

    @testset "S9: high temperature still spreads across units" begin
        Random.seed!(7)
        losses = Float32[3, 1, 2, 5, 4]
        picks = [G._weighted_pick(copy(losses), 5.0f0) for _ in 1:400]
        # A hot sampler must not collapse onto one unit.
        @test length(unique(picks)) >= 3
        # ...and must still respect the pool bounds.
        @test all(p -> 1 <= p <= 5, picks)
    end

    # ─────────────────────────────────────────────────────────────────────
    # S10: all-equal / all-zero loss buffers stay in range
    #
    # `_weighted_pick` divides by the total weight; an all-zero buffer makes
    # that zero, and negative entries are clamped to zero, so both land in
    # the uniform-fallback branch.
    # ─────────────────────────────────────────────────────────────────────
    @testset "S10: zero and negative losses fall back to a valid pick" begin
        Random.seed!(8)
        for v in (zeros(Float32, 5), Float32[-1, -2, -3, -4, -5])
            picks = [G._weighted_pick(copy(v), 1.0f0) for _ in 1:60]
            @test all(p -> 1 <= p <= 5, picks)
        end
    end

    # ─────────────────────────────────────────────────────────────────────
    # S11: TopKWorst clamps k into 1:n_units
    #
    # k is user-supplied and documented as clamped. k=0 must not produce an
    # empty candidate set, and k > n_units must not index past the pool.
    # ─────────────────────────────────────────────────────────────────────
    @testset "S11: TopKWorst clamps k" begin
        losses = Float32[3, 1, 2, 5, 4]
        for k in (0, 1, 5, 99)
            sched = TopKWorst(3, k)
            for i in 6:9
                @test G.next_index(sched, i, 5, copy(losses), false) in 1:5
            end
        end
        # k <= 1 degenerates to WorstLoss (pure argmax).
        @test G.next_index(TopKWorst(3, 1), 6, 5, copy(losses), false) == 4
        @test G.next_index(TopKWorst(3, 0), 6, 5, copy(losses), false) == 4
    end

    # ─────────────────────────────────────────────────────────────────────
    # S12: EpsilonGreedy endpoints
    # ─────────────────────────────────────────────────────────────────────
    @testset "S12: EpsilonGreedy epsilon endpoints" begin
        Random.seed!(2)
        losses = Float32[3, 1, 2, 5, 4]
        # epsilon = 0 is documented as equivalent to WorstLoss.
        @test all(
            G.next_index(EpsilonGreedy(3; epsilon=0.0f0), 6, 5, copy(losses), false) == 4
            for _ in 1:50
        )
        # epsilon = 1 explores every time but stays in range.
        picks = [
            G.next_index(EpsilonGreedy(3; epsilon=1.0f0), 6, 5, copy(losses), false) for
            _ in 1:200
        ]
        @test all(p -> 1 <= p <= 5, picks)
        @test length(unique(picks)) >= 3
    end

    # ─────────────────────────────────────────────────────────────────────
    # S13: per-pass state resets when the loop wraps to i == 1
    #
    # UCB / LearningProgress / Staleness keep per-unit state in mutable
    # fields because the interface passes no cross-pass history. That state
    # must be rebuilt at the start of each pass, mirroring the loop's fresh
    # Inf32 losses buffer -- otherwise visit counts leak across passes and
    # the bandit stops exploring.
    # ─────────────────────────────────────────────────────────────────────
    @testset "S13: stateful schedulers reset at i == 1" begin
        n_units = 3
        for sched in (UCB(2), LearningProgress(2), Staleness(2))
            losses = fill(Inf32, n_units)
            first_pass = Int[]
            for i in 1:G.total_iters(sched, n_units)
                idx = G.next_index(sched, i, n_units, losses, false)
                push!(first_pass, idx)
                G.update_state!(sched, i, idx, Float32(idx), n_units, losses)
            end
            # A fresh pass over a fresh buffer must reproduce the first pass.
            losses2 = fill(Inf32, n_units)
            second_pass = Int[]
            for i in 1:G.total_iters(sched, n_units)
                idx = G.next_index(sched, i, n_units, losses2, false)
                push!(second_pass, idx)
                G.update_state!(sched, i, idx, Float32(idx), n_units, losses2)
            end
            @test first_pass == second_pass
        end
    end

    # ─────────────────────────────────────────────────────────────────────
    # S14: global step counters advance only via update_state!
    #
    # ProgressiveHorizon and AnnealedWeighted anneal over the whole run, so
    # unlike the per-pass schedulers their counters must NOT reset at i == 1.
    # ─────────────────────────────────────────────────────────────────────
    @testset "S14: annealing counters persist across passes" begin
        for sched in
            (ProgressiveHorizon(0; growth_steps=8), AnnealedWeighted(0; decay_steps=8))
            buf = buffer_for(sched, fill(1.0f0, 4))
            @test sched.step == 0
            for i in 1:4
                G.update_state!(sched, i, i, 1.0f0, 4, buf)
            end
            @test sched.step == 4
            # Wrapping to a new pass must not rewind the global counter.
            G.next_index(sched, 1, 4, buf, false)
            @test sched.step == 4
        end
    end

    # ─────────────────────────────────────────────────────────────────────
    # S15: TemporalWindow window geometry
    #
    # D1r covers the scalar pick; this covers clipping at both pool edges,
    # which is where an off-by-one would silently train the wrong timesteps.
    # ─────────────────────────────────────────────────────────────────────
    @testset "S15: TemporalWindow clips at pool edges" begin
        sched = TemporalWindow(2; radius=2)
        @test G.window_indices(sched, 5, 10) == (3, 4, 5, 6, 7)
        # Clipped low: no index below 1.
        w_lo = G.window_indices(sched, 1, 10)
        @test minimum(w_lo) == 1
        @test issorted(collect(w_lo))
        @test all(i -> 1 <= i <= 10, w_lo)
        # Clipped high: no index above n_units.
        w_hi = G.window_indices(sched, 10, 10)
        @test maximum(w_hi) == 10
        @test all(i -> 1 <= i <= 10, w_hi)
        # A window wider than the pool collapses to the whole pool.
        wide = G.window_indices(TemporalWindow(1; radius=50), 2, 4)
        @test collect(wide) == [1, 2, 3, 4]
        # radius 0 is a single unit, matching the non-pooling default shape.
        @test G.window_indices(TemporalWindow(1; radius=0), 4, 10) == (4,)
    end

    # ─────────────────────────────────────────────────────────────────────
    # S16: constructor argument validation
    #
    # TemporalWindow is currently the only scheduler that rejects bad input.
    # ─────────────────────────────────────────────────────────────────────
    @testset "S16: TemporalWindow rejects negative arguments" begin
        @test_throws ArgumentError TemporalWindow(-1)
        @test_throws ArgumentError TemporalWindow(1; radius=-1)
        @test TemporalWindow(0; radius=0) isa TemporalWindow
    end

    # ─────────────────────────────────────────────────────────────────────
    # S17: update_state! is total over the pool
    #
    # The loop calls update_state! with whatever index next_index returned,
    # so every scheduler must accept every in-range index without resizing
    # or reordering the caller's buffer.
    # ─────────────────────────────────────────────────────────────────────
    @testset "S17: update_state! preserves buffer length" begin
        n_units = 5
        for sched in all_scheds()
            buf = buffer_for(sched, fill(Inf32, n_units))
            # Follow the loop's protocol: a pass always starts at i == 1, which
            # is where the stateful schedulers (UCB, LearningProgress,
            # Staleness) allocate their per-unit state. Calling update_state!
            # before any next_index would hit a zero-length state vector.
            for i in 1:n_units
                idx = G.next_index(sched, i, n_units, buf, false)
                G.update_state!(sched, i, idx, Float32(idx), n_units, buf)
            end
            if buf !== nothing
                @test length(buf) == n_units
                @test all(isfinite, buf)
            end
        end
    end
end
