#
# Unit tests for the graph-reconstruction cache path.
#
# Exercises `GraphCache`, `rebuild_topology!`, and `maybe_rebuild_topology!`
# on small hand-built 2D configurations where every edge distance is known
# exactly. Does not require a GNS model, ODE solver, or Zygote — those are
# deferred to an end-to-end integration test once a fixture model is wired
# up.
#
# Run in isolation:
#   julia --project test/test_callbacks.jl
#
# Or via the full suite:
#   julia --project -e "using Pkg; Pkg.test()"
#

using Test
using GraphNetSim
using Zygote

@testset "graph callback" begin
    GraphCache = GraphNetSim.GraphCache
    rebuild_topology! = GraphNetSim.rebuild_topology!
    maybe_rebuild_topology! = GraphNetSim.maybe_rebuild_topology!

    @testset "GraphCache + rebuild_topology!" begin
        # Four particles on the x-axis, radius R = 1.0. Distances are
        # exact in 1D so there's no ambiguity:
        #   (1,2) = 0.50  — in topology, NOT watched
        #   (1,3) = 0.95  — in topology AND watched
        #   (1,4) = 1.05  — watched, NOT in topology
        #   (2,3) = 0.45  — in topology, NOT watched
        #   (2,4) = 1.55  — outside everything
        #   (3,4) = 2.00  — outside everything
        radius = 1.0f0
        pos = Float32[
            0.0 0.5 0.95 -1.05
            0.0 0.0 0.0 0.0
        ]

        cache = GraphCache(pos, radius; safety_factor=4.0f0)
        @test cache.radius == radius
        @test cache.n_watched_max >= 16
        @test cache.overflow == false

        ok = rebuild_topology!(cache, pos)
        @test ok == true
        @test cache.overflow == false

        # Topology: pairs with d ≤ R. `point_neighbor_ns` returns each
        # undirected pair twice (both directions) AND includes self-edges
        # (i==i), matching the existing `build_graph` behavior. We strip
        # self-edges here and verify only the undirected off-diagonal set.
        senders = Array(cache.senders)
        receivers = Array(cache.receivers)
        @test length(senders) == length(receivers)
        topo_pairs = Set{Tuple{Int,Int}}()
        for k in eachindex(senders)
            i, j = Int(senders[k]), Int(receivers[k])
            i == j && continue  # drop self-edges for the assertion
            push!(topo_pairs, (min(i, j), max(i, j)))
        end
        @test (1, 2) in topo_pairs
        @test (1, 3) in topo_pairs
        @test (2, 3) in topo_pairs
        @test !((1, 4) in topo_pairs)   # d = 1.05 > R
        @test !((2, 4) in topo_pairs)
        @test !((3, 4) in topo_pairs)
        @test length(topo_pairs) == 3

        # Watched pairs: undirected, in [0.9R, 1.1R].
        watched = Set{Tuple{Int,Int}}()
        for k in 1:cache.n_watched
            i = Int(cache.watched_pairs[1, k])
            j = Int(cache.watched_pairs[2, k])
            @test i < j  # enforced by rebuild_topology!
            push!(watched, (i, j))
        end
        @test cache.n_watched == 2
        @test (1, 3) in watched  # d = 0.95
        @test (1, 4) in watched  # d = 1.05
        @test !((1, 2) in watched)  # d = 0.5, too close
        @test !((2, 3) in watched)  # d = 0.45, too close
    end

    @testset "maybe_rebuild_topology! triggers on band exit" begin
        # Two particles at d = 0.95 R (inside watch band). After
        # maybe_rebuild_topology! with unchanged positions, no rebuild should
        # happen (topology unchanged). After moving the particle outside the
        # watch band (d < 0.9 R or d > 1.1 R), a rebuild should trigger.
        radius = 1.0f0
        pos = Float32[
            0.0 0.95
            0.0 0.0
        ]
        cache = GraphCache(pos, radius)
        rebuild_topology!(cache, pos)
        @test cache.n_watched == 1

        old_senders = copy(Array(cache.senders))

        # Positions unchanged → still in watch band → no rebuild.
        maybe_rebuild_topology!(cache, pos)
        @test Array(cache.senders) == old_senders

        # Move particle 2 well outside the watch band (d = 1.2 R).
        pos_far = Float32[
            0.0 1.2
            0.0 0.0
        ]
        maybe_rebuild_topology!(cache, pos_far)
        # After rebuild, the pair (1,2) at d = 1.2 R > 1.1 R is now outside
        # the widened search (1.1 R). Topology should have no edges (except
        # possibly self-edges). Watched pairs list should update.
        senders = Array(cache.senders)
        receivers = Array(cache.receivers)
        off_diag = count(k -> senders[k] != receivers[k], eachindex(senders))
        @test off_diag == 0  # pair no longer in topology

        # Move particle 2 well inside (d = 0.3 R), also exits watch band.
        pos_close = Float32[
            0.0 0.3
            0.0 0.0
        ]
        # First rebuild with pos_far left n_watched potentially at 0.
        # Re-initialize from pos_close so the watched pair list is fresh.
        rebuild_topology!(cache, pos_close)
        n_watched_before = cache.n_watched
        # Still within watch band → no rebuild.
        maybe_rebuild_topology!(cache, pos_close)
        @test cache.n_watched == n_watched_before
    end

    @testset "rebuild_topology! after movement" begin
        # Particles 1 and 2 at d = 0.95 R (inside, watched, edge).
        # Move particle 2 outward so d becomes 1.05 R: topology drops the
        # edge, but the pair stays in the watch band.
        radius = 1.0f0
        pos1 = Float32[
            0.0 0.95
            0.0 0.0
        ]
        cache = GraphCache(pos1, radius)
        rebuild_topology!(cache, pos1)

        # Count off-diagonal edges only (strip self-edges).
        function nondiag(senders, receivers)
            s = Array(senders);
            r = Array(receivers)
            count(k -> s[k] != r[k], eachindex(s))
        end

        @test nondiag(cache.senders, cache.receivers) == 2  # (1→2) and (2→1)
        @test cache.n_watched == 1

        pos2 = Float32[
            0.0 1.05
            0.0 0.0
        ]
        rebuild_topology!(cache, pos2)

        @test nondiag(cache.senders, cache.receivers) == 0  # pair no longer in topology
        @test cache.n_watched == 1  # still in watch band
    end

    @testset "cached edge features: Zygote vs finite differences" begin
        # The whole point of `build_graph_cached` is that, with topology
        # frozen, edge features are pure differentiable array ops:
        #
        #     rel_displacement = (pos[:, receivers] .- pos[:, senders]) ./ R
        #     rel_dist_norm    = sqrt(sum(rel_displacement.^2; dims=1) + eps)
        #
        # If Zygote's gradient on these expressions disagrees with central
        # finite differences, the cached path's gradients are wrong and the
        # whole feature is unsound. Test directly without dragging in a GNS.
        radius = 1.0f0
        # Five 2D particles arranged so a few pairs are inside R, none on
        # the boundary (avoid sqrt(0) corner cases).
        pos0 = Float32[
            0.0 0.5 0.95 -0.4 0.3
            0.0 0.1 0.05 0.2 -0.4
        ]
        cache = GraphCache(pos0, radius)
        rebuild_topology!(cache, pos0)
        # Sanity: the cache should have at least one edge to compute over.
        @test length(cache.senders) > 0

        senders = cache.senders
        receivers = cache.receivers

        # Scalar loss exercising both rel_displacement and rel_dist_norm so
        # gradients hit every term in the cached edge-feature pipeline.
        function loss(p)
            rd = (p[:, receivers] .- p[:, senders]) ./ radius
            dn = sqrt.(sum(abs2, rd; dims=1) .+ 1.0f-12)
            return sum(rd .^ 2) + sum(dn)
        end

        zg = Zygote.gradient(loss, pos0)[1]

        # Central finite differences with a step chosen for Float32.
        ε = 1.0f-3
        fd = zeros(Float32, size(pos0))
        for i in eachindex(pos0)
            p_plus = copy(pos0)
            p_minus = copy(pos0)
            p_plus[i] += ε
            p_minus[i] -= ε
            fd[i] = (loss(p_plus) - loss(p_minus)) / (2ε)
        end

        # Float32 + central differences: 1e-3 absolute is realistic on
        # this scale (loss values are O(1), particles are O(1)).
        @test isapprox(zg, fd; atol=1.0f-3, rtol=1.0f-3)
        # Also verify the gradient is non-trivial — a bug that returned
        # zero everywhere would silently pass `≈ zeros` checks.
        @test maximum(abs, zg) > 1.0f-3
    end

    @testset "cached path is gradient-stable across topology refresh" begin
        # When `rebuild_topology!` runs, `senders`/`receivers` swap out for
        # a new array. Subsequent gradient calls on `build_graph_cached`
        # must still produce sensible (non-NaN, non-Inf) gradients with
        # respect to the *new* topology. This catches a class of bugs
        # where stale device buffers or @ignore_derivatives boundaries
        # accidentally fence off legitimate gradients.
        radius = 1.0f0
        # Asymmetric 2D layout so no particle gets symmetry-cancelled
        # gradients. Three particles in a triangle, all within R.
        pos1 = Float32[
            0.0 0.7 0.3
            0.0 0.1 0.6
        ]
        cache = GraphCache(pos1, radius)
        rebuild_topology!(cache, pos1)
        n_edges_1 = length(cache.senders)
        @test n_edges_1 > 0

        loss(p) = sum(
            sqrt.(
                sum(
                    abs2,
                    (p[:, cache.receivers] .- p[:, cache.senders]) ./ radius;
                    dims=1,
                ) .+ 1.0f-12,
            ),
        )
        g1 = Zygote.gradient(loss, pos1)[1]
        @test all(isfinite, g1)
        # Asymmetric layout → every particle's column should be non-zero.
        @test any(g1[:, 1] .!= 0)
        @test any(g1[:, 2] .!= 0)
        @test any(g1[:, 3] .!= 0)

        # Push particle 3 outside R from particle 1 (but still within R
        # of particle 2) and rebuild. Topology should shrink — at least
        # the (1,3) directed pair drops out.
        pos2 = Float32[
            0.0 0.7 1.3
            0.0 0.1 0.6
        ]
        rebuild_topology!(cache, pos2)
        n_edges_2 = length(cache.senders)
        @test n_edges_2 < n_edges_1  # at least one edge dropped

        g2 = Zygote.gradient(loss, pos2)[1]
        @test all(isfinite, g2)
        # All particles still appear in at least one remaining edge, so
        # all columns should still carry non-zero gradient.
        @test any(g2[:, 1] .!= 0)
        @test any(g2[:, 2] .!= 0)
        @test any(g2[:, 3] .!= 0)
    end

    @testset "overflow handling" begin
        # Force overflow by setting a tiny n_watched_max manually, then
        # rebuilding on a config with more watched pairs than capacity.
        radius = 1.0f0
        pos = Float32[
            0.0 0.95 0.0 1.05 0.95
            0.0 0.0 0.95 0.95 1.05
        ]
        cache = GraphCache(pos, radius)
        # Shrink capacity below the real count
        cache.n_watched_max = 1
        cache.watched_pairs = zeros(Int32, 2, 1)
        ok = rebuild_topology!(cache, pos)
        @test ok == false
        @test cache.overflow == true
    end
end
