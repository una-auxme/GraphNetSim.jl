#
# Copyright (c) 2026 Josef Jouaux, Julian Trommer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
# Regression tests for the wall-distance node feature (`_wall_distance`, src/graph.jl).
#
# Guards the boundary-particle-free / implicit-domain-box case. `_wall_distance`
# depends only on `meta["bounds"]` (+ positions) — never on node types or on the
# presence of boundary particles — so it must be computed for EVERY particle and must
# return `2 * dims` rows (low + high bound per spatial dimension), matching the width
# `calc_norms` reserves (`length(meta["bounds"]) * 2`).
#
# The pre-fix code short-circuited to `ones(dims, n)` whenever `length(mask) ==
# size(position, 2)` (a single-type dataset with no boundary particles). That produced
# a constant feature of the WRONG width (`dims` instead of `2*dims`) that disagreed
# with `calc_norms` and threw away the geometric box information the feature exists to
# provide. These tests fail against that old behaviour.

using Test
using GraphNetSim
using JSON

@testset "wall_distance" begin
    device = identity   # keep everything on CPU; `device` only places `meta["bounds"]`
    radius = 1.0f0

    # Three particles, ALL fluid (no boundary particles → mirrors the single-type
    # `mask == 1:n` case). p2 and p3 lie outside the unit box, so the feature must
    # contain signed distances — something the old constant-`ones` fallback can never
    # produce.
    #                    p1    p2    p3
    position = Float32[0.3 -0.5 0.5    # x
                       0.7  0.5 1.5]   # y
    meta = Dict{String,Any}(
        "bounds" => [[0.0, 1.0], [0.0, 1.0]],
        "default_connectivity_radius" => radius,
    )
    dims = length(meta["bounds"])

    wd = Array(GraphNetSim._wall_distance(position, meta, device))

    @testset "shape is 2*dims, not dims" begin
        @test size(wd, 1) == 2 * dims           # regression: old code returned `dims`
        @test size(wd, 2) == size(position, 2)
    end

    @testset "values are the clamped box distances (not constant ones)" begin
        # rows: [low-x; low-y; high-x; high-y]; distances / radius, clamped to [-1, 1].
        expected = Float32[
            0.3 -0.5 0.5     # low-x  = pos_x - lo_x
            0.7  0.5 1.0     # low-y  = pos_y - lo_y   (1.5 clamped to 1.0)
            0.7  1.0 0.5     # high-x = hi_x - pos_x   (1.5 clamped to 1.0)
            0.3  0.5 -0.5    # high-y = hi_y - pos_y
        ]
        @test wd ≈ expected
        @test any(<(0), wd)                     # signed distances, never constant ones
    end

    @testset "missing meta[\"bounds\"] is a hard error" begin
        bad_meta = Dict{String,Any}("default_connectivity_radius" => radius)
        @test_throws ArgumentError GraphNetSim._wall_distance(position, bad_meta, device)
    end
end

# The wall-distance feature is driven purely by `meta["bounds"]`: a bounded dataset
# reserves `2 * dims` extra input rows; an unbounded one yields a shorter vector with no
# error. `calc_norms` only reads `dataset.meta`, so a lightweight `(; meta=...)` suffices.
@testset "wall_distance gating (calc_norms, bounds-driven)" begin
    dev = GraphNetSim.cpu_device()
    args = GraphNetSim.Args()
    base = JSON.parsefile(joinpath(@__DIR__, "fixtures", "ballistic_small", "meta.json"))

    q_bounded, = GraphNetSim.calc_norms((; meta=deepcopy(base)), dev, args)

    unbounded = deepcopy(base)
    delete!(unbounded, "bounds")
    q_unbounded, = GraphNetSim.calc_norms((; meta=unbounded), dev, args)

    @test q_bounded == q_unbounded + 2 * length(base["bounds"])
end

# The only bounds-related error path: loading a model whose boundary feature disagrees
# with the dataset (bounded model + unbounded dataset, or vice versa).
@testset "bounds/model consistency" begin
    @test GraphNetSim._check_bounds_consistency(true, true) === nothing
    @test GraphNetSim._check_bounds_consistency(false, false) === nothing
    @test_throws ArgumentError GraphNetSim._check_bounds_consistency(true, false)
    @test_throws ArgumentError GraphNetSim._check_bounds_consistency(false, true)
end
