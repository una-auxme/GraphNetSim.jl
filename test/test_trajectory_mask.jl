#
# Copyright (c) 2026 Josef Jouaux, Julian Trommer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
# Regression tests for the per-trajectory updated-particle mask
# (`_updated_particle_indices`, src/dataset.jl).
#
# A multi-type dataset (meta declares >1 node type) may contain individual trajectories
# that hold none of the `types_updated` — e.g. an all-boundary scene when
# `types_updated=[1]` (the default). That produces an EMPTY mask, which used to flow
# silently into training/validation as `NaN` losses (mean over zero elements) and zero
# gradient signal. The mask builder now turns that into a clear, actionable error.
# Non-updated particles (boundaries) are expected and fine as long as ≥1 updated
# particle is present.

using Test
using GraphNetSim

@testset "updated-particle mask" begin
    @testset "boundaries are fine when an updated particle exists" begin
        # types_updated=[1]; particles 2,3 are non-updated boundaries — mask = [1, 4].
        @test GraphNetSim._updated_particle_indices([1, 2, 2, 1], [1], 3, "traj_3") ==
            [1, 4]
        # All particles updated (types_updated covers every present type).
        @test GraphNetSim._updated_particle_indices([2, 2], [1, 2], 1, "t") == [1, 2]
    end

    @testset "a trajectory with no updated particles is a hard error" begin
        # Only type-2 particles present, but types_updated=[1] → empty mask.
        @test_throws ArgumentError GraphNetSim._updated_particle_indices(
            [2, 2, 2], [1], 7, "traj_7"
        )
    end
end
