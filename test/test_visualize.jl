#
# Round-trip tests for the VTK HDF5 export (src/visualize.jl).
#
# Build a synthetic trajectory .h5 with the structure `visualize` expects, run
# `visualize` / `visualize_eval`, then re-read the emitted `.vtkhdf` files and
# check the Points/PointData match the input. Only needs HDF5 (no WriteVTK), so
# it runs both standalone under `--project` and inside `Pkg.test`.
#
using GraphNetSim
using Test
using HDF5

# Dataset key exactly as visualize.jl reads it: `name * "[$t]"`.
_dsname(name, t) = name * "[" * string(t) * "]"

# Write an input file laid out as:
#   /<traj>/timesteps                      (scalar upper bound)
#   /<traj>/<subgroup>/<name>[t]           (dim × npts, for t in 1:T)
# Returns a dict (traj, subgroup, name, t) => array of the exact data written.
function _write_synthetic_traj(path; ntraj=1, T=3, npts=5, dim=2, Position="pos", subgroups)
    data = Dict{Tuple{Int,String,String,Int},Matrix{Float64}}()
    HDF5.h5open(path, "w") do f
        for traj in 1:ntraj
            g = HDF5.create_group(f, string(traj))
            g["timesteps"] = T
            for (sg, params) in subgroups
                sgg = HDF5.create_group(g, sg)
                for t in 1:T
                    pos = rand(dim, npts)
                    sgg[_dsname(Position, t)] = pos
                    data[(traj, sg, Position, t)] = pos
                    for p in params
                        v = rand(dim, npts)
                        sgg[_dsname(p, t)] = v
                        data[(traj, sg, p, t)] = v
                    end
                end
            end
        end
    end
    return data
end

@testset "visualize.jl VTK HDF5 export" begin
    println("Running: visualize.jl VTK export round-trip")
    T, npts, dim = 3, 5, 2
    subgroups = Dict("prediction" => ["vel", "acc", "err"], "gt" => ["vel", "acc"])

    mktempdir() do dir
        inPath = joinpath(dir, "trajectories.h5")
        data = _write_synthetic_traj(inPath; T=T, npts=npts, dim=dim, subgroups=subgroups)

        @testset "V1: visualize returns the read datasets" begin
            readDict = visualize(
                inPath, joinpath(dir, "v1"), "pos", "prediction", ["vel", "acc", "err"]
            )
            # pos (1) + 3 params, per timestep
            @test length(readDict) == T * 4
            # params are not padded → exact round-trip of what we wrote
            @test readDict[(1, "vel", 1)] == data[(1, "prediction", "vel", 1)]
            @test readDict[(1, "acc", T)] == data[(1, "prediction", "acc", T)]
        end

        @testset "V2: emitted .vtkhdf re-reads to padded Points + PointData" begin
            outFolder = joinpath(dir, "v2")
            visualize(inPath, outFolder, "pos", "prediction", ["vel", "acc", "err"])
            vtk = joinpath(
                outFolder, "1Trajectory", "prediction", "prediction_1Trajectory_1.vtkhdf"
            )
            @test isfile(vtk)
            HDF5.h5open(vtk, "r") do f
                pts = HDF5.read(f, "VTKHDF/Points")
                @test size(pts) == (3, npts)                       # 2D padded to 3D
                @test pts[1:2, :] == data[(1, "prediction", "pos", 1)]
                @test all(pts[3, :] .== 0)                         # z padding
                @test HDF5.read(f, "VTKHDF/PointData/vel") ==
                    data[(1, "prediction", "vel", 1)]
                @test HDF5.read(f, "VTKHDF/NumberOfPoints") == [npts]
                @test HDF5.read(f, "VTKHDF/NumberOfCells") == [npts]
            end
        end

        @testset "V3: visualize_eval writes both prediction and gt trees" begin
            outFolder = joinpath(dir, "v3")
            visualize_eval(inPath, outFolder)   # defaults: pos / prediction / gt
            @test isfile(
                joinpath(
                    outFolder,
                    "1Trajectory",
                    "prediction",
                    "prediction_1Trajectory_1.vtkhdf",
                ),
            )
            @test isfile(
                joinpath(outFolder, "1Trajectory", "gt", "gt_1Trajectory_$(T).vtkhdf")
            )
        end

        @testset "V4: small point cloud (<= 3 points) Points branch" begin
            small = joinpath(dir, "small.h5")
            _write_synthetic_traj(
                small; T=1, npts=3, dim=2, subgroups=Dict("prediction" => ["vel"])
            )
            out = joinpath(dir, "v4")
            visualize(small, out, "pos", "prediction", ["vel"])
            vtk = joinpath(
                out, "1Trajectory", "prediction", "prediction_1Trajectory_1.vtkhdf"
            )
            @test isfile(vtk)
            HDF5.h5open(vtk, "r") do f
                @test HDF5.read(f, "VTKHDF/NumberOfPoints") == [3]
                @test size(HDF5.read(f, "VTKHDF/Points")) == (3, 3)
            end
        end
    end
end
