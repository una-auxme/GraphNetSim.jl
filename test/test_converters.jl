#
# Copyright (c) 2026 Josef Jouaux, Julian Trommer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
# Tests for the data-import converters `vtk_to_hdf5` and `csv_to_hdf5`.
#
# The VTK fixtures are generated on the fly: leaf `.vtu` pieces are written with
# WriteVTK (whose output ReadVTK is guaranteed to decode), while the `.pvd`/`.pvtu`
# wrappers and the empty (0-point) frames are hand-written text so the tests
# exercise *our* wrapper parsing and empty-frame guard — the parts ReadVTK itself
# cannot handle.
#

using GraphNetSim
using Test
using HDF5
using JSON
using Statistics
using WriteVTK: WriteVTK

# --- Fixture generation -----------------------------------------------------

# Four particles with deliberately shuffled ids so id-sorting is observable.
const _IDS = Int32[40, 10, 30, 20]          # sortperm -> [2, 4, 3, 1]
const _VX = Float32[1, 2, 3, 4]             # per-particle x-velocity (constant in time)
const _TYPES = Int32[2, 2, 5, 5]            # remap {2,5} -> {1,2}
const _BASE_X = Float32[10, 20, 30, 40]     # per-particle starting x
const _PERM = [2, 4, 3, 1]                  # expected id-sorted column order

function _write_leaf(path_noext, positions, vel, types, ids; acc=nothing)
    n = size(positions, 2)
    cells = [WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_VERTEX, (i,)) for i in 1:n]
    vtk = WriteVTK.vtk_grid(path_noext, positions, cells)
    vtk["Vel"] = vel
    vtk["phaseID"] = Int32.(types)
    vtk["prtlID"] = Int32.(ids)
    acc === nothing || (vtk["Acc"] = acc)
    return WriteVTK.vtk_save(vtk)
end

function _write_pvd(path, entries)
    open(path, "w") do io
        println(io, "<VTKFile type=\"Collection\" version=\"1.0\">")
        println(io, "  <Collection>")
        for (t, f) in entries
            println(io, "    <DataSet timestep=\"$t\" group=\"\" part=\"0\" file=\"$f\"/>")
        end
        println(io, "  </Collection>")
        println(io, "</VTKFile>")
    end
end

function _write_pvtu(path, leaf_relfile)
    open(path, "w") do io
        print(
            io,
            """
            <VTKFile type="PUnstructuredGrid" version="1.0">
            <PUnstructuredGrid GhostLevel="0">
            <PPointData>
            <PDataArray type="Float32" Name="Vel" NumberOfComponents="3"/>
            <PDataArray type="Int32" Name="phaseID"/>
            <PDataArray type="Int32" Name="prtlID"/>
            </PPointData>
            <PPoints><PDataArray type="Float32" Name="Points" NumberOfComponents="3"/></PPoints>
            <Piece Source="$leaf_relfile"/>
            </PUnstructuredGrid>
            </VTKFile>
            """,
        )
    end
end

function _write_empty_vtu(path)
    open(path, "w") do io
        print(
            io,
            """
            <VTKFile type="UnstructuredGrid" version="1.0">
             <UnstructuredGrid>
              <Piece NumberOfPoints="0" NumberOfCells="0"></Piece>
             </UnstructuredGrid>
            </VTKFile>
            """,
        )
    end
end

"""Write a minimal legacy BINARY VTK POLYDATA frame (big-endian), DualSPHysics-style."""
function _write_legacy_vtk(path, points, vel, types, ids)
    n = size(points, 2)
    open(path, "w") do io
        write(io, "# vtk DataFile Version 3.0\n")
        write(io, "test output\n")
        write(io, "BINARY\n")
        write(io, "DATASET POLYDATA\n")
        write(io, "POINTS $n float\n")
        for j in 1:n, d in 1:3
            write(io, hton(Float32(points[d, j])))
        end
        write(io, "\nVERTICES $n $(2n)\n")
        for j in 1:n
            write(io, hton(Int32(1)))
            write(io, hton(Int32(j - 1)))
        end
        write(io, "\nPOINT_DATA $n\n")
        write(io, "SCALARS Idp unsigned_int\nLOOKUP_TABLE default\n")
        for j in 1:n
            write(io, hton(UInt32(ids[j])))
        end
        write(io, "\nFIELD FieldData 2\n")
        write(io, "Vel 3 $n float\n")
        for j in 1:n, d in 1:3
            write(io, hton(Float32(vel[d, j])))
        end
        write(io, "\nType 1 $n unsigned_char\n")
        for j in 1:n
            write(io, UInt8(types[j]))
        end
        write(io, "\n")
    end
end

"""Build a synthetic VTK trajectory in `dir`; return the `.pvd` path."""
function _make_vtk_dataset(dir; T=5, dt=0.1, with_pvtu=false, empty_tail=0, with_acc=true)
    n = length(_IDS)
    entries = Tuple{Float64,String}[]
    for t in 1:T
        posx = _BASE_X .+ Float32((t - 1) * dt) .* _VX
        positions = vcat(reshape(posx, 1, n), zeros(Float32, 2, n))
        vel = vcat(reshape(_VX, 1, n), zeros(Float32, 2, n))
        acc = with_acc ? fill(99.0f0, 3, n) : nothing
        _write_leaf(joinpath(dir, "leaf_$t"), positions, vel, _TYPES, _IDS; acc=acc)
        if with_pvtu
            _write_pvtu(joinpath(dir, "part_$t.pvtu"), "leaf_$t.vtu")
            push!(entries, ((t - 1) * dt, "part_$t.pvtu"))
        else
            push!(entries, ((t - 1) * dt, "leaf_$t.vtu"))
        end
    end
    for e in 1:empty_tail
        _write_empty_vtu(joinpath(dir, "empty_$e.vtu"))
        push!(entries, ((T - 1 + e) * dt, "empty_$e.vtu"))
    end
    pvd = joinpath(dir, "particles.pvd")
    _write_pvd(pvd, entries)
    return pvd
end

# --- Tests ------------------------------------------------------------------

@testset "Converters" begin
    println("Running: test_converters.jl")

    @testset "CV1 vtk_to_hdf5 .pvd->.vtu (read Acc, id-sort, remap)" begin
        mktempdir() do dir
            pvd = _make_vtk_dataset(dir; T=5, dt=0.1, with_acc=true)
            out = joinpath(dir, "train.h5")
            vtk_to_hdf5(pvd, out; connectivity_radius=0.5)

            h5open(out, "r") do fid
                @test "trajectory_1" in keys(fid)
                g = fid["trajectory_1"]
                @test read(g["n_particles"]) == 4
                @test read(g["trajectory_length"]) == 5
                @test read(g["dt"]) ≈ 0.1 rtol = 1e-4
                # types remapped {2,5}->{1,2}, reordered by id: [2,5,5,2] -> [1,2,2,1]
                @test read(g["type"]) == [1, 2, 2, 1]
                # velocity x-row reordered by ascending prtlID
                @test vec(read(g["vel[1]"])[1, :]) == _VX[_PERM]
                # position x at t=1 is the (id-sorted) base positions
                @test vec(read(g["pos[1]"])[1, :]) == _BASE_X[_PERM]
                # acceleration read straight from the file (all 99)
                @test all(read(g["acc[1]"]) .== 99.0f0)
                @test size(read(g["acc[1]"])) == (3, 4)
            end
        end
    end

    @testset "CV2 .pvd->.pvtu->.vtu wrapper expansion" begin
        mktempdir() do dir
            pvd = _make_vtk_dataset(dir; T=4, with_pvtu=true, with_acc=true)
            out = joinpath(dir, "train.h5")
            vtk_to_hdf5(pvd, out; connectivity_radius=0.5)
            h5open(out, "r") do fid
                g = fid["trajectory_1"]
                @test read(g["n_particles"]) == 4
                @test read(g["trajectory_length"]) == 4
                @test vec(read(g["vel[1]"])[1, :]) == _VX[_PERM]
            end
        end
    end

    @testset "CV3 trailing empty frames dropped" begin
        mktempdir() do dir
            pvd = _make_vtk_dataset(dir; T=5, empty_tail=3, with_acc=true)
            out = joinpath(dir, "train.h5")
            vtk_to_hdf5(pvd, out; connectivity_radius=0.5)
            h5open(out, "r") do fid
                @test read(fid["trajectory_1"]["trajectory_length"]) == 5  # 3 empties skipped
            end
        end
    end

    @testset "CV4 recompute_acc (kinematic ~0) + central_diff shortens" begin
        mktempdir() do dir
            pvd = _make_vtk_dataset(dir; T=5, dt=0.1, with_acc=true)
            out = joinpath(dir, "train.h5")
            # linear motion -> recomputed acceleration is ~0, unlike the file's 99
            vtk_to_hdf5(
                pvd,
                out;
                recompute_acc=true,
                interpolation_scheme="central_diff",
                connectivity_radius=0.5,
            )
            h5open(out, "r") do fid
                g = fid["trajectory_1"]
                @test read(g["trajectory_length"]) == 3   # central_diff drops 2 of 5
                @test all(abs.(read(g["acc[1]"])) .< 1e-3)
            end
        end
    end

    @testset "CV5 field auto-detect + required-field error" begin
        # alias fallback: requested name absent, resolves via alias table
        @test GraphNetSim._resolve_field(
            "Vel", ["Velocity", "phaseID"]; role=:velocity, required=true
        ) == "Velocity"
        # exact (case-insensitive) match wins
        @test GraphNetSim._resolve_field("vel", ["Vel"]; role=:velocity, required=true) ==
            "Vel"
        # unresolved + required -> throws, listing what's available
        @test_throws ArgumentError GraphNetSim._resolve_field(
            "momentum", ["flux", "phaseID"]; role=:velocity, required=true
        )
        # unresolved + optional -> nothing
        @test GraphNetSim._resolve_field(
            "zzz", ["flux"]; role=:acceleration, required=false
        ) === nothing
    end

    @testset "CV6 meta.json Tier B skeleton" begin
        mktempdir() do dir
            pvd = _make_vtk_dataset(dir; T=5, with_acc=true)
            out = joinpath(dir, "train.h5")
            vtk_to_hdf5(pvd, out; write_meta=true, connectivity_radius=0.5)
            meta = JSON.parsefile(joinpath(dir, "meta.json"))
            @test meta["dims"] == 3
            @test meta["trajectory_length"] == 5
            @test Set(meta["feature_names"]) ==
                Set(["node_type", "position", "velocity", "acceleration"])
            @test meta["features"]["node_type"]["data_max"] == 2   # two distinct types
            @test meta["default_connectivity_radius"] == 0.5
            @test length(meta["bounds"]) == 3
            @test length(meta["features"]["velocity"]["data_mean"]) == 3
            @test all(isfinite, meta["features"]["acceleration"]["data_std"])
        end
    end

    @testset "CV7 meta.json Tier A (provided is validated + copied)" begin
        mktempdir() do dir
            pvd = _make_vtk_dataset(dir; T=5, with_acc=true)
            # first generate a valid meta, stash it away from the dataset dir
            tmp = joinpath(dir, "train0.h5")
            vtk_to_hdf5(pvd, tmp; write_meta=true, connectivity_radius=0.5)
            provided = joinpath(dir, "provided_meta.json")
            cp(joinpath(dir, "meta.json"), provided; force=true)
            rm(joinpath(dir, "meta.json"))

            outdir = mkpath(joinpath(dir, "ds"))
            out = joinpath(outdir, "train.h5")
            vtk_to_hdf5(pvd, out; meta=provided)
            @test isfile(joinpath(outdir, "meta.json"))
            @test JSON.parsefile(joinpath(outdir, "meta.json"))["dims"] == 3
        end
    end

    @testset "CV8 variable particle count is rejected" begin
        mktempdir() do dir
            n = length(_IDS)
            # frame 3 has only 3 points -> assembly must throw
            for t in 1:3
                cnt = t == 3 ? 3 : n
                posx = _BASE_X[1:cnt] .+ Float32((t - 1) * 0.1) .* _VX[1:cnt]
                positions = vcat(reshape(posx, 1, cnt), zeros(Float32, 2, cnt))
                vel = vcat(reshape(_VX[1:cnt], 1, cnt), zeros(Float32, 2, cnt))
                _write_leaf(
                    joinpath(dir, "leaf_$t"), positions, vel, _TYPES[1:cnt], _IDS[1:cnt]
                )
            end
            _write_pvd(
                joinpath(dir, "particles.pvd"),
                [(Float64(t - 1) * 0.1, "leaf_$t.vtu") for t in 1:3],
            )
            @test_throws ArgumentError vtk_to_hdf5(
                joinpath(dir, "particles.pvd"),
                joinpath(dir, "train.h5");
                connectivity_radius=0.5,
            )
        end
    end

    @testset "CV9 vtk dataset loads through Dataset/getobs" begin
        mktempdir() do dir
            pvd = _make_vtk_dataset(dir; T=5, with_acc=true)
            out = joinpath(dir, "train.h5")
            vtk_to_hdf5(
                pvd, out; recompute_acc=true, write_meta=true, connectivity_radius=0.5
            )
            args = GraphNetSim.Args(;
                training_strategy=DerivativeTraining(),
                types_updated=[1],
                types_noisy=[0],
                noise_stddevs=[0.0f0],
                use_cuda=false,
            )
            ds = GraphNetSim.Dataset(:train, dir, args)
            @test GraphNetSim.MLUtils.numobs(ds) == 1
            ds.meta["device"] = identity
            buf = GraphNetSim.MLUtils.getobs(ds, 1)
            @test haskey(buf, "target|acceleration")
            @test size(buf["position"], 2) == 4
        end
    end

    @testset "CV11 legacy BINARY .vtk directory (auto-recompute acc)" begin
        mktempdir() do dir
            ids = Int32[30, 10, 20]          # sortperm -> [2, 3, 1]
            vx = Float32[1, 2, 3]
            types = Int32[2, 2, 5]           # remap {2,5} -> {1,2}
            base_x = Float32[10, 20, 30]
            perm = [2, 3, 1]
            for t in 1:3
                posx = base_x .+ Float32((t - 1) * 0.1) .* vx
                points = vcat(reshape(posx, 1, 3), zeros(Float32, 2, 3))
                vel = vcat(reshape(vx, 1, 3), zeros(Float32, 2, 3))
                _write_legacy_vtk(
                    joinpath(dir, "PartAll_000$t.vtk"), points, vel, types, ids
                )
            end
            out = joinpath(dir, "train.h5")
            # no Acc field in legacy files -> acceleration is auto-recomputed
            vtk_to_hdf5(dir, out; dt=0.1, connectivity_radius=0.5)
            h5open(out, "r") do fid
                g = fid["trajectory_1"]
                @test read(g["n_particles"]) == 3
                @test read(g["trajectory_length"]) == 3           # pchip keeps all frames
                @test read(g["type"]) == [1, 2, 1]                # id-sorted then remapped
                @test vec(read(g["vel[1]"])[1, :]) == vx[perm]    # Vel decoded + id-sorted
                @test vec(read(g["pos[1]"])[1, :]) == base_x[perm]
            end
        end
    end

    @testset "CV10 csv_to_hdf5 round-trip (shared _write_trajectory!)" begin
        mktempdir() do dir
            csv = joinpath(dir, "tiny.csv")
            open(csv, "w") do io
                println(io, "Idp,Type,Points:0,Points:1,Vel:0,Vel:1")
                for t in 0:4  # particle 0 moves +x, particle 1 moves -y
                    println(io, "0,3,$(0.1*t),0.0,1.0,0.0")
                end
                for t in 0:4
                    println(io, "1,7,0.0,$(1.0 - 0.1*t),0.0,-1.0")
                end
            end
            out = joinpath(dir, "csv.h5")
            csv_to_hdf5(csv, out; dt=0.1, dims=[1, 2], interpolation_scheme="pchip")
            h5open(out, "r") do fid
                g = fid["trajectory_1"]
                @test read(g["n_particles"]) == 2
                @test read(g["trajectory_length"]) == 5
                @test read(g["type"]) == [1, 2]                 # {3,7} remapped to {1,2}
                @test vec(read(g["vel[1]"])[:, 1]) == Float32[1.0, 0.0]
                @test size(read(g["acc[1]"])) == (2, 2)
            end
        end
    end
end
