#
# Regression tests for the Sanchez-Gonzalez 2020 history-stack input
# (Args.history_size + per-particle C-velocity stacking for DerivativeTraining).
#
# Run in isolation:
#   julia --project test/test_history_stack.jl
#

using Test
using GraphNetSim
using Lux
using CUDA
using JLD2
using JSON
using MLUtils
import OrdinaryDiffEq: Euler, Tsit5
import Optimisers: Adam

include(joinpath(@__DIR__, "generate_fixtures.jl"))

const HAS_CUDA = CUDA.functional()
const DEVICE = HAS_CUDA ? gpu_device() : cpu_device()

const BALLISTIC = joinpath(@__DIR__, "fixtures", "ballistic_small")
const DAM_BREAK = joinpath(@__DIR__, "fixtures", "dam_break_small")

function _args_for(path; history_size=1)
    is_dam_break = path == DAM_BREAK
    return GraphNetSim.Args(;
        use_cuda=HAS_CUDA,
        show_progress_bars=false,
        mps=4,
        layer_size=32,
        hidden_layers=2,
        training_strategy=DerivativeTraining(),
        solver_valid=!is_dam_break && history_size == 1 ? Tsit5() : Euler(),
        solver_valid_dt=is_dam_break ? 0.001f0 : 0.002f0,
        types_updated=is_dam_break ? [2] : [1],
        types_noisy=is_dam_break ? [2] : [1],
        noise_stddevs=[0.0f0],
        norm_steps=0,
        history_size=history_size,
    )
end

function _dam_break_train_kwargs(; history_size=1, kws...)
    base = (
        use_cuda=HAS_CUDA,
        show_progress_bars=false,
        mps=4,
        layer_size=32,
        hidden_layers=2,
        training_strategy=DerivativeTraining(),
        solver_valid=Euler(),
        solver_valid_dt=0.001f0,
        types_updated=[2],
        types_noisy=[2],
        noise_stddevs=[0.0f0],
        norm_steps=0,
        history_size=history_size,
    )
    return merge(base, NamedTuple(kws))
end

@testset "Sanchez-Gonzalez history-stack input" begin
    @testset "H1: history_size=1 keeps quantities unchanged (back-compat)" begin
        # ballistic_small: single-type, input_features=[velocity], no wall_distance.
        # Single-type one-hot is omitted, so quantities = velocity dim = 3.
        args = _args_for(BALLISTIC; history_size=1)
        ds = GraphNetSim.Dataset(:train, BALLISTIC, args)
        ds.meta["device"] = DEVICE
        q, _, _, _ = GraphNetSim.calc_norms(ds, DEVICE, args)
        @test q == 3

        # dam_break_small: 2 types, input_features=[velocity], multi-type wall_distance.
        # quantities = velocity (2) + onehot (2) + 2*length(bounds) (4) = 8.
        args2 = _args_for(DAM_BREAK; history_size=1)
        ds2 = GraphNetSim.Dataset(:train, DAM_BREAK, args2)
        ds2.meta["device"] = DEVICE
        q2, _, _, _ = GraphNetSim.calc_norms(ds2, DEVICE, args2)
        @test q2 == 8
    end

    @testset "H2: history_size=5 quantities formula" begin
        args = _args_for(BALLISTIC; history_size=5)
        ds = GraphNetSim.Dataset(:train, BALLISTIC, args)
        ds.meta["device"] = DEVICE
        q, _, _, _ = GraphNetSim.calc_norms(ds, DEVICE, args)
        # single-type, history=5: 5 * vel_dim(3) = 15
        @test q == 15

        args2 = _args_for(DAM_BREAK; history_size=5)
        ds2 = GraphNetSim.Dataset(:train, DAM_BREAK, args2)
        ds2.meta["device"] = DEVICE
        q2, _, _, _ = GraphNetSim.calc_norms(ds2, DEVICE, args2)
        # multi-type, history=5: 5*2 + 4(walls) + 2(onehot) = 16
        @test q2 == 16
    end

    @testset "H3: training smoke (history_size=5) runs without OOB" begin
        mktempdir() do cp_path
            # dam_break_small: 4 train trajectories, traj_length=150.
            # After C=5 stacking, effective M = 146. Run ~1 outer iteration.
            min_val_loss = train_network(
                Adam(1.0f-3),
                DAM_BREAK,
                cp_path;
                _dam_break_train_kwargs(; history_size=5)...,
                steps=146 * 2,
                checkpoint=146,
            )
            @test isfinite(min_val_loss)

            # Confirm the persisted ModelConfig recorded history_size.
            cfg = GraphNetSim.load_model_config(cp_path)
            @test !isnothing(cfg)
            @test cfg.history_size == 5
        end
    end

    @testset "H4: wall_distance flag on a single-type dataset" begin
        mktempdir() do tmpdir
            for f in ("meta.json", "train.h5", "valid.h5", "test.h5")
                cp(joinpath(BALLISTIC, f), joinpath(tmpdir, f))
            end
            meta = JSON.parsefile(joinpath(tmpdir, "meta.json"))
            push!(meta["input_features"], "wall_distance")
            open(joinpath(tmpdir, "meta.json"), "w") do io
                JSON.print(io, meta, 2)
            end

            args = _args_for(tmpdir; history_size=5)
            ds = GraphNetSim.Dataset(:train, tmpdir, args)
            ds.meta["device"] = DEVICE
            q, _, _, _ = GraphNetSim.calc_norms(ds, DEVICE, args)
            # ballistic dims=3, single-type, history=5, wall_distance enabled:
            # 5*3 + 2*length(bounds)(6) = 21.
            @test q == 21
        end
    end

    @testset "H6: _prepare_rollout_inputs paper-faithful warmup" begin
        args = _args_for(DAM_BREAK; history_size=5)
        ds = GraphNetSim.Dataset(:test, DAM_BREAK, args)
        ds.meta["device"] = DEVICE
        ds.meta["history_size"] = 5
        ds.meta["training_strategy"] = nothing

        # Materialize a single trajectory the way DataLoader would
        traj = MLUtils.getobs(ds, 1)
        traj["dt"] = ds.meta["features"]["acceleration"]["dim"] == 2 ? 0.001f0 : 0.002f0

        # Test the C>1 path: start = (C-1)*dt should produce stepstart = C and
        # velocity_window of shape (dim, np, C) covering frames 1..C.
        dt = 0.001f0
        start = Float32((5 - 1) * dt)
        initial_state, _, stepstart = GraphNetSim._prepare_rollout_inputs(
            traj, ds, start, dt, DEVICE
        )
        @test stepstart == 5
        @test haskey(initial_state, "velocity_window")
        vw = initial_state["velocity_window"]
        @test size(vw, 3) == 5
        @test size(initial_state["position"]) == size(traj["position"])[1:2]
        # vw[:, :, end] must equal data["velocity"][:, :, stepstart]
        last_slice = Array(vw[:, :, end])
        expected = Array(traj["velocity"][:, :, stepstart])
        @test last_slice ≈ expected

        # start < (C-1)*dt must throw (insufficient warmup history)
        @test_throws ArgumentError GraphNetSim._prepare_rollout_inputs(
            traj, ds, 0.0f0, dt, DEVICE
        )

        # C = 1 path: stepstart = 1, no velocity_window
        args1 = _args_for(DAM_BREAK; history_size=1)
        ds1 = GraphNetSim.Dataset(:test, DAM_BREAK, args1)
        ds1.meta["device"] = DEVICE
        ds1.meta["training_strategy"] = nothing
        traj1 = MLUtils.getobs(ds1, 1)
        is1, _, ss1 = GraphNetSim._prepare_rollout_inputs(traj1, ds1, 0.0f0, dt, DEVICE)
        @test ss1 == 1
        @test !haskey(is1, "velocity_window")
    end

    @testset "H5: checkpoint round-trip with history_size=5" begin
        mktempdir() do cp_path
            min1 = train_network(
                Adam(1.0f-3),
                DAM_BREAK,
                cp_path;
                _dam_break_train_kwargs(; history_size=5)...,
                steps=146,
                checkpoint=146,
            )
            cfg = GraphNetSim.load_model_config(cp_path)
            @test !isnothing(cfg)
            @test cfg.history_size == 5

            # Architecture-mismatch guard: same cp_path with a different history_size
            # must error (proves history_size is part of the persisted architecture).
            @test_throws ErrorException train_network(
                Adam(1.0f-3),
                DAM_BREAK,
                cp_path;
                _dam_break_train_kwargs(; history_size=3)...,
                steps=10,
                checkpoint=10,
            )

            # Resume with the original history_size.
            min2 = train_network(
                Adam(1.0f-3),
                DAM_BREAK,
                cp_path;
                _dam_break_train_kwargs(; history_size=5)...,
                steps=146 * 2,
                checkpoint=146,
            )
            @test isfinite(min2)
        end
    end
end
