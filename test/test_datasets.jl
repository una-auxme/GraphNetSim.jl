#
# Unified functional and convergence tests for GraphNetSim.jl.
# Parameterised over datasets so every test group runs for both
# ballistic_small and dam_break_small.
#
# Runs on GPU when CUDA is available, falls back to CPU otherwise.
#
# Run in isolation:
#   julia --project -t 2 test/test_datasets.jl
#
# Or via the full suite:
#   julia --project -e "using Pkg; Pkg.test()"
#

using Test
using GraphNetSim
using Lux
using CUDA
using HDF5
using JLD2
using MLUtils
using DataFrames
using LinearAlgebra
using JSON
using PointNeighbors
import OrdinaryDiffEq: Tsit5, Euler
import Optimisers: Adam
import GraphNetCore

# ─────────────────────────────────────────────────────────────────────────────
# Device detection
# ─────────────────────────────────────────────────────────────────────────────
const HAS_CUDA = CUDA.functional()
const DEVICE = HAS_CUDA ? gpu_device() : cpu_device()

if HAS_CUDA
    @info "CUDA GPU detected — running tests on GPU"
else
    @warn "No CUDA GPU detected — running tests on CPU"
end

# ─────────────────────────────────────────────────────────────────────────────
# Dataset configuration
# ─────────────────────────────────────────────────────────────────────────────
struct DatasetConfig
    name::String
    path::String
    dims::Int
    n_particles::Int
    n_fluid::Int            # particles in mask (types_updated match)
    traj_length::Int
    dt::Float32
    splits::Vector{Tuple{Symbol,Int}}
    types_updated::Vector{Int}
    types_noisy::Vector{Int}
    noise_stddevs::Vector{Float32}
    solver_valid::Any       # ODE solver for validation rollouts
    solver_valid_dt::Float32
    connectivity_radius::Float32
end

const CONFIGS = [
    DatasetConfig(
        "ballistic_small",
        joinpath(@__DIR__, "fixtures", "ballistic_small"),
        3,       # dims
        10,      # n_particles
        10,      # n_fluid (all type=1)
        67,      # traj_length
        0.002f0, # dt
        [(:train, 5), (:valid, 2), (:test, 1)],
        [1],     # types_updated
        [1],     # types_noisy
        [0.0f0], # noise_stddevs
        Tsit5(),
        0.002f0, # solver_valid_dt
        3.9999998989515007f-5,  # connectivity_radius from meta.json
    ),
    DatasetConfig(
        "dam_break_small",
        joinpath(@__DIR__, "fixtures", "dam_break_small"),
        2,       # dims
        47,      # n_particles (38 boundary + 9 fluid)
        9,       # n_fluid (9 fluid type=2)
        150,     # traj_length
        0.001f0, # dt
        [(:train, 4), (:valid, 2), (:test, 1)],
        [2],     # types_updated
        [2],     # types_noisy
        [0.0f0], # noise_stddevs
        Euler(),
        0.001f0, # solver_valid_dt
        0.072f0, # connectivity_radius from meta.json
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers (shared across all datasets)
# ─────────────────────────────────────────────────────────────────────────────

function make_args(cfg::DatasetConfig; kws...)
    return GraphNetSim.Args(;
        use_cuda=HAS_CUDA,
        show_progress_bars=false,
        mps=5,
        layer_size=64,
        hidden_layers=2,
        training_strategy=DerivativeTraining(),
        solver_valid=cfg.solver_valid,
        solver_valid_dt=cfg.solver_valid_dt,
        types_updated=cfg.types_updated,
        types_noisy=cfg.types_noisy,
        noise_stddevs=cfg.noise_stddevs,
        norm_steps=0,
        kws...,
    )
end

function make_train_kwargs(cfg::DatasetConfig; kws...)
    return (
        use_cuda=HAS_CUDA,
        show_progress_bars=false,
        mps=5,
        layer_size=64,
        hidden_layers=2,
        solver_valid=cfg.solver_valid,
        solver_valid_dt=cfg.solver_valid_dt,
        types_updated=cfg.types_updated,
        types_noisy=cfg.types_noisy,
        noise_stddevs=cfg.noise_stddevs,
        norm_steps=0,
        kws...,
    )
end

function load_latest_checkpoint(cp_path)
    cp_files = filter(f -> endswith(f, ".jld2"), readdir(cp_path; join=true))
    @assert !isempty(cp_files) "No checkpoint files found in $cp_path"
    # Sort numerically by step number (checkpoint_67 < checkpoint_335)
    sort!(cp_files; by=f -> parse(Int, match(r"checkpoint_(\d+)", f)[1]))
    data = jldopen(last(cp_files), "r")
    df_train = data["df_train"]
    df_valid = data["df_valid"]
    close(data)
    return df_train, df_valid
end

# ─────────────────────────────────────────────────────────────────────────────
# Main test loop — every group runs for every dataset
# ─────────────────────────────────────────────────────────────────────────────

for cfg in CONFIGS
    @testset "$(cfg.name)" begin
        println("\n═══ Dataset: $(cfg.name) ═══")

        # ─────────────────────────────────────────────────────────────────
        # Group A: Dataset loading & shape contract
        # ─────────────────────────────────────────────────────────────────
        @testset "A: Dataset loading" begin
            println("Running: A — Dataset loading ($(cfg.name))")
            args = make_args(cfg)

            @testset "A1: split sizes" begin
                for (split, expected) in cfg.splits
                    ds = GraphNetSim.Dataset(split, cfg.path, args)
                    @test ds.meta["n_trajectories"] == expected
                end
            end

            @testset "A2-A3: getobs shapes" begin
                ds = GraphNetSim.Dataset(:train, cfg.path, args)
                ds.meta["device"] = DEVICE
                ds.meta["training_strategy"] = nothing
                traj = MLUtils.getobs(ds, 1)

                @test size(traj["position"]) == (cfg.dims, cfg.n_particles, cfg.traj_length)
                @test size(traj["velocity"]) == (cfg.dims, cfg.n_particles, cfg.traj_length)
                @test size(traj["acceleration"]) ==
                    (cfg.dims, cfg.n_particles, cfg.traj_length)
                @test size(traj["node_type"]) == (1, cfg.n_particles, 1)
                @test eltype(traj["position"]) == Float32
                @test eltype(traj["velocity"]) == Float32
                @test eltype(traj["node_type"]) == Int32
                @test traj["n_particles"] == cfg.n_particles
                @test traj["trajectory_length"] == cfg.traj_length
            end

            @testset "A4: mask includes correct particles" begin
                ds = GraphNetSim.Dataset(:train, cfg.path, args)
                ds.meta["device"] = DEVICE
                ds.meta["training_strategy"] = nothing
                traj = MLUtils.getobs(ds, 1)
                @test length(traj["mask"]) == cfg.n_fluid
            end

            @testset "A5: types_updated with absent type throws ArgumentError" begin
                args0 = make_args(cfg; types_updated=[0])
                @test_throws ArgumentError GraphNetSim.Dataset(:train, cfg.path, args0)
            end
        end

        # ─────────────────────────────────────────────────────────────────
        # Group B: Normalization statistics
        # ─────────────────────────────────────────────────────────────────
        @testset "B: Normalization statistics" begin
            println("Running: B — Normalization statistics ($(cfg.name))")
            @testset "B1: data_meanstd finite" begin
                result = data_meanstd(cfg.path)
                for key in ["velocity", "position", "acceleration", "target|acceleration"]
                    @test haskey(result, key)
                    m, s = result[key]
                    @test all(isfinite, m)
                    @test all(isfinite, s)
                    @test all(s .>= 0)
                end
            end

            @testset "B2: feature variance is non-trivial" begin
                result = data_meanstd(cfg.path)
                vel_std = result["velocity"][2]
                acc_std = result["acceleration"][2]
                @test maximum(vel_std) > 0.1f0
                @test maximum(acc_std) > 0.1f0
            end

            @testset "B3: node_type excluded" begin
                result = data_meanstd(cfg.path)
                @test !haskey(result, "node_type")
            end
        end

        # ─────────────────────────────────────────────────────────────────
        # Group C: Neighbor-search sanity
        # ─────────────────────────────────────────────────────────────────
        @testset "C: Neighbor search" begin
            println("Running: C — Neighbor search ($(cfg.name))")
            pos = h5open(joinpath(cfg.path, "train.h5"), "r") do f
                Float32.(read_dataset(open_group(f, "trajectory_1"), "pos[1]"))
            end
            cr = cfg.connectivity_radius

            @testset "C1: particles within connectivity radius" begin
                np = size(pos, 2)
                min_dist = minimum(
                    norm(pos[:, i] - pos[:, j]) for i in 1:np for j in (i + 1):np
                )
                @test min_dist < cr
            end

            @testset "C2: PointNeighbors finds neighbors" begin
                nhs = GridNeighborhoodSearch{size(pos, 1)}(;
                    search_radius=cr, n_points=size(pos, 2)
                )
                initialize!(nhs, pos, pos)
                n_edges = Ref(0)
                foreach_point_neighbor(pos, pos, nhs) do i, j, pos_diff, dist
                    n_edges[] += 1
                end
                @test n_edges[] > 0
            end
        end

        # ─────────────────────────────────────────────────────────────────
        # Group D: Strategy logic (pure logic, no ODE or GPU compute)
        # ─────────────────────────────────────────────────────────────────
        @testset "D: Strategy logic" begin
            println("Running: D — Strategy logic ($(cfg.name))")
            tl = cfg.traj_length

            @testset "D1: get_delta default" begin
                @test GraphNetSim.get_delta(DerivativeTraining(), tl) == tl
            end

            @testset "D2: get_delta with window" begin
                s = DerivativeTraining(; window_size=10)
                @test GraphNetSim.get_delta(s, tl) == 10
                @test GraphNetSim.get_delta(s, 5) == 5
            end

            @testset "D3: batchTrajectory" begin
                batch_steps = 20
                bs = BatchingStrategy(0.0f0, cfg.dt * batch_steps, Euler(), batch_steps)
                data = Dict{String,Any}("dt" => cfg.dt, "trajectory_length" => tl)
                batches = batchTrajectory(bs, data)
                expected_min = div(tl - 1, batch_steps)
                @test length(batches) >= expected_min
                @test all(b.loss == Inf32 for b in batches)
                @test batches[1].batchStart ≈ 0.0f0
            end

            @testset "D4: nextBatch scheduling" begin
                b1 = GraphNetSim.Batch(0.0f0, 0.02f0, Inf32)
                b2 = GraphNetSim.Batch(0.02f0, 0.04f0, 1.5f0)
                b3 = GraphNetSim.Batch(0.04f0, 0.06f0, 3.0f0)
                batches = [b1, b2, b3]
                @test GraphNetSim.nextBatch(batches) == 1    # first uncomputed
                b1.loss = 0.5f0
                @test GraphNetSim.nextBatch(batches) == 3    # highest loss
            end
        end

        # ─────────────────────────────────────────────────────────────────
        # Group E: Training convergence (DerivativeTraining)
        # ─────────────────────────────────────────────────────────────────
        @testset "E: DerivativeTraining convergence" begin
            println("Running: E — DerivativeTraining convergence ($(cfg.name))")
            mktempdir() do cp_path
                # steps must exceed traj_length since DerivativeTraining's
                # delta = traj_length (one full pass per outer loop iteration)
                n_steps = cfg.traj_length * 3
                cp_interval = cfg.traj_length

                min_val_loss = train_network(
                    Adam(1.0f-4),
                    cfg.path,
                    cp_path;
                    make_train_kwargs(cfg)...,
                    training_strategy=DerivativeTraining(),
                    steps=n_steps,
                    checkpoint=cp_interval,
                )

                df_train, df_valid = load_latest_checkpoint(cp_path)

                @test isfinite(min_val_loss)
                @test nrow(df_train) >= 3
                @test last(df_valid.loss) <= first(df_valid.loss)
                @test nrow(df_valid) >= 1
                @test min_val_loss < 1.0f0
            end
        end

        # ─────────────────────────────────────────────────────────────────
        # Group F: SingleShooting smoke test
        # Just verifies that a few training steps complete without error.
        # Uses Tsit5 (adaptive) — Euler can't be used because the
        # InterpolatingAdjoint backward pass doesn't forward dt.
        # ─────────────────────────────────────────────────────────────────
        @testset "F: SingleShooting smoke" begin
            println("Running: F — SingleShooting smoke ($(cfg.name))")
            mktempdir() do cp_path
                n_steps = min(10, cfg.traj_length - 1)
                tstop = cfg.dt * n_steps

                min_val_loss = train_network(
                    Adam(1.0f-4),
                    cfg.path,
                    cp_path;
                    make_train_kwargs(cfg)...,
                    training_strategy=SingleShooting(0.0f0, cfg.dt, tstop, Tsit5()),
                    steps=3,
                    checkpoint=3,
                )

                @test isfinite(min_val_loss)
            end
        end

        # ─────────────────────────────────────────────────────────────────
        # Group G: MultipleShooting smoke test
        # ─────────────────────────────────────────────────────────────────
        @testset "G: MultipleShooting smoke" begin
            println("Running: G — MultipleShooting smoke ($(cfg.name))")
            mktempdir() do cp_path
                n_steps = min(10, cfg.traj_length - 1)
                tstop = cfg.dt * n_steps

                min_val_loss = train_network(
                    Adam(1.0f-4),
                    cfg.path,
                    cp_path;
                    make_train_kwargs(cfg)...,
                    training_strategy=MultipleShooting(0.0f0, cfg.dt, tstop, Tsit5(), 5),
                    steps=3,
                    checkpoint=3,
                )

                @test isfinite(min_val_loss)
            end
        end

        # ─────────────────────────────────────────────────────────────────
        # Group H: Checkpoint resume
        # ─────────────────────────────────────────────────────────────────
        @testset "H: Checkpoint resume" begin
            println("Running: H — Checkpoint resume ($(cfg.name))")
            mktempdir() do cp_path
                # One epoch processes ALL training trajectories, advancing
                # step by n_train * traj_length. Phase2 must exceed phase1's
                # actual step count to verify resumption works.
                n_train = cfg.splits[findfirst(s -> s[1] == :train, cfg.splits)][2]
                epoch_steps = n_train * cfg.traj_length
                cp_interval = cfg.traj_length
                steps_phase1 = epoch_steps      # exactly 1 epoch
                steps_phase2 = epoch_steps * 2  # 2 epochs

                min_val_loss_1 = train_network(
                    Adam(1.0f-4),
                    cfg.path,
                    cp_path;
                    make_train_kwargs(cfg)...,
                    training_strategy=DerivativeTraining(),
                    steps=steps_phase1,
                    checkpoint=cp_interval,
                )
                df_train_1, _ = load_latest_checkpoint(cp_path)

                min_val_loss_2 = train_network(
                    Adam(1.0f-4),
                    cfg.path,
                    cp_path;
                    make_train_kwargs(cfg)...,
                    training_strategy=DerivativeTraining(),
                    steps=steps_phase2,
                    checkpoint=cp_interval,
                )
                df_train_2, _ = load_latest_checkpoint(cp_path)

                @test nrow(df_train_2) > nrow(df_train_1)
                @test last(df_train_2.step) > last(df_train_1.step)
                @test min_val_loss_2 <= min_val_loss_1
            end
        end

        # ─────────────────────────────────────────────────────────────────
        # Group I: Regression anchor — two-arg Dataset constructor bug
        # ─────────────────────────────────────────────────────────────────
        @testset "I: Two-arg Dataset constructor bug (dataset.jl:57)" begin
            println("Running: I — Two-arg Dataset constructor bug ($(cfg.name))")
            args = make_args(cfg)
            @test_throws ArgumentError GraphNetSim.Dataset(
                joinpath(cfg.path, "train.h5"), joinpath(cfg.path, "meta.json"), args
            )
        end

        # ─────────────────────────────────────────────────────────────────
        # Group J: ModelConfig persistence
        # ─────────────────────────────────────────────────────────────────
        @testset "J: ModelConfig persistence" begin
            println("Running: J — ModelConfig persistence ($(cfg.name))")
            j_base = (
                use_cuda=HAS_CUDA,
                show_progress_bars=false,
                steps=5,
                norm_steps=0,
                checkpoint=10000,
                types_updated=cfg.types_updated,
                types_noisy=cfg.types_noisy,
                noise_stddevs=[3.0f-7],
                training_strategy=DerivativeTraining(),
                solver_valid=cfg.solver_valid,
                solver_valid_dt=cfg.solver_valid_dt,
            )

            mktempdir() do cp_path
                @testset "J1: config written on first train_network call" begin
                    train_network(
                        Adam(1.0f-4),
                        cfg.path,
                        cp_path;
                        mps=2,
                        layer_size=32,
                        hidden_layers=1,
                        j_base...,
                    )

                    cfg_path = joinpath(cp_path, "model_config.json")
                    @test isfile(cfg_path)

                    d = JSON.parsefile(cfg_path)
                    @test d["architecture"]["mps"] == 2
                    @test d["architecture"]["layer_size"] == 32
                    @test d["architecture"]["hidden_layers"] == 1
                    @test d["training"]["norm_steps"] == 0
                    @test d["training"]["types_updated"] == cfg.types_updated
                end

                @testset "J2: load_model_config round-trip" begin
                    mc = load_model_config(cp_path)
                    @test !isnothing(mc)
                    @test mc.mps == 2
                    @test mc.layer_size == 32
                    @test mc.hidden_layers == 1
                    @test mc.norm_steps == 0
                    @test mc.noise_stddevs == Float32[3.0f-7]
                end

                @testset "J3: Phase 2 without arch params" begin
                    train_network(Adam(1.0f-6), cfg.path, cp_path; j_base...)
                    mc = load_model_config(cp_path)
                    @test mc.mps == 2
                    @test mc.layer_size == 32
                    @test mc.hidden_layers == 1
                end

                @testset "J4: eval_network without arch params" begin
                    tstart = 0.0f0
                    tstop = cfg.dt * 10
                    saves = collect(tstart:cfg.dt:tstop)
                    eval_path = mktempdir()
                    eval_network(
                        cfg.path,
                        cp_path,
                        eval_path,
                        Tsit5();
                        start=tstart,
                        stop=tstop,
                        dt=cfg.dt,
                        saves=saves,
                        mse_steps=saves,
                        types_updated=cfg.types_updated,
                        use_cuda=HAS_CUDA,
                    )
                    @test !isempty(readdir(eval_path))
                end

                @testset "J5: arch mismatch raises error" begin
                    @test_throws ErrorException train_network(
                        Adam(1.0f-4),
                        cfg.path,
                        cp_path;
                        mps=99,
                        layer_size=32,
                        hidden_layers=1,
                        j_base...,
                    )
                end

                @testset "J6: load_model_config returns nothing when absent" begin
                    @test isnothing(load_model_config(mktempdir()))
                end
            end
        end

        # ─────────────────────────────────────────────────────────────────
        # Group K: Optimizer learning-rate decay
        # ─────────────────────────────────────────────────────────────────
        @testset "K: LR decay" begin
            println("Running: K — LR decay ($(cfg.name))")
            function gns_decay_lr(step, lr_start, lr_stop)
                return Float32(lr_stop + (lr_start - lr_stop) * 0.1^(step / 5.0f6))
            end

            lr_start = 1.0f-4
            lr_stop = 1.0f-6

            @testset "K1: step=0 → lr_start" begin
                @test gns_decay_lr(0, lr_start, lr_stop) ≈ lr_start atol = 1.0f-8
            end

            @testset "K2: step=5e6 → correct intermediate" begin
                expected = lr_stop + (lr_start - lr_stop) * 0.1f0
                @test gns_decay_lr(5_000_000, lr_start, lr_stop) ≈ expected rtol = 1.0f-5
            end

            @testset "K3: large step → lr approaches lr_stop" begin
                lr_large = gns_decay_lr(100_000_000, lr_start, lr_stop)
                @test abs(lr_large - lr_stop) < abs(lr_large - lr_start)
            end

            @testset "K4: monotonically decreasing" begin
                steps = [0, 100_000, 500_000, 1_000_000, 5_000_000]
                lrs = [gns_decay_lr(s, lr_start, lr_stop) for s in steps]
                @test all(lrs[i] >= lrs[i + 1] for i in 1:(length(lrs) - 1))
            end

            @testset "K5: lr_stop=nothing runs without error" begin
                mktempdir() do cp_path
                    n_steps = cfg.traj_length + 1
                    loss = train_network(
                        Adam(1.0f-4),
                        cfg.path,
                        cp_path;
                        make_train_kwargs(cfg)...,
                        training_strategy=DerivativeTraining(),
                        steps=n_steps,
                        checkpoint=n_steps,
                        optimizer_learning_rate_stop=nothing,
                    )
                    @test isfinite(loss)
                end
            end

            @testset "K6: lr_stop != nothing runs without error" begin
                mktempdir() do cp_path
                    n_steps = cfg.traj_length + 1
                    loss = train_network(
                        Adam(1.0f-4),
                        cfg.path,
                        cp_path;
                        make_train_kwargs(cfg)...,
                        training_strategy=DerivativeTraining(),
                        steps=n_steps,
                        checkpoint=n_steps,
                        optimizer_learning_rate_start=1.0f-4,
                        optimizer_learning_rate_stop=1.0f-6,
                    )
                    @test isfinite(loss)
                end
            end
        end

        # ─────────────────────────────────────────────────────────────────
        # Group L: Graph-callback equivalence
        #
        # On a short rollout where no particle pair crosses the search
        # radius, the callback path (`build_graph_cached` on a frozen
        # topology) and the legacy path (`build_graph` rebuilding every
        # step) must produce identical predictions to within float
        # tolerance. The callback's `affect!` should never fire because
        # `condition!` returns no zero crossings.
        #
        # Strategy: train one tiny model with DerivativeTraining, then run
        # `eval_network` twice over the same checkpoint — once with the
        # default no-callback eval path, once with a SolverStrategy that
        # carries `use_graph_callback=true`. Compare returned trajectory
        # arrays.
        # ─────────────────────────────────────────────────────────────────
        @testset "L: Graph-callback equivalence" begin
            println("Running: L — Graph-callback equivalence ($(cfg.name))")
            mktempdir() do cp_path
                # Tiny training run just to produce a checkpoint.
                n_steps = cfg.traj_length + 1
                train_network(
                    Adam(1.0f-4),
                    cfg.path,
                    cp_path;
                    make_train_kwargs(cfg)...,
                    training_strategy=DerivativeTraining(),
                    steps=n_steps,
                    checkpoint=n_steps,
                )

                # Short eval window: 5 steps. Few enough that no crossings
                # occur on the test fixtures (verified empirically — if
                # this test ever flips to crossings, the assertion below
                # would still hold *only* if the callback's edge math
                # tracks the topology rebuild correctly, which is a
                # stronger property we don't claim here).
                tstart = 0.0f0
                tstop = cfg.dt * 5
                saves = collect(tstart:cfg.dt:tstop)
                ms = saves

                # Path A: legacy (use_graph_callback=false on the strategy
                # — also the default when no SolverStrategy is passed).
                eval_path_a = mktempdir()
                traj_a, _ = eval_network(
                    cfg.path,
                    cp_path,
                    eval_path_a,
                    cfg.solver_valid;
                    start=tstart,
                    stop=tstop,
                    dt=cfg.dt,
                    saves=saves,
                    mse_steps=ms,
                    types_updated=cfg.types_updated,
                    use_cuda=HAS_CUDA,
                    training_strategy=SingleShooting(
                        tstart, cfg.dt, tstop, cfg.solver_valid; use_graph_callback=false
                    ),
                )

                # Path B: callback enabled.
                eval_path_b = mktempdir()
                traj_b, _ = eval_network(
                    cfg.path,
                    cp_path,
                    eval_path_b,
                    cfg.solver_valid;
                    start=tstart,
                    stop=tstop,
                    dt=cfg.dt,
                    saves=saves,
                    mse_steps=ms,
                    types_updated=cfg.types_updated,
                    use_cuda=HAS_CUDA,
                    training_strategy=SingleShooting(
                        tstart, cfg.dt, tstop, cfg.solver_valid; use_graph_callback=true
                    ),
                )

                # Compare predictions across all test trajectories.
                @test !isempty(traj_a)
                @test keys(traj_a) == keys(traj_b)
                for k in keys(traj_a)
                    k[2] == "prediction" || continue
                    pred_a = traj_a[k]
                    pred_b = traj_b[k]
                    @test size(pred_a.pos) == size(pred_b.pos)
                    # Tight tolerance: legitimate floating-point reordering
                    # in the cached path comes only from the
                    # `position[:, idx] .- position[:, idx]` indexing,
                    # which is the same arithmetic just laid out
                    # differently — `≈` with default rtol is generous.
                    @test pred_a.pos ≈ pred_b.pos rtol = 1.0f-4
                    @test pred_a.vel ≈ pred_b.vel rtol = 1.0f-4
                end
            end
        end

        # ─────────────────────────────────────────────────────────────────
        # Group M: Gradient equivalence (callback ON vs OFF)
        #
        # The strongest correctness claim of the graph-callback work: when
        # no particle pair crosses the search radius during a solve, the
        # adjoint backward pass through the cached topology must produce
        # gradients identical (within float tolerance) to those from the
        # legacy `build_graph` path that re-runs neighbor search every
        # step. If they diverge, then either:
        #   - the cached differentiable edge-feature math is wrong, or
        #   - the InterpolatingAdjoint reverse pass is reading stale cache
        #     state during the backward sweep,
        # both of which would silently corrupt training gradients.
        #
        # This builds on Group L's prediction equivalence: with predictions
        # already shown to match, gradient equivalence pins down the
        # remaining (more important) half — that *training* with the
        # callback gives the same parameter updates as without.
        # ─────────────────────────────────────────────────────────────────
        @testset "M: Gradient equivalence (callback ON vs OFF)" begin
            println("Running: M — Gradient equivalence ($(cfg.name))")
            mktempdir() do cp_path
                # Train just enough to populate normalizers and have a
                # checkpoint to load from. We don't actually use the
                # trained weights for accuracy — only as a non-trivial
                # parameter set so gradients are non-zero.
                n_steps = cfg.traj_length + 1
                train_network(
                    Adam(1.0f-4),
                    cfg.path,
                    cp_path;
                    make_train_kwargs(cfg)...,
                    training_strategy=DerivativeTraining(),
                    steps=n_steps,
                    checkpoint=n_steps,
                )

                # Reload via the same path eval_network uses, but stop
                # short of running an eval. We need a `gns`, a dataset
                # tuple shaped like the one `init_train_step` consumes,
                # and to call `train_step` directly with two strategies.
                args = make_args(cfg)
                ds = GraphNetSim.Dataset(:train, cfg.path, args)
                ds.meta["device"] = DEVICE
                ds.meta["training_strategy"] = nothing
                quantities, e_norms, n_norms, o_norms = GraphNetSim.calc_norms(
                    ds, DEVICE, args
                )
                outputs = sum(
                    ds.meta["features"][tf]["dim"] for tf in ds.meta["output_features"]
                )
                gns_dim = ds.meta["dims"] isa AbstractArray ? length(ds.meta["dims"]) :
                    ds.meta["dims"]
                gns, _, _, _ = GraphNetSim.load(
                    quantities,
                    gns_dim,
                    e_norms,
                    n_norms,
                    o_norms,
                    outputs,
                    args.mps,
                    args.layer_size,
                    args.hidden_layers,
                    nothing,
                    DEVICE,
                    cp_path,
                )

                # Pull one trajectory and build node_type / mask just like
                # the training loop would.
                data = MLUtils.getobs(ds, 1)
                node_type = DEVICE(
                    Float32.(
                        GraphNetCore.one_hot(
                            vec(data["node_type"][:, :, 1]),
                            ds.meta["features"]["node_type"]["data_max"] -
                            ds.meta["features"]["node_type"]["data_min"] + 1,
                            1 - ds.meta["features"]["node_type"]["data_min"],
                        ),
                    ),
                )

                # Short window: 3 steps to stay within solver stability
                # for untrained models. With a freshly trained tiny model
                # the predicted accelerations are small, so no particle pair
                # should cross the radius.
                tstart = 0.0f0
                tstop = cfg.dt * 3

                function run_train_step(use_cb::Bool)
                    # Always use Tsit5 for adjoint-based training: Euler()
                    # is incompatible with InterpolatingAdjoint (backward
                    # solve doesn't receive dt). Same approach as Groups F/G.
                    strategy = SingleShooting(
                        tstart,
                        cfg.dt,
                        tstop,
                        Tsit5();
                        use_graph_callback=use_cb,
                    )
                    init_in = (
                        gns,
                        data,
                        ds.meta,
                        ds.meta["output_features"],
                        ds.meta["solver_target_features"],
                        node_type,
                        data["mask"],
                        data["val_mask"],
                        DEVICE,
                        nothing,
                        nothing,
                        false,  # show_progress_bars
                    )
                    init_out = GraphNetSim.init_train_step(strategy, init_in)
                    return GraphNetSim.train_step(strategy, init_out)
                end

                gs_off, loss_off = run_train_step(false)
                gs_on, loss_on = run_train_step(true)

                # Both paths must produce the same (possibly Inf) loss.
                @test loss_off ≈ loss_on rtol = 1.0f-4 nans = true

                # Walk both gradient pytrees in lockstep. Use the same
                # leaf iteration as Optimisers/Zygote: ComponentArrays
                # flatten cleanly via `Array(...)`.
                g_off_flat = vec(Array(gs_off[1]))
                g_on_flat = vec(Array(gs_on[1]))
                @test length(g_off_flat) == length(g_on_flat)

                # If both losses are finite AND gradients are non-trivial,
                # compare them tightly. An unstable ODE solve (untrained
                # tiny model) can produce Inf loss and zero gradients —
                # that's a solver issue, not a graph-cache bug. We still
                # verify the two paths agree (both zero).
                if isfinite(loss_off) && maximum(abs, g_off_flat) > 0
                    @test all(isfinite, g_off_flat)
                    @test all(isfinite, g_on_flat)
                    @test maximum(abs, g_off_flat) > 0
                    # Tight tolerance: same arithmetic, same sensitivity
                    # algorithm, just a different topology source.
                    @test isapprox(g_off_flat, g_on_flat; rtol=1.0f-3, atol=1.0f-6)
                else
                    # Solver went unstable — just verify both paths agree.
                    @test isapprox(g_off_flat, g_on_flat; atol=1.0f-6)
                end
            end
        end
    end  # @testset cfg.name
end  # for cfg
