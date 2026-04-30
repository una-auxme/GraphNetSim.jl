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

            @testset "D1b: scheduler defaults" begin
                @test DerivativeTraining().scheduler isa Sequential
                @test GraphNetSim.total_iters(Sequential(), 10) == 10
                @test GraphNetSim.tracks_losses(Sequential()) == false
            end

            @testset "D1c: get_delta invariant across schedulers" begin
                # Schedulers change outer-loop iterations (see E2/E2b) but not
                # validation/data_interval sizing, so get_delta is stable.
                @test GraphNetSim.get_delta(
                    DerivativeTraining(; scheduler=WorstLoss(5)), tl
                ) == tl
                @test GraphNetSim.get_delta(
                    DerivativeTraining(; scheduler=UniqueWorst(5)), tl
                ) == tl
                @test GraphNetSim.get_delta(
                    DerivativeTraining(; window_size=10, scheduler=WorstLoss(3)), tl
                ) == 10
            end

            @testset "D1d: WorstLoss iteration budget + selection" begin
                sched = WorstLoss(5)
                @test GraphNetSim.total_iters(sched, 10) == 15
                @test GraphNetSim.tracks_losses(sched) == true
                # Base pass returns the linear index.
                losses = Float32[1, 9, 3, 2, 7]
                @test GraphNetSim.next_index(sched, 1, 5, losses, false) == 1
                @test GraphNetSim.next_index(sched, 5, 5, losses, false) == 5
                # Rerun phase outside norm_steps: pick argmax.
                @test GraphNetSim.next_index(sched, 6, 5, losses, false) == 2
                # Rerun phase inside norm_steps: wrap around instead of argmax.
                @test GraphNetSim.next_index(sched, 7, 5, losses, true) == 2
                # WorstLoss update_state! just writes the fresh loss.
                GraphNetSim.update_state!(sched, 6, 2, 0.5f0, 5, losses)
                @test losses[2] == 0.5f0
            end

            @testset "D1e: UniqueWorst one-shot selection" begin
                @test GraphNetSim.total_iters(UniqueWorst(5), 10) == 15
                @test GraphNetSim.total_iters(UniqueWorst(20), 10) == 20  # clamped
                @test GraphNetSim.tracks_losses(UniqueWorst(5)) == true

                losses = Float32[3, 1, 2, 5, 4]
                sched = UniqueWorst(3)
                picks = Int[]
                for i in 6:8
                    idx = GraphNetSim.next_index(sched, i, 5, losses, false)
                    push!(picks, idx)
                    GraphNetSim.update_state!(sched, i, idx, 0.0f0, 5, losses)
                end
                @test picks == [4, 5, 1]
                @test allunique(picks)
                @test losses[4] == -Inf32
                @test losses[5] == -Inf32
                @test losses[1] == -Inf32
            end

            @testset "D1f: Inf32 init invariant (WorstLoss rerun pass)" begin
                # The training loop initialises losses to fill(Inf32, n_units).
                # For BatchingStrategy this is load-bearing: when base-pass
                # iterations have only filled some entries, the rerun pass's
                # argmax must still pick remaining Inf32 (uncomputed) entries
                # ahead of finite ones. Regression guard against flipping the
                # init back to zeros or undef.
                sched = WorstLoss(0)
                n = 3
                losses_inf = fill(Inf32, n)
                losses_inf[1] = 0.5f0
                @test GraphNetSim.next_index(sched, 4, n, losses_inf, false) == 2

                losses_zero = zeros(Float32, n)
                losses_zero[1] = 0.5f0
                # With zero-init, argmax instead returns the sole finite entry
                # — confirming the init value matters for rerun-pass semantics.
                @test GraphNetSim.next_index(sched, 4, n, losses_zero, false) == 1
            end

            @testset "D1g: norm phase suppresses scheduler reruns" begin
                # During norm accumulation the training loop forces Sequential
                # to prevent rerun steps from biasing NormaliserOnline stats.
                # We verify this via on_grad callback count: with norm_steps
                # covering the whole run, the loop must execute exactly
                # n_train * traj_length iterations (no reruns), even though
                # WorstLoss(5) would normally add 5 extra per trajectory.
                # The train_loader iterates all training trajectories per
                # epoch, so one full epoch = n_train * traj_length steps.
                mktempdir() do cp_path
                    rerun = 5
                    n_train = cfg.splits[findfirst(s -> s[1] == :train, cfg.splits)][2]
                    base = cfg.traj_length
                    n_steps = base  # one epoch processes n_train trajectories
                    grad_calls = Ref(0)
                    train_network(
                        Adam(1.0f-4),
                        cfg.path,
                        cp_path;
                        make_train_kwargs(cfg)...,
                        training_strategy=DerivativeTraining(; scheduler=WorstLoss(rerun)),
                        steps=n_steps,
                        norm_steps=n_steps * 10,
                        checkpoint=n_steps * 10,
                        on_grad=(step, gs, ps, loss) -> (grad_calls[] += 1),
                    )
                    # Each trajectory gets base gradient calls (no reruns).
                    # The train_loader visits all n_train trajectories before
                    # the while-loop re-checks `step < steps`.
                    @test grad_calls[] == n_train * base
                end
            end

            @testset "D3: batchTrajectory" begin
                batch_steps = 20
                bs = BatchingStrategy(0.0f0, cfg.dt * batch_steps, Euler(), batch_steps)
                data = Dict{String,Any}("dt" => cfg.dt, "trajectory_length" => tl)
                batches = batchTrajectory(bs, data)
                expected_min = div(tl - 1, batch_steps)
                @test length(batches) >= expected_min
                @test batches[1].batchStart ≈ 0.0f0
            end

            @testset "D4a: WorstLoss scheduler contract (Inf32-init base pass)" begin
                sched = WorstLoss(0)
                losses = fill(Inf32, 3)
                # Base pass returns linear i regardless of buffer contents.
                @test GraphNetSim.next_index(sched, 1, 3, losses, false) == 1
                @test GraphNetSim.next_index(sched, 2, 3, losses, false) == 2
                @test GraphNetSim.next_index(sched, 3, 3, losses, false) == 3
            end

            @testset "D4b: BatchingStrategy curriculum regression" begin
                # Drive WorstLoss(0) + Inf32-init buffer by hand and confirm the
                # selected-index sequence matches the old nextBatch curriculum:
                # first uncomputed, then argmax(loss). Guards against a scheduler
                # swap silently altering BatchingStrategy's learning dynamics.
                sched = WorstLoss(0)
                n = 3
                losses = fill(Inf32, n)
                injected = Float32[1.5, 0.3, 2.0, 0.1, 0.4]
                picks = Int[]
                for (i, lv) in enumerate(injected)
                    idx = GraphNetSim.next_index(sched, i, n, losses, false)
                    push!(picks, idx)
                    GraphNetSim.update_state!(sched, i, idx, lv, n, losses)
                end
                # Steps 1-3: base pass picks 1, 2, 3 (filling the Inf32 buffer).
                # Step 4: argmax([1.5, 0.3, 2.0]) = 3 → writes 0.1 to losses[3].
                # Step 5: argmax([1.5, 0.3, 0.1]) = 1.
                @test picks == [1, 2, 3, 3, 1]
            end

            @testset "D6: BatchingStrategy scheduler integration" begin
                bs_default = BatchingStrategy(0.0f0, cfg.dt * 20, Euler(), 20)
                @test bs_default.scheduler isa WorstLoss
                @test GraphNetSim.get_scheduler(bs_default) isa WorstLoss

                # outer_iters for BatchingStrategy always == strategy.steps,
                # ignoring n_batches and the scheduler's own rerun_steps.
                @test GraphNetSim.outer_iters(bs_default, WorstLoss(0), 3) == 20
                @test GraphNetSim.outer_iters(bs_default, UniqueWorst(99), 3) == 20

                bs_unique = BatchingStrategy(
                    0.0f0, cfg.dt * 20, Euler(), 20; scheduler=UniqueWorst(2)
                )
                @test bs_unique.scheduler isa UniqueWorst
                @test bs_unique.scheduler.rerun_steps == 2

                @test GraphNetSim.get_scheduler(
                    SingleShooting(0.0f0, cfg.dt, cfg.dt * 5, Tsit5())
                ) === nothing
                @test GraphNetSim.get_scheduler(
                    MultipleShooting(0.0f0, cfg.dt, cfg.dt * 5, Tsit5(), 2)
                ) === nothing
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
        # Group E2: WorstLoss scheduler smoke test
        # Confirms DerivativeTraining(; scheduler=WorstLoss(…)) runs end-to-end
        # and the outer loop executes `base + rerun_steps` iterations per pass
        # (observed via the on_grad callback count).
        # ─────────────────────────────────────────────────────────────────
        @testset "E2: DerivativeTraining WorstLoss smoke" begin
            println("Running: E2 — DerivativeTraining WorstLoss smoke ($(cfg.name))")
            mktempdir() do cp_path
                rerun = 2
                base = cfg.traj_length
                # 3 full passes of (base + rerun) iterations
                n_steps = (base + rerun) * 3
                cp_interval = base + rerun

                grad_calls = Ref(0)
                min_val_loss = train_network(
                    Adam(1.0f-4),
                    cfg.path,
                    cp_path;
                    make_train_kwargs(cfg)...,
                    training_strategy=DerivativeTraining(; scheduler=WorstLoss(rerun)),
                    steps=n_steps,
                    checkpoint=cp_interval,
                    on_grad=(step, gs, ps, loss) -> (grad_calls[] += 1),
                )

                df_train, df_valid = load_latest_checkpoint(cp_path)

                @test isfinite(min_val_loss)
                @test nrow(df_valid) >= 1
                # on_grad fires once per inner iteration — confirms the outer
                # loop ran base + rerun_steps iterations per trajectory pass.
                @test grad_calls[] >= n_steps
            end
        end

        # ─────────────────────────────────────────────────────────────────
        # Group E2b: UniqueWorst scheduler smoke test
        # Same shape as E2 but with the one-shot top-K scheduler.
        # ─────────────────────────────────────────────────────────────────
        @testset "E2b: DerivativeTraining UniqueWorst smoke" begin
            println("Running: E2b — DerivativeTraining UniqueWorst smoke ($(cfg.name))")
            mktempdir() do cp_path
                rerun = 2
                base = cfg.traj_length
                n_steps = (base + rerun) * 3
                cp_interval = base + rerun

                grad_calls = Ref(0)
                min_val_loss = train_network(
                    Adam(1.0f-4),
                    cfg.path,
                    cp_path;
                    make_train_kwargs(cfg)...,
                    training_strategy=DerivativeTraining(; scheduler=UniqueWorst(rerun)),
                    steps=n_steps,
                    checkpoint=cp_interval,
                    on_grad=(step, gs, ps, loss) -> (grad_calls[] += 1),
                )

                df_train, df_valid = load_latest_checkpoint(cp_path)

                @test isfinite(min_val_loss)
                @test nrow(df_valid) >= 1
                @test grad_calls[] >= n_steps
            end
        end

        # ─────────────────────────────────────────────────────────────────
        # Group E3: BatchingStrategy scheduler end-to-end
        # Drives BatchingStrategy(…; scheduler=UniqueWorst(2)) through the
        # training loop to exercise the unified scheduler dispatch on the
        # ODE-based training path (not just the DerivativeTraining path).
        # ─────────────────────────────────────────────────────────────────
        @testset "E3: BatchingStrategy UniqueWorst smoke" begin
            println("Running: E3 — BatchingStrategy UniqueWorst smoke ($(cfg.name))")
            mktempdir() do cp_path
                batch_steps = 10
                n_steps = 3

                grad_calls = Ref(0)
                min_val_loss = train_network(
                    Adam(1.0f-4),
                    cfg.path,
                    cp_path;
                    make_train_kwargs(cfg)...,
                    training_strategy=BatchingStrategy(
                        0.0f0,
                        cfg.dt * batch_steps,
                        Tsit5(),
                        n_steps;
                        scheduler=UniqueWorst(2),
                    ),
                    steps=n_steps,
                    checkpoint=n_steps,
                    on_grad=(step, gs, ps, loss) -> (grad_calls[] += 1),
                )

                @test isfinite(min_val_loss)
                @test grad_calls[] >= n_steps
            end
        end

        # ─────────────────────────────────────────────────────────────────
        # Group E3b: BatchingStrategy WorstLoss smoke
        # Mirrors E3 but exercises the WorstLoss scheduler (which permits
        # repeats) on the ODE-based path. BatchingStrategy.outer_iters
        # ignores scheduler.rerun_steps and uses strategy.steps for the
        # per-trajectory budget, so we assert grad_calls scales with steps,
        # not with rerun_steps. This closes the coverage gap left by E3
        # (UniqueWorst only).
        # ─────────────────────────────────────────────────────────────────
        @testset "E3b: BatchingStrategy WorstLoss smoke" begin
            println("Running: E3b — BatchingStrategy WorstLoss smoke ($(cfg.name))")
            mktempdir() do cp_path
                batch_steps = 10
                n_steps = 3

                grad_calls = Ref(0)
                min_val_loss = train_network(
                    Adam(1.0f-4),
                    cfg.path,
                    cp_path;
                    make_train_kwargs(cfg)...,
                    training_strategy=BatchingStrategy(
                        0.0f0,
                        cfg.dt * batch_steps,
                        Tsit5(),
                        n_steps;
                        scheduler=WorstLoss(2),
                    ),
                    steps=n_steps,
                    checkpoint=n_steps,
                    on_grad=(step, gs, ps, loss) -> (grad_calls[] += 1),
                )

                @test isfinite(min_val_loss)
                @test grad_calls[] >= n_steps
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
    end  # @testset cfg.name
end  # for cfg
