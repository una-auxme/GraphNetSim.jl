#
# Benchmark: cached (GraphCache) vs legacy (build_graph) path
#
# Measures wall-clock time and memory for:
#   A. Single graph construction  (build_graph vs build_graph_cached)
#   B. Full ODE rollout           (rollout with cache=nothing vs cache=GraphCache)
#
# Uses the dam_break_small fixture (2D, 47 particles, Euler solver) and
# ballistic_small fixture (3D, 10 particles, Tsit5 solver).
#
# Run:
#   julia --project test/bench_callback.jl
#

using Printf
using GraphNetSim
using Lux
using CUDA
using HDF5
using JLD2
using MLUtils
using JSON
using PointNeighbors
import OrdinaryDiffEq: Tsit5, Euler
import Optimisers: Adam
import GraphNetCore

# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────

HAS_CUDA = CUDA.functional()
DEVICE = HAS_CUDA ? gpu_device() : cpu_device()
println("Device: ", HAS_CUDA ? "CUDA GPU" : "CPU")

struct BenchConfig
    name::String
    path::String
    dt::Float32
    solver::Any
    types_updated::Vector{Int}
    types_noisy::Vector{Int}
    noise_stddevs::Vector{Float32}
end

CONFIGS = [
    BenchConfig(
        "ballistic_small",
        joinpath(@__DIR__, "fixtures", "ballistic_small"),
        0.002f0,
        Tsit5(),
        [1], [1], [0.0f0],
    ),
    BenchConfig(
        "dam_break_small",
        joinpath(@__DIR__, "fixtures", "dam_break_small"),
        0.001f0,
        Euler(),
        [2], [2], [0.0f0],
    ),
]

function make_args(cfg::BenchConfig)
    return GraphNetSim.Args(;
        use_cuda=HAS_CUDA,
        show_progress_bars=false,
        mps=5,
        layer_size=64,
        hidden_layers=2,
        training_strategy=GraphNetSim.DerivativeTraining(),
        solver_valid=cfg.solver,
        solver_valid_dt=cfg.dt,
        types_updated=cfg.types_updated,
        types_noisy=cfg.types_noisy,
        noise_stddevs=cfg.noise_stddevs,
        norm_steps=0,
    )
end

# Helper: time a function N times, return (min_time_sec, total_alloc_bytes)
function bench(f, n_warmup=2, n_runs=10)
    # Warmup
    for _ in 1:n_warmup
        f()
        GC.gc()
        HAS_CUDA && CUDA.reclaim()
    end
    # Timed runs
    times = Float64[]
    allocs = Int[]
    for _ in 1:n_runs
        GC.gc()
        HAS_CUDA && CUDA.reclaim()
        stats = @timed f()
        push!(times, stats.time)
        push!(allocs, stats.bytes)
    end
    return (
        min_time=minimum(times),
        median_time=sort(times)[div(length(times), 2) + 1],
        mean_time=sum(times) / length(times),
        min_alloc=minimum(allocs),
        median_alloc=sort(allocs)[div(length(allocs), 2) + 1],
    )
end

function fmt_time(t)
    if t < 1e-3
        return @sprintf("%.1f μs", t * 1e6)
    elseif t < 1.0
        return @sprintf("%.2f ms", t * 1e3)
    else
        return @sprintf("%.3f s", t)
    end
end

function fmt_mem(b)
    if b < 1024
        return @sprintf("%d B", b)
    elseif b < 1024^2
        return @sprintf("%.1f KiB", b / 1024)
    elseif b < 1024^3
        return @sprintf("%.1f MiB", b / 1024^2)
    else
        return @sprintf("%.2f GiB", b / 1024^3)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Run benchmarks
# ─────────────────────────────────────────────────────────────────────────────

for cfg in CONFIGS
    println("\n", "="^70)
    println("BENCHMARK: $(cfg.name)")
    println("="^70)

    # Train a tiny model to get a checkpoint
    args = make_args(cfg)
    ds = GraphNetSim.Dataset(:train, cfg.path, args)
    ds.meta["device"] = DEVICE
    ds.meta["training_strategy"] = nothing

    quantities, e_norms, n_norms, o_norms = GraphNetSim.calc_norms(ds, DEVICE, args)
    outputs = sum(
        ds.meta["features"][tf]["dim"] for tf in ds.meta["output_features"]
    )
    gns_dim = ds.meta["dims"] isa AbstractArray ? length(ds.meta["dims"]) : ds.meta["dims"]

    cp_path = mktempdir()
    n_steps = ds.meta["trajectory_length"] + 1
    GraphNetSim.train_network(
        Adam(1.0f-4),
        cfg.path,
        cp_path;
        use_cuda=HAS_CUDA,
        show_progress_bars=false,
        mps=5,
        layer_size=64,
        hidden_layers=2,
        solver_valid=cfg.solver,
        solver_valid_dt=cfg.dt,
        types_updated=cfg.types_updated,
        types_noisy=cfg.types_noisy,
        noise_stddevs=cfg.noise_stddevs,
        norm_steps=0,
        training_strategy=GraphNetSim.DerivativeTraining(),
        steps=n_steps,
        checkpoint=n_steps,
    )

    # Load model
    gns, _, _, _ = GraphNetSim.load(
        quantities, gns_dim, e_norms, n_norms, o_norms, outputs,
        args.mps, args.layer_size, args.hidden_layers,
        nothing, DEVICE, cp_path,
    )

    # Prepare data
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
    mask = data["mask"]
    val_mask = data["val_mask"]
    radius = Float32(ds.meta["default_connectivity_radius"])
    position = DEVICE(Float32.(data["position"][:, :, 1]))
    velocity = DEVICE(Float32.(data["velocity"][:, :, 1]))

    n_particles = size(data["position"], 2)
    println("  Particles: $n_particles, Dims: $gns_dim, Radius: $radius")

    # ─────────────────────────────────────────────────────────────────────
    # A: Single graph construction
    # ─────────────────────────────────────────────────────────────────────
    println("\n--- A: Single graph construction ---")

    # Legacy path
    legacy_result = bench() do
        GraphNetSim.build_graph(gns, position, velocity, ds.meta, node_type, mask, DEVICE)
    end

    # Cached path (setup cache once, then benchmark build_graph_cached)
    cache = GraphNetSim.GraphCache(position, radius)
    GraphNetSim.rebuild_topology!(cache, position)

    cached_result = bench() do
        GraphNetSim.build_graph_cached(gns, cache, position, velocity, ds.meta, node_type, mask, DEVICE)
    end

    println("  Legacy  (build_graph):        min=$(fmt_time(legacy_result.min_time))  median=$(fmt_time(legacy_result.median_time))  alloc=$(fmt_mem(legacy_result.median_alloc))")
    println("  Cached  (build_graph_cached): min=$(fmt_time(cached_result.min_time))  median=$(fmt_time(cached_result.median_time))  alloc=$(fmt_mem(cached_result.median_alloc))")
    speedup_graph = legacy_result.median_time / cached_result.median_time
    mem_ratio = legacy_result.median_alloc / max(cached_result.median_alloc, 1)
    println("  Speedup: $(round(speedup_graph; digits=2))x   Memory ratio: $(round(mem_ratio; digits=2))x")

    # ─────────────────────────────────────────────────────────────────────
    # B: rebuild_topology! cost (amortized overhead of the cache)
    # ─────────────────────────────────────────────────────────────────────
    println("\n--- B: rebuild_topology! cost ---")

    rebuild_result = bench() do
        GraphNetSim.rebuild_topology!(cache, position)
    end

    println("  rebuild_topology!:            min=$(fmt_time(rebuild_result.min_time))  median=$(fmt_time(rebuild_result.median_time))  alloc=$(fmt_mem(rebuild_result.median_alloc))")

    # ─────────────────────────────────────────────────────────────────────
    # C: maybe_rebuild_topology! (condition check, no rebuild)
    # ─────────────────────────────────────────────────────────────────────
    println("\n--- C: maybe_rebuild_topology! (no-op check) ---")

    # Ensure cache is fresh so condition check doesn't trigger rebuild
    GraphNetSim.rebuild_topology!(cache, position)

    maybe_result = bench(2, 50) do
        GraphNetSim.maybe_rebuild_topology!(cache, position)
    end

    println("  maybe_rebuild (no trigger):   min=$(fmt_time(maybe_result.min_time))  median=$(fmt_time(maybe_result.median_time))  alloc=$(fmt_mem(maybe_result.median_alloc))")

    # ─────────────────────────────────────────────────────────────────────
    # D: Full ODE rollout (eval path)
    # ─────────────────────────────────────────────────────────────────────
    println("\n--- D: Full ODE rollout (eval, 10 steps) ---")

    initial_state = Dict(
        "position" => data["position"][:, :, 1],
        "velocity" => data["velocity"][:, :, 1],
    )
    tstart = 0.0f0
    tstop = cfg.dt * 10
    saves = collect(tstart:cfg.dt:tstop)
    solver_dt = cfg.dt

    # Legacy rollout (no cache)
    legacy_rollout = bench(1, 5) do
        GraphNetSim.rollout(
            cfg.solver, gns, initial_state,
            ds.meta["output_features"], ds.meta,
            ds.meta["solver_target_features"],
            node_type, mask, val_mask,
            tstart, tstop, solver_dt, saves, DEVICE, nothing;
            cache=nothing,
        )
    end

    # Cached rollout
    cached_rollout = bench(1, 5) do
        c = GraphNetSim.GraphCache(DEVICE(Float32.(initial_state["position"])), radius)
        GraphNetSim.rebuild_topology!(c, DEVICE(Float32.(initial_state["position"])))
        GraphNetSim.rollout(
            cfg.solver, gns, initial_state,
            ds.meta["output_features"], ds.meta,
            ds.meta["solver_target_features"],
            node_type, mask, val_mask,
            tstart, tstop, solver_dt, saves, DEVICE, nothing;
            cache=c,
        )
    end

    println("  Legacy  rollout (no cache):   min=$(fmt_time(legacy_rollout.min_time))  median=$(fmt_time(legacy_rollout.median_time))  alloc=$(fmt_mem(legacy_rollout.median_alloc))")
    println("  Cached  rollout (with cache): min=$(fmt_time(cached_rollout.min_time))  median=$(fmt_time(cached_rollout.median_time))  alloc=$(fmt_mem(cached_rollout.median_alloc))")
    speedup_rollout = legacy_rollout.median_time / cached_rollout.median_time
    mem_ratio_rollout = legacy_rollout.median_alloc / max(cached_rollout.median_alloc, 1)
    println("  Speedup: $(round(speedup_rollout; digits=2))x   Memory ratio: $(round(mem_ratio_rollout; digits=2))x")

    # ─────────────────────────────────────────────────────────────────────
    # E: Full ODE rollout (longer, 50 steps)
    # ─────────────────────────────────────────────────────────────────────
    println("\n--- E: Full ODE rollout (eval, 50 steps) ---")

    tstop_long = cfg.dt * 50
    saves_long = collect(tstart:cfg.dt:tstop_long)

    legacy_rollout_long = bench(1, 3) do
        GraphNetSim.rollout(
            cfg.solver, gns, initial_state,
            ds.meta["output_features"], ds.meta,
            ds.meta["solver_target_features"],
            node_type, mask, val_mask,
            tstart, tstop_long, solver_dt, saves_long, DEVICE, nothing;
            cache=nothing,
        )
    end

    cached_rollout_long = bench(1, 3) do
        c = GraphNetSim.GraphCache(DEVICE(Float32.(initial_state["position"])), radius)
        GraphNetSim.rebuild_topology!(c, DEVICE(Float32.(initial_state["position"])))
        GraphNetSim.rollout(
            cfg.solver, gns, initial_state,
            ds.meta["output_features"], ds.meta,
            ds.meta["solver_target_features"],
            node_type, mask, val_mask,
            tstart, tstop_long, solver_dt, saves_long, DEVICE, nothing;
            cache=c,
        )
    end

    println("  Legacy  rollout (no cache):   min=$(fmt_time(legacy_rollout_long.min_time))  median=$(fmt_time(legacy_rollout_long.median_time))  alloc=$(fmt_mem(legacy_rollout_long.median_alloc))")
    println("  Cached  rollout (with cache): min=$(fmt_time(cached_rollout_long.min_time))  median=$(fmt_time(cached_rollout_long.median_time))  alloc=$(fmt_mem(cached_rollout_long.median_alloc))")
    speedup_long = legacy_rollout_long.median_time / cached_rollout_long.median_time
    mem_ratio_long = legacy_rollout_long.median_alloc / max(cached_rollout_long.median_alloc, 1)
    println("  Speedup: $(round(speedup_long; digits=2))x   Memory ratio: $(round(mem_ratio_long; digits=2))x")

    println()
end

println("\nBenchmark complete.")
