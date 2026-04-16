#
# Ballistic small: compare execution times + errors across 4 eval scenarios.
# Trains a fresh model (10k derivative steps), then runs each scenario 3 times.
#

using GraphNetSim
import OrdinaryDiffEq: Euler, Tsit5
import Optimisers: Adam

# ── Generate dataset if needed ────────────────────────────────────────
include(joinpath(dirname(dirname(@__DIR__)), "test", "generators.jl"))
let _gen_dir = joinpath(dirname(dirname(@__DIR__)), "data", "ballistic_small")
    if _needs_generation(_gen_dir)
        @info "Generating dataset: ballistic_small"
        _GenBallistic.generate(_gen_dir)
        @info "  Done."
    end
end

# ── Parameters (from BallisticSmall.jl) ───────────────────────────────

tstart = 0.0f0
dt = 0.002f0
tstop = 0.132f0  # 67 timesteps

ds_path = "data/ballistic_small"
chk_path = "data/ballistic_small/checkpoints_timing_test"

message_steps = 5
layer_size = 64
hidden_layers = 2
types_updated = [1]
types_noisy = [1]
noise_stddevs = [3.0f-7]
norm_steps = 20

saves = tstart:dt:tstop
mse_steps = collect(saves)
n_saves = length(saves)
n_solver_steps = n_saves - 1

# ── Normalization ─────────────────────────────────────────────────────
data_minmax(ds_path; types_updated=types_updated, types_noisy=types_noisy, noise_stddevs=noise_stddevs)
data_meanstd(ds_path; types_updated=types_updated, types_noisy=types_noisy, noise_stddevs=noise_stddevs)

# ── Train fresh model ─────────────────────────────────────────────────
opt = Adam(1.0f-4)
rm(chk_path; force=true, recursive=true)

println("="^60)
println("TRAINING (10k derivative steps, 10 particles)")
println("="^60)

train_network(
    opt, ds_path, chk_path;
    mps=message_steps, layer_size=layer_size, hidden_layers=hidden_layers,
    batchsize=1, epochs=1, steps=10000, use_cuda=true, checkpoint=5000,
    norm_steps=norm_steps,
    types_updated=types_updated, types_noisy=types_noisy, noise_stddevs=noise_stddevs,
    training_strategy=DerivativeTraining(),
    solver_valid=Tsit5(), solver_valid_dt=dt,
    optimizer_learning_rate_start=1.0f-4, optimizer_learning_rate_stop=1.0f-6,
    show_progress_bars=true, save_step=false,
)

# ── Eval scenarios ────────────────────────────────────────────────────

scenarios = [
    ("Euler (no cache)", Euler(), dt, nothing),
    ("Euler (GraphCache)", Euler(), dt,
        BatchingStrategy(tstart, tstop, Tsit5(), 66;
            use_graph_callback=true, graph_callback_safety=4.0f0)),
    ("Tsit5 (no cache)", Tsit5(), dt, nothing),
    ("Tsit5 (GraphCache)", Tsit5(), dt,
        BatchingStrategy(tstart, tstop, Tsit5(), 66;
            use_graph_callback=true, graph_callback_safety=4.0f0)),
]

# ── Warmup ────────────────────────────────────────────────────────────
println("\n" * "="^60)
println("WARMUP (JIT compile all paths)")
println("="^60)
for (i, (label, solver, step_dt, strategy)) in enumerate(scenarios)
    print("  $label ... ")
    kws = Dict{Symbol,Any}(
        :start => tstart, :stop => tstart + 5*dt,
        :saves => tstart:dt:(tstart + 5*dt),
        :dt => step_dt, :mse_steps => Float32[tstart, tstart + 5*dt],
        :mps => message_steps, :types_updated => types_updated,
        :layer_size => layer_size, :hidden_layers => hidden_layers,
        :use_cuda => true, :show_progress_bars => false,
    )
    if !isnothing(strategy)
        kws[:training_strategy] = strategy
    end
    eval_network(ds_path, chk_path, "data/ballistic_small/eval_warmup_$i", solver; kws...)
    println("done")
end

# ── Timed + error runs ───────────────────────────────────────────────
n_repeats = 3
results = []

for (i, (label, solver, step_dt, strategy)) in enumerate(scenarios)
    println("\n" * "="^60)
    println("SCENARIO $i: $label ($n_repeats repeats)")
    println("="^60)

    times = Float64[]
    for rep in 1:n_repeats
        kws = Dict{Symbol,Any}(
            :start => tstart, :stop => tstop, :saves => saves,
            :dt => step_dt, :mse_steps => mse_steps,
            :mps => message_steps, :types_updated => types_updated,
            :layer_size => layer_size, :hidden_layers => hidden_layers,
            :use_cuda => true, :show_progress_bars => false,
        )
        if !isnothing(strategy)
            kws[:training_strategy] = strategy
        end
        t = @elapsed eval_network(
            ds_path, chk_path,
            "data/ballistic_small/eval_time_s$(i)_r$(rep)",
            solver; kws...
        )
        push!(times, t)
        println("  Run $rep: $(round(t; digits=3))s")
    end

    avg = sum(times) / length(times)
    min_t = minimum(times)
    push!(results, (label=label, avg=avg, min=min_t, times=times))
    println("  Average: $(round(avg; digits=3))s | Min: $(round(min_t; digits=3))s")
end

# ── Summary ───────────────────────────────────────────────────────────
println("\n" * "="^70)
println("TIMING SUMMARY  (ballistic_small, 10 particles, $(n_solver_steps) steps)")
println("="^70)
println(rpad("Scenario", 24), rpad("Avg (s)", 12), rpad("Min (s)", 12), "Speedup vs Euler")
println("-"^70)
baseline = results[1].avg
for r in results
    speedup = baseline / r.avg
    println(
        rpad(r.label, 24),
        rpad(string(round(r.avg; digits=3)), 12),
        rpad(string(round(r.min; digits=3)), 12),
        round(speedup; digits=2), "x",
    )
end
println("="^70)
