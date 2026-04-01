#
# Normalizer correctness tests for GraphNetSim.jl.
#
# Groups A, B: CPU only — pure math, no dataset, no GPU.
# Groups C, D, E: CPU only — use ballistic_small dataset via calc_norms.
# Group F: GPU required — checkpoint persistence of online normalizer state.
#
# Run in isolation:
#   julia --project test/test_normalizer.jl
#
# Or via the full suite:
#   julia --project -e "using Pkg; Pkg.test()"
#

using Test
using Statistics
using GraphNetSim
using GraphNetCore
using Lux
using CUDA
using JLD2
using MLUtils
import Optimisers
import OrdinaryDiffEq

const NORM_DATA_PATH = joinpath(@__DIR__, "fixtures", "ballistic_small")
device = cpu_device()

@testset "normalizer" begin

    # ─────────────────────────────────────────────────────────────────────────
    # Group A — Unit math, all three normalizer types (CPU, no dataset)
    # ─────────────────────────────────────────────────────────────────────────
    @testset "A: NormaliserOfflineMinMax unit math" begin
        println("Running: A — NormaliserOfflineMinMax unit math")
        @testset "A1: data_min→0, data_max→1, midpoint→0.5" begin
            n = NormaliserOfflineMinMax(-3.0f0, 5.0f0)
            @test only(n([-3.0f0])) ≈ 0.0f0  atol=1f-6
            @test only(n([5.0f0]))  ≈ 1.0f0  atol=1f-6
            @test only(n([1.0f0]))  ≈ 0.5f0  atol=1f-6   # midpoint: (-3+5)/2 = 1
        end

        @testset "A2: inverse_data is exact left-inverse" begin
            n = NormaliserOfflineMinMax(-3.0f0, 5.0f0)
            x = Float32[-3.0, -1.0, 0.0, 2.5, 5.0]
            @test all(isapprox.(inverse_data(n, n(x)), x; atol=1f-6))
        end

        @testset "A3: non-default target range [-1, 1]" begin
            n = NormaliserOfflineMinMax(0.0f0, 10.0f0, -1.0f0, 1.0f0)
            @test only(n([0.0f0]))  ≈ -1.0f0 atol=1f-6
            @test only(n([10.0f0])) ≈  1.0f0 atol=1f-6
            @test only(n([5.0f0]))  ≈  0.0f0 atol=1f-6
            x = Float32[0.0, 3.0, 7.0, 10.0]
            @test all(isapprox.(inverse_data(n, n(x)), x; atol=1f-6))
        end

        @testset "A4: NormaliserOfflineMeanStd mean→0, mean±std→±1" begin
            n = NormaliserOfflineMeanStd(2.0f0, 3.0f0, device)
            @test only(n([2.0f0]))  ≈  0.0f0 atol=1f-6   # mean → 0
            @test only(n([5.0f0]))  ≈  1.0f0 atol=1f-6   # mean + std → +1
            @test only(n([-1.0f0])) ≈ -1.0f0 atol=1f-6   # mean - std → -1
        end

        @testset "A5: NormaliserOfflineMeanStd inverse_data is exact left-inverse" begin
            n = NormaliserOfflineMeanStd(2.0f0, 3.0f0, device)
            x = Float32[-5.0, -1.0, 2.0, 5.0, 8.0]
            @test all(isapprox.(inverse_data(n, n(x)), x; atol=1f-6))
        end

        @testset "A6: std_epsilon prevents div-by-zero when std=0" begin
            n = NormaliserOfflineMeanStd(1.0f0, 0.0f0, device)
            result = n([1.0f0])
            @test all(isfinite, result)
            @test only(result) ≈ 0.0f0 atol=1f-6        # (1 - 1) / eps ≈ 0
            @test only(inverse_data(n, result)) ≈ 1.0f0 atol=1f-5
        end
    end

    # ─────────────────────────────────────────────────────────────────────────
    # Group B — NormaliserOnline state machine (CPU, synthetic data)
    # ─────────────────────────────────────────────────────────────────────────
    @testset "B: NormaliserOnline state machine" begin
        println("Running: B — NormaliserOnline state machine")
        @testset "B1: fresh normalizer has zero counters" begin
            n = NormaliserOnline(3, cpu_device())
            @test n.num_accumulations == 0.0f0
            @test n.acc_count         == 0.0f0
            @test all(iszero, n.acc_sum)
            @test all(iszero, n.acc_sum_squared)
        end

        @testset "B2: one call accumulates correct acc_count and acc_sum" begin
            n = NormaliserOnline(2, cpu_device(); max_acc=10.0f0)
            # Shape (2, 4): 2-dim feature, 4 samples
            data = Float32[1.0 3.0 5.0 7.0;
                           2.0 4.0 6.0 8.0]
            n(data)
            @test n.num_accumulations ≈ 1.0f0
            @test n.acc_count ≈ 4.0f0
            # acc_sum = row sums: [1+3+5+7, 2+4+6+8] = [16, 20]
            @test n.acc_sum == Float32[16.0, 20.0]
        end

        @testset "B3: normalized midpoint ≈ 0 after warm-up" begin
            n = NormaliserOnline(1, cpu_device(); max_acc=100.0f0)
            for v in 1.0f0:1.0f0:100.0f0
                n(reshape([v], 1, 1))
            end
            # mean ≈ 50.5; normalizing 50.5 should give ≈ 0
            result = n(reshape([50.5f0], 1, 1), false)
            @test abs(result[1]) < 0.01f0
        end

        @testset "B4: accumulation stops at max_accumulations" begin
            n = NormaliserOnline(1, cpu_device(); max_acc=3.0f0)
            for _ in 1:5
                n(reshape([1.0f0], 1, 1))
            end
            @test n.num_accumulations == 3.0f0
        end

        @testset "B5: acc=false does not mutate state" begin
            n = NormaliserOnline(2, cpu_device(); max_acc=10.0f0)
            data = Float32[1.0 2.0; 3.0 4.0]
            n(data)
            count_before = n.num_accumulations
            sum_before   = copy(n.acc_sum)
            n(data, false)
            @test n.num_accumulations == count_before
            @test n.acc_sum == sum_before
        end

        @testset "B6: serialize/deserialize round-trip is lossless" begin
            n = NormaliserOnline(3, cpu_device(); max_acc=5.0f0)
            data = Float32[1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]
            n(data); n(data .* 2)

            d  = GraphNetCore.serialize(n)
            n2 = GraphNetCore.deserialize(d, cpu_device())

            @test n2.max_accumulations  == n.max_accumulations
            @test n2.num_accumulations  == n.num_accumulations
            @test n2.acc_count          == n.acc_count
            @test n2.acc_sum         == n.acc_sum
            @test n2.acc_sum_squared == n.acc_sum_squared

            # Applying both to the same input should give identical output
            x = reshape(Float32[1.5, 4.5, 7.5], 3, 1)
            @test n(x, false) == n2(x, false)
        end
    end

    # ─────────────────────────────────────────────────────────────────────────
    # Group C — calc_norms selects correct normalizer types (CPU, ballistic_small)
    # ─────────────────────────────────────────────────────────────────────────
    @testset "C: calc_norms selection from meta.json" begin
        println("Running: C — calc_norms selection from meta.json")
        args = GraphNetSim.Args(;
            use_cuda=false,
            show_progress_bars=false,
            norm_steps=50,
            norm_type=:meanstd,
            mps=2, layer_size=16, hidden_layers=1,
            training_strategy=DerivativeTraining(),
            solver_valid=OrdinaryDiffEq.Tsit5(),
            solver_valid_dt=0.002f0,
            types_updated=[1],
            types_noisy=[0],
            noise_stddevs=[0.0f0],
        )
        ds = GraphNetSim.Dataset(:train, NORM_DATA_PATH, args)
        ds.meta["types_updated"] = args.types_updated
        ds.meta["types_noisy"]   = args.types_noisy
        ds.meta["noise_stddevs"] = args.noise_stddevs
        ds.meta["device"]        = cpu_device()
        ds.meta["training_strategy"] = nothing

        _, e_norms, n_norms, o_norms = GraphNetSim.calc_norms(ds, cpu_device(), args)

        @testset "C1: e_norms is NormaliserOnline with default max_acc" begin
            @test e_norms isa NormaliserOnline
            # calc_norms creates e_norms with the default max_acc (1e7), not norm_steps
            @test e_norms.max_accumulations == 1.0f7
        end

        @testset "C2: node_type absent from n_norms (normaliser is identity, never created)" begin
            @test !haskey(n_norms, "node_type")
        end

        @testset "C3: velocity normalizer is NormaliserOfflineMeanStd; data_mean is per-dim Vector" begin
            @test haskey(n_norms, "velocity")
            @test n_norms["velocity"] isa NormaliserOfflineMeanStd
            # calc_norms passes Float32.(vector) from meta.json, so data_mean holds a
            # Vector{Float32} (not a scalar Float32) — document actual runtime behavior.
            @test n_norms["velocity"].data_mean isa Vector{Float32}
            @test length(n_norms["velocity"].data_mean) == 3   # 3D feature
        end

        @testset "C4: o_norms[acceleration] is NormaliserOfflineMeanStd" begin
            @test haskey(o_norms, "acceleration")
            @test o_norms["acceleration"] isa NormaliserOfflineMeanStd
        end

        @testset "C5: acceleration is output-only; absent from n_norms" begin
            @test !haskey(n_norms, "acceleration")
        end

        @testset "C6: norm_type=:online forces NormaliserOnline" begin
            args_online = GraphNetSim.Args(;
                use_cuda=false, show_progress_bars=false, norm_steps=50,
                norm_type=:online,
                mps=2, layer_size=16, hidden_layers=1,
                training_strategy=DerivativeTraining(),
                solver_valid=OrdinaryDiffEq.Tsit5(), solver_valid_dt=0.002f0,
                types_updated=[1], types_noisy=[0], noise_stddevs=[0.0f0],
            )
            _, _, n_norms_ol, o_norms_ol = GraphNetSim.calc_norms(ds, cpu_device(), args_online)
            @test n_norms_ol["velocity"] isa NormaliserOnline
            @test o_norms_ol["acceleration"] isa NormaliserOnline
        end

        @testset "C7: norm_type=:meanstd returns NormaliserOfflineMeanStd" begin
            @test n_norms["velocity"] isa NormaliserOfflineMeanStd
            @test o_norms["acceleration"] isa NormaliserOfflineMeanStd
        end

        @testset "C8: norm_type=:minmax errors on missing stats" begin
            args_mm = GraphNetSim.Args(;
                use_cuda=false, show_progress_bars=false, norm_steps=50,
                norm_type=:minmax,
                mps=2, layer_size=16, hidden_layers=1,
                training_strategy=DerivativeTraining(),
                solver_valid=OrdinaryDiffEq.Tsit5(), solver_valid_dt=0.002f0,
                types_updated=[1], types_noisy=[0], noise_stddevs=[0.0f0],
            )
            @test_throws ArgumentError GraphNetSim.calc_norms(ds, cpu_device(), args_mm)
        end
    end

    # ─────────────────────────────────────────────────────────────────────────
    # Group D — Normalizer application on real trajectory data (CPU, ballistic_small)
    # ─────────────────────────────────────────────────────────────────────────
    @testset "D: Normalizer application on real data" begin
        println("Running: D — Normalizer application on real data")
        args = GraphNetSim.Args(;
            use_cuda=false,
            show_progress_bars=false,
            norm_steps=20,
            norm_type=:meanstd,
            mps=2, layer_size=16, hidden_layers=1,
            training_strategy=DerivativeTraining(),
            solver_valid=OrdinaryDiffEq.Tsit5(),
            solver_valid_dt=0.002f0,
            types_updated=[1],
            types_noisy=[0],
            noise_stddevs=[0.0f0],
        )
        ds = GraphNetSim.Dataset(:train, NORM_DATA_PATH, args)
        ds.meta["types_updated"] = args.types_updated
        ds.meta["types_noisy"]   = args.types_noisy
        ds.meta["noise_stddevs"] = args.noise_stddevs
        ds.meta["device"]        = cpu_device()
        ds.meta["training_strategy"] = nothing

        _, e_norms, n_norms, o_norms = GraphNetSim.calc_norms(ds, cpu_device(), args)

        traj = MLUtils.getobs(ds, 1)
        vel  = traj["velocity"][:, :, 5]     # (3, 10) at timestep 5

        @testset "D1: e_norm accumulates when called with default acc=true" begin
            n_before = e_norms.num_accumulations
            fake_edges = randn(Float32, 4, 20)  # dims+1=4 edge features, 20 edges
            e_norms(fake_edges)
            @test e_norms.num_accumulations == n_before + 1.0f0
        end

        @testset "D2: e_norm does NOT accumulate with acc=false" begin
            n_before = e_norms.num_accumulations
            fake_edges = randn(Float32, 4, 15)
            e_norms(fake_edges, false)
            @test e_norms.num_accumulations == n_before
        end

        @testset "D3: velocity normalizer changes value (not identity)" begin
            normed = n_norms["velocity"](vel)
            @test !all(isapprox.(normed, vel; atol=1.0f0))
            @test all(isfinite, normed)
        end

        @testset "D4: velocity normalizer output has reduced magnitude (mean-subtracted)" begin
            normed = n_norms["velocity"](vel)
            # After mean-std normalization, values should be closer to 0
            @test mean(abs.(normed)) < mean(abs.(vel))
        end
    end

    # ─────────────────────────────────────────────────────────────────────────
    # Group E — Output normalizer round-trips (CPU, ballistic_small)
    # ─────────────────────────────────────────────────────────────────────────
    @testset "E: Output normalizer (o_norms) round-trips" begin
        println("Running: E — Output normalizer round-trips")
        args = GraphNetSim.Args(;
            use_cuda=false,
            show_progress_bars=false,
            norm_steps=0,
            norm_type=:meanstd,
            mps=2, layer_size=16, hidden_layers=1,
            training_strategy=DerivativeTraining(),
            solver_valid=OrdinaryDiffEq.Tsit5(),
            solver_valid_dt=0.002f0,
            types_updated=[1],
            types_noisy=[0],
            noise_stddevs=[0.0f0],
        )
        ds = GraphNetSim.Dataset(:train, NORM_DATA_PATH, args)
        ds.meta["types_updated"] = args.types_updated
        ds.meta["types_noisy"]   = args.types_noisy
        ds.meta["noise_stddevs"] = args.noise_stddevs
        ds.meta["device"]        = cpu_device()
        ds.meta["training_strategy"] = DerivativeTraining()

        _, _, _, o_norms = GraphNetSim.calc_norms(ds, cpu_device(), args)
        traj   = MLUtils.getobs(ds, 1)
        target = traj["target|acceleration"][:, :, 5]   # (3, 10) at timestep 5

        @testset "E1: o_norm[acceleration] changes value (not identity)" begin
            normed = o_norms["acceleration"](target)
            @test !all(isapprox.(normed, target; atol=2.0f0))
            @test all(isfinite, normed)
        end

        @testset "E2: inverse_data recovers original target within atol=1f-5" begin
            normed    = o_norms["acceleration"](target)
            recovered = inverse_data(o_norms["acceleration"], normed)
            @test all(isapprox.(recovered, target; atol=1f-5))
        end

        @testset "E3: double-normalization is NOT identity" begin
            # Applying the normalizer twice does not recover the original
            normed_once  = o_norms["acceleration"](target)
            normed_twice = o_norms["acceleration"](normed_once)
            @test !all(isapprox.(normed_twice, target; atol=5.0f0))
        end
    end

    # ─────────────────────────────────────────────────────────────────────────
    # Group F — Online normalizer state persists through checkpoint
    # Runs on GPU when available, falls back to CPU otherwise.
    # ─────────────────────────────────────────────────────────────────────────
    @testset "F: Online normalizer checkpoint persistence" begin
        println("Running: F — Online normalizer checkpoint persistence")
        has_cuda = CUDA.functional()
        mktempdir() do cp_path
            train_network(
                Optimisers.Adam(1.0f-4),
                NORM_DATA_PATH,
                cp_path;
                use_cuda=has_cuda,
                show_progress_bars=false,
                mps=2,
                layer_size=16,
                hidden_layers=1,
                training_strategy=DerivativeTraining(),
                solver_valid=OrdinaryDiffEq.Tsit5(),
                solver_valid_dt=0.002f0,
                types_updated=[1],
                types_noisy=[0],
                noise_stddevs=[0.0f0],
                steps=20,
                norm_steps=10,
                checkpoint=20,
            )

            cp_files = sort(filter(
                f -> endswith(f, ".jld2"), readdir(cp_path; join=true)
            ))
            @assert !isempty(cp_files) "No checkpoint files found in $cp_path"

            @testset "F1: checkpoint contains e_norm, n_norm, o_norm keys" begin
                jldopen(last(cp_files), "r") do f
                    @test haskey(f, "e_norm")
                    @test haskey(f, "n_norm")
                    @test haskey(f, "o_norm")
                end
            end

            @testset "F2: e_norm state persisted in checkpoint" begin
                jldopen(last(cp_files), "r") do f
                    e = f["e_norm"]
                    # e_norm is created with default max_acc=1f7 (calc_norms does not
                    # limit e_norm to norm_steps — only n_norm/o_norm use norm_steps)
                    @test e["max_accumulations"] == 1.0f7
                    @test e["num_accumulations"] > 0.0f0
                end
            end

            @testset "F3: deserialize/serialize round-trip is lossless" begin
                e_dict = jldopen(last(cp_files), "r") do f
                    f["e_norm"]
                end
                n1 = GraphNetCore.deserialize(e_dict, cpu_device())
                n2 = GraphNetCore.deserialize(GraphNetCore.serialize(n1), cpu_device())

                x = ones(Float32, 4, 10)
                @test n1(x, false) == n2(x, false)
            end
        end
    end

end
