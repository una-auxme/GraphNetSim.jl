using GraphNetSim
using Test
using Aqua

# Generate test fixture datasets (HDF5 + meta.json) if they don't exist yet
include("generate_fixtures.jl")

@testset "GraphNetSim.jl" begin
    @testset "Aqua.jl" begin
        println("Running: Aqua.jl")
        # Ambiguities in external packages
        @testset "Method ambiguity" begin
            Aqua.test_ambiguities([GraphNetSim])
        end
        # Deps that are intentionally not `import`ed from src/, so exclude them from the
        # stale-deps check:
        #  - GPUCompiler: deps-only version pin (see Project.toml [compat]), loaded
        #    transitively via CUDA/Reactant.
        #  - JuliaFormatter: dev/CI tool invoked as `using JuliaFormatter; format(".")`.
        Aqua.test_all(
            GraphNetSim;
            ambiguities=false,
            stale_deps=(ignore=[:GPUCompiler, :JuliaFormatter],),
        )
    end

    include("test_converters.jl")
    include("test_normalizer.jl")
    include("test_datasets.jl")
end
