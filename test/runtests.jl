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
        # JuliaFormatter is a project dep for the `format(".")` CI/dev workflow
        # (see CLAUDE.md) but is intentionally never `import`ed in `src/`, so
        # Aqua's stale-dependency check would otherwise flag it.
        Aqua.test_all(
            GraphNetSim; ambiguities=false, stale_deps=(ignore=[:JuliaFormatter],)
        )
    end

    include("test_normalizer.jl")
    include("test_datasets.jl")
end
