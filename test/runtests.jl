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
        Aqua.test_all(GraphNetSim; ambiguities=false)
    end

    include("test_converters.jl")
    include("test_schedulers.jl")
    include("test_normalizer.jl")
    include("test_datasets.jl")
    include("test_wall_distance.jl")
    include("test_trajectory_mask.jl")
    include("test_visualize.jl")
end
