using GraphNetSim
using Test
using Aqua

@testset "GraphNetSim.jl" begin
    # TODO
    @testset "Aqua.jl" begin
        # Ambiguities in external packages
        @testset "Method ambiguity" begin
            Aqua.test_ambiguities([GraphNetSim])
        end
        Aqua.test_all(GraphNetSim; ambiguities = false)
    end
end
