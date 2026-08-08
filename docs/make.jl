#
# Copyright (c) 2026 Josef Jouaux, Julian Trommer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using Pkg: Pkg
Pkg.develop(; path=joinpath(@__DIR__, "../../GraphNetSim.jl"))
using Documenter, GraphNetSim

# The docs home page is generated from the top-level README (the CI workflow does the same),
# so local builds `julia --project=docs/ docs/make.jl` work without a checked-in index.md.
cp(joinpath(@__DIR__, "..", "README.md"), joinpath(@__DIR__, "src", "index.md"); force=true)

makedocs(;
    sitename="GraphNetSim.jl",
    format=Documenter.HTML(; sidebar_sitename=false, edit_link=nothing),
    authors="Josef Jouaux, Julian Trommer, and contributors.",
    modules=[GraphNetSim],
    checkdocs=:exports,
    linkcheck=false,
    pages=[
        "Home" => "index.md",
        "Loading Data" => "loading_data.md",
        "API Reference" => "api.md",
    ],
)

deploydocs(; repo="github.com/una-auxme/GraphNetSim.jl.git", devbranch="main")
