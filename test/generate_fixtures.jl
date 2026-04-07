#
# Generates test fixture datasets (HDF5 + meta.json) at test time.
# Generator modules live in generators.jl; this file just runs them
# for the test/fixtures/ directory.
#
# Called once at the top of runtests.jl before any tests run.
#

include(joinpath(@__DIR__, "generators.jl"))

const FIXTURE_DIR = joinpath(@__DIR__, "fixtures")

for (name, gen_fn) in [
    ("ballistic_small", _GenBallistic.generate), ("dam_break_small", _GenDamBreak.generate)
]
    ds_dir = joinpath(FIXTURE_DIR, name)
    if _needs_generation(ds_dir)
        @info "Generating test fixture: $name"
        gen_fn(ds_dir)
        @info "  Done."
    else
        @info "Test fixture already exists: $name"
    end
end
