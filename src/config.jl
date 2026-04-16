#
# Copyright (c) 2026 Josef Kircher, Julian Trommer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

import JSON: print as json_print, parsefile as json_parsefile

const MODEL_CONFIG_FILENAME = "model_config.json"

"""
    ModelConfig

Persists the minimal set of parameters required to reconstruct the GNN model
from a checkpoint without re-specifying them at the call site.

Only the three architecture fields (`mps`, `layer_size`, `hidden_layers`) are
strictly required for model reconstruction.  All other fields are derived from
`meta.json` or the JLD2 checkpoint at load time.  The training fields are saved
as documentation and may legitimately differ between training phases.

## Fields
- `mps`: Number of message passing steps.
- `layer_size`: Hidden layer width.
- `hidden_layers`: Number of hidden layers per MLP block.
- `norm_steps`: Steps before weight updates start (online normalizer warm-up).
- `types_updated`: Node types whose outputs are predicted.
- `types_noisy`: Node types receiving noise injection during training.
- `noise_stddevs`: Per-type noise standard deviations.
- `norm_type`: Normalization strategy for Float32 features (`:online`, `:minmax`, `:meanstd`).
- `boundary_distance`: Method for the distance-to-boundary feature
  (`:closest_particle` — displacement to closest boundary particle, feature dim = `dims`; or
  `:bounding_box` — axis-aligned bounding-box distance, feature dim = `2*dims`).
  Architectural: input feature dim depends on this, so it is validated on resumption.
"""
@kwdef struct ModelConfig
    format_version::Int = 1
    mps::Int
    layer_size::Int
    hidden_layers::Int
    norm_steps::Int
    types_updated::Vector{Int}
    types_noisy::Vector{Int}
    noise_stddevs::Vector{Float32}
    norm_type::Symbol = :online
    boundary_distance::Symbol = :closest_particle
end

"""
    save_model_config(cfg, cp_path)

Write `cfg` to `cp_path/model_config.json`.

On the first call the file is created.  On subsequent calls the architecture
fields (`mps`, `layer_size`, `hidden_layers`) are validated against the saved
values — a mismatch with an existing checkpoint would cause a weight-shape
incompatibility and is therefore an error.  Training fields are updated
silently, as they may change between training phases.
"""
function save_model_config(cfg::ModelConfig, cp_path::String)
    path = joinpath(cp_path, MODEL_CONFIG_FILENAME)

    if isfile(path)
        existing = load_model_config(cp_path)
        if !isnothing(existing)
            if existing.mps != cfg.mps ||
                existing.layer_size != cfg.layer_size ||
                existing.hidden_layers != cfg.hidden_layers ||
                existing.boundary_distance != cfg.boundary_distance
                error(
                    "Architecture mismatch between supplied arguments and saved " *
                    "model config at \"$path\".\n" *
                    "  Saved:    mps=$(existing.mps), layer_size=$(existing.layer_size), " *
                    "hidden_layers=$(existing.hidden_layers), " *
                    "boundary_distance=:$(existing.boundary_distance)\n" *
                    "  Supplied: mps=$(cfg.mps), layer_size=$(cfg.layer_size), " *
                    "hidden_layers=$(cfg.hidden_layers), " *
                    "boundary_distance=:$(cfg.boundary_distance)\n" *
                    "These parameters must match the existing checkpoint. " *
                    "Use a different cp_path to start a new training run.",
                )
            end
        end
    else
        mkpath(cp_path)
    end

    open(path, "w") do f
        json_print(
            f,
            Dict(
                "format_version" => cfg.format_version,
                "architecture" => Dict(
                    "mps" => cfg.mps,
                    "layer_size" => cfg.layer_size,
                    "hidden_layers" => cfg.hidden_layers,
                    "boundary_distance" => String(cfg.boundary_distance),
                ),
                "training" => Dict(
                    "norm_steps" => cfg.norm_steps,
                    "types_updated" => cfg.types_updated,
                    "types_noisy" => cfg.types_noisy,
                    "noise_stddevs" => cfg.noise_stddevs,
                    "norm_type" => String(cfg.norm_type),
                ),
            ),
            2,
        )
    end
end

"""
    load_model_config(cp_path) -> Union{ModelConfig, Nothing}

Read `cp_path/model_config.json` and return a `ModelConfig`, or `nothing` if
the file does not exist.  Returns `nothing` (with a warning) if the file exists
but cannot be parsed, preserving backwards compatibility with checkpoints that
predate this feature.
"""
function load_model_config(cp_path::String)::Union{ModelConfig,Nothing}
    path = joinpath(cp_path, MODEL_CONFIG_FILENAME)
    isfile(path) || return nothing

    try
        d = json_parsefile(path)
        arch = d["architecture"]
        train = d["training"]
        return ModelConfig(;
            format_version=d["format_version"],
            mps=arch["mps"],
            layer_size=arch["layer_size"],
            hidden_layers=arch["hidden_layers"],
            norm_steps=train["norm_steps"],
            types_updated=Int.(train["types_updated"]),
            types_noisy=Int.(train["types_noisy"]),
            noise_stddevs=Float32.(train["noise_stddevs"]),
            norm_type=Symbol(get(train, "norm_type", "online")),
            boundary_distance=Symbol(get(arch, "boundary_distance", "closest_particle")),
        )
    catch e
        @warn "Could not parse model config at \"$path\": $e. Falling back to supplied arguments."
        return nothing
    end
end
