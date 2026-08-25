#
# Copyright (c) 2026 Josef Jouaux
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

import JSON: print as json_print, parsefile as json_parsefile

const MODEL_CONFIG_FILENAME = "model_config.json"

"""
    ModelConfig

Persists the minimal set of parameters required to reconstruct the GNN model
from a checkpoint without re-specifying them at the call site.

The architecture fields (`mps`, `layer_size`, `hidden_layers`, `bounded`) affect
weight shapes and must match on resume; `bounded` additionally must agree with the
dataset's `meta["bounds"]` presence at load time (a bounded model has a wider encoder
input).  The remaining fields are derived from `meta.json` or the JLD2 checkpoint and
are saved as documentation; they may legitimately differ between training phases.

## Fields
- `mps`: Number of message passing steps.
- `layer_size`: Hidden layer width.
- `hidden_layers`: Number of hidden layers per MLP block.
- `norm_steps`: Steps before weight updates start (online normalizer warm-up).
- `types_updated`: Node types whose outputs are predicted.
- `types_noisy`: Node types receiving noise injection during training.
- `noise_stddevs`: Per-type noise standard deviations.
- `norm_type`: Normalization strategy for Float32 features (`:online`, `:minmax`, `:meanstd`).
- `bounded`: Whether the model was trained with the boundary (wall-distance) node feature,
  i.e. whether the training dataset defined `meta["bounds"]`. Part of the architecture: a
  bounded model has a wider encoder input than an unbounded one, so this must match the
  dataset at load time.
"""
@kwdef struct ModelConfig
    format_version::Int = 2
    mps::Int
    layer_size::Int
    hidden_layers::Int
    norm_steps::Int
    types_updated::Vector{Int}
    types_noisy::Vector{Int}
    noise_stddevs::Vector{Float32}
    norm_type::Symbol = :online
    bounded::Bool = false
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
                existing.hidden_layers != cfg.hidden_layers
                error(
                    "Architecture mismatch between supplied arguments and saved " *
                    "model config at \"$path\".\n" *
                    "  Saved:    mps=$(existing.mps), layer_size=$(existing.layer_size), " *
                    "hidden_layers=$(existing.hidden_layers)\n" *
                    "  Supplied: mps=$(cfg.mps), layer_size=$(cfg.layer_size), " *
                    "hidden_layers=$(cfg.hidden_layers)\n" *
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
                    "bounded" => cfg.bounded,
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
            bounded=Bool(get(arch, "bounded", false)),
            norm_steps=train["norm_steps"],
            types_updated=Int.(train["types_updated"]),
            types_noisy=Int.(train["types_noisy"]),
            noise_stddevs=Float32.(train["noise_stddevs"]),
            norm_type=Symbol(get(train, "norm_type", "online")),
        )
    catch e
        @warn "Could not parse model config at \"$path\": $e. Falling back to supplied arguments."
        return nothing
    end
end
