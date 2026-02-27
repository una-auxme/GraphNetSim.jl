#
# Copyright (c) 2026 Josef Kircher, Julian Trommer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

## new
import Distributions: Normal
import Random: MersenneTwister

import HDF5: h5open, Group
import JLD2: jldopen
import JSON: parse
import Random: seed!, shuffle
using PointNeighbors
using Adapt

include("strategies.jl")

"""
    Dataset

A mutable structure containing trajectory data and associated metadata for training, validation, or testing.

## Fields
- `meta::Dict{String,Any}`: Dictionary containing all metadata for the dataset, including feature names, trajectory information, and device settings.
- `datafile::String`: Path to the data file (usually .h5 or .jld2 format).
- `lock::ReentrantLock`: Lock for thread-safe access to the datafile during concurrent operations.
"""
mutable struct Dataset
    meta::Dict{String,Any}
    datafile::String
    lock::ReentrantLock
end

"""
    Dataset(datafile::String, metafile::String, args)

Construct a Dataset from separate data and metadata files.

Validates that both files exist and have correct formats (.h5 or .jld2 for data, .json for metadata),
then loads trajectories and merges metadata with provided arguments.

## Arguments
- `datafile::String`: Path to the data file (.h5 or .jld2 containing trajectory data).
- `metafile::String`: Path to the metadata file (.json containing dataset configuration).
- `args`: A structure or object whose fields will be merged into the metadata dictionary.

## Returns
- `Dataset`: A new Dataset object initialized with the provided data and metadata.

## Throws
- `ArgumentError`: If datafile or metafile do not exist or have invalid formats.
"""
function Dataset(datafile::String, metafile::String, args)
    if !isfile(datafile)
        throw(ArgumentError("Invalid datafile: $datafile"))
    elseif !endswith(datafile, ".jld2") || !endswith(datafile, ".h5")
        throw(
            ArgumentError(
                "Invalid file format for datafile: $datafile. Possible formats are [.jld2, .h5]",
            ),
        )
    end
    if !isfile(metafile)
        throw(ArgumentError("Invalid metafile: $metafile"))
    elseif !endswith(metafile, ".json")
        throw(
            ArgumentError(
                "Invalid file format for metafile: $metafile. Possible formats are [.json]"
            ),
        )
    end

    meta = parse(Base.read(metafile), String)
    keys_traj = keystraj(datafile)
    meta["n_trajectories"] = length(keys_traj)
    meta["keys_trajectories"] = keys_traj
    merge!(meta, Dict(String(key) => getfield(args, key) for key in propertynames(args)))

    Dataset(meta, datafile, ReentrantLock())
end

"""
    Dataset(split::Symbol, path::String, args)

Construct a Dataset by specifying a split type and directory path.

Locates and loads the appropriate data file based on the split type (:train, :valid, or :test),
expecting a "meta.json" file in the given directory.

## Arguments
- `split::Symbol`: The dataset split, one of `:train`, `:valid`, or `:test`.
- `path::String`: Directory path containing the metadata file "meta.json" and the corresponding data file.
- `args`: A structure or object whose fields will be merged into the metadata dictionary.

## Returns
- `Dataset`: A new Dataset object initialized with data from the specified split.

## Throws
- `ArgumentError`: If split is invalid, if meta.json is not found, or if the corresponding data file cannot be found.
"""
function Dataset(split::Symbol, path::String, args)
    if split != :train && split != :valid && split != :test
        throw(
            ArgumentError(
                "Invalid symbol for dataset: $split. Possible values are [:train, :valid, :test]",
            ),
        )
    end
    if !isfile(joinpath(path, "meta.json"))
        throw(
            ArgumentError(
                "Metafile not found in path: $path. Check that your metafile is named \"meta.json\"",
            ),
        )
    end

    meta = parse(Base.read(joinpath(path, "meta.json"), String))
    datafile = get_file(split, path)
    keys_traj = keystraj(datafile)
    meta["n_trajectories"] = length(keys_traj)
    meta["keys_trajectories"] = keys_traj
    merge!(meta, Dict(String(key) => getfield(args, key) for key in propertynames(args)))

    Dataset(meta, datafile, ReentrantLock())
end

"""
    get_file(split::Symbol, path::String)

Locate the data file corresponding to a dataset split in the given directory.

Attempts to find a .jld2 file first, then a .h5 file with the split name (e.g., "train.jld2" or "train.h5").

## Arguments
- `split::Symbol`: The dataset split name (converted to string).
- `path::String`: Directory path to search for the data file.

## Returns
- `String`: Full path to the located data file.

## Throws
- `ArgumentError`: If no data file with the specified split name is found in the directory.
"""
function get_file(split::Symbol, path::String)
    filename = String(split)
    if isfile(joinpath(path, "$filename.jld2"))
        return joinpath(path, "$filename.jld2")
    elseif isfile(joinpath(path, "$filename.h5"))
        return joinpath(path, "$filename.h5")
    else
        throw(ArgumentError("No datafile for $filename was found at the given path: $path"))
    end
end

"""
    keystraj(datafile::String)

Extract trajectory keys from a data file.

Opens either a .jld2 or .h5 file and returns all top-level keys, representing individual trajectories stored in the file.

## Arguments
- `datafile::String`: Path to the data file (.h5 or .jld2).

## Returns
- `Array{String,1}`: Array of trajectory keys from the file.
"""
function keystraj(datafile::String)
    if endswith(datafile, ".jld2")
        file = jldopen(datafile, "r")
    elseif endswith(datafile, ".h5")
        file = h5open(datafile, "r")
    end
    keys_traj = keys(file)
    close(file)

    return keys_traj
end

MLUtils.numobs(ds::Dataset) = ds.meta["n_trajectories"]

"""
    MLUtils.getobs!(buffer::Dict{String,Any}, ds::Dataset, idx::Int)

Load trajectory data into a pre-allocated buffer using the MLUtils interface.

Retrieves a single trajectory by index, populates all metadata and features into the provided buffer dictionary,
and applies trajectory preparation (device transfer, masking, validation masks). Modifies the buffer in-place.

## Arguments
- `buffer::Dict{String,Any}`: Pre-allocated dictionary to store trajectory data (modified in-place).
- `ds::Dataset`: The dataset object.
- `idx::Int`: Index of the trajectory to retrieve (1-indexed).

## Returns
- `Dict{String,Any}`: The modified buffer containing the trajectory data.
"""
function MLUtils.getobs!(buffer, ds::Dataset, idx)
    key = ds.meta["keys_trajectories"][idx]

    set_meta!(buffer, ds, key)

    for fn in ds.meta["feature_names"]
        alloc_traj!(buffer, ds, fn)

        set_traj_data!(buffer, ds, fn, key)
    end

    prepare_trajectory!(buffer, ds.meta, ds.meta["device"])
    buffer["mask"] = ds.meta["device"](Int32.(
        findall(x -> x in ds.meta["types_updated"], buffer["node_type"][1, :, 1])
    ))

    val_mask = Float32.(
        map(x -> x in ds.meta["types_updated"], buffer["node_type"][:, :, 1])
    )

    buffer["val_mask"] = ds.meta["device"](repeat(
        val_mask, sum(size(buffer[field], 1) for field in ds.meta["output_features"]), 1
    ))

    # if !isnothing(ds.meta["training_strategy"]) && 
    #     (typeof(ds.meta["training_strategy"]) <: DerivativeStrategy)
    #     create_edges(buffer, ds.meta)
    # end

    return buffer
end

"""
    MLUtils.getobs(ds::Dataset, idx)

Retrieve a single trajectory observation from the dataset.

Implements the MLUtils interface for accessing individual dataset samples. Returns a new dictionary
containing the trajectory data and metadata, without modifying any buffer.

## Arguments
- `ds::Dataset`: The dataset object.
- `idx::Int`: Index of the trajectory to retrieve (1-indexed).

## Returns
- `Dict{String,Any}`: Dictionary containing the trajectory data with keys for each feature name, dt, n_particles, dims, mask, and val_mask.
"""
function MLUtils.getobs(ds::Dataset, idx)
    traj_dict = Dict{String,Any}()

    getobs!(traj_dict, ds, idx)

    return traj_dict
end

"""
    set_meta!(traj_dict::Dict{String,Any}, ds::Dataset, key::String)

Populate trajectory metadata fields in the data dictionary.

Extract and compute trajectory parameters such as dt (time step), trajectory_length, n_particles, and dims
from the dataset metadata and data files. Handles various metadata specifications including static values,
integers, and references to keys within the data files.

## Arguments
- `traj_dict::Dict{String,Any}`: Dictionary to populate with trajectory metadata (modified in-place).
- `ds::Dataset`: The dataset object.
- `key::String`: Trajectory key identifier in the dataset.

## Throws
- `ArgumentError`: If metadata specifications are inconsistent or invalid.
"""
function set_meta!(traj_dict::Dict{String,Any}, ds::Dataset, key::String)
    dt = ds.meta["dt"]
    tl = ds.meta["trajectory_length"]
    dims = ds.meta["dims"]

    if typeof(dt) <: AbstractFloat
        if tl == -1
            throw(
                ArgumentError(
                    "The metadata \"dt\" was specified as static and \"trajectory_length\" as -1 inside the metafile. You need to specify one of them as a vector with the length equal to the number of steps to infer the other one.",
                ),
            )
        elseif typeof(tl) <: Integer
            dt = range(0.0, dt * (tl - 1); step=dt)
        elseif (typeof(tl)) == String
            lock(ds.lock) do
                if endswith(ds.datafile, ".jld2")
                    file = jldopen(ds.datafile, "r")
                    traj = file[key]
                    tl = file[key][tl]
                else
                    file = h5open(ds.datafile, "r")
                    traj = open_group(file, key)
                    tl = Base.read(traj, tl)
                end
                close(file)
            end
            dt = range(0.0, dt * (tl - 1); step=dt)
        else
            throw(
                ArgumentError(
                    "The metadata \"trajectory_length\" is invalid. Possible values are: [-1 (for inferring the length), Integer (for specifying the length), String (as key inside the datafile)]",
                ),
            )
        end
    elseif typeof(dt) == String
        lock(ds.lock) do
            if endswith(ds.datafile, ".jld2")
                file = jldopen(ds.datafile, "r")
                traj = file[key]
                dt = file[key][dt]
            else
                file = h5open(ds.datafile, "r")
                traj = open_group(file, key)
                dt = Base.read(traj, dt)
            end
            close(file)
        end
        if (typeof(tl)) == String
            lock(ds.lock) do
                if endswith(ds.datafile, ".jld2")
                    file = jldopen(ds.datafile, "r")
                    traj = file[key]
                    tl = file[key][tl]
                else
                    file = h5open(ds.datafile, "r")
                    traj = open_group(file, key)
                    tl = Base.read(traj, tl)
                end
                close(file)
            end
        elseif !(typeof(tl) <: Integer)
            throw(
                ArgumentError(
                    "The metadata \"trajectory_length\" is invalid. Possible values are: [-1 (for inferring the length), Integer (for specifying the length), String (as key inside the datafile)]",
                ),
            )
        end
        if length(dt) == 1
            dt = range(0.0, dt * (tl - 1); step=dt)
        end
    else
        throw(
            ArgumentError(
                "The metadata \"dt\" is invalid. Possible values are: [Float (for specifying the static time delta), String (as key inside the datafile)]",
            ),
        )
    end

    if typeof(dims) == String
        lock(ds.lock) do
            if endswith(ds.datafile, ".jld2")
                file = jldopen(ds.datafile, "r")
                traj = file[key]
                dims = file[key][dims]
            else
                file = h5open(ds.datafile, "r")
                traj = open_group(file, key)
                dims = Base.read(traj, dims)
            end
            close(file)
        end
    end
    if typeof(dims) <: Integer
        if haskey(ds.meta, "n_particles")
            n_particles = ds.meta["n_particles"]
            if typeof(n_particles) == String
                lock(ds.lock) do
                    if endswith(ds.datafile, ".jld2")
                        file = jldopen(ds.datafile, "r")
                        traj = file[key]
                        n_particles = file[key][n_particles]
                    else
                        file = h5open(ds.datafile, "r")
                        traj = open_group(file, key)
                        n_particles = Base.read(traj, n_particles)
                    end
                    close(file)
                end
            elseif !(typeof(n_particles) <: Integer)
                throw(
                    ArgumentError(
                        "The metadata \"n_particles\" is invalid. Possible values are: [Integer (for specifying the number of nodes), String (as key inside the datafile)]",
                    ),
                )
            end
        else
            throw(
                ArgumentError(
                    "The metadata \"dims\" is specified as Integer but no metadata \"n_particles\" was provided. The number of nodes can only be inferred from a vector of dimensions. Either provide the number of nodes or use a vector of static dimensions.",
                ),
            )
        end
    elseif typeof(dims) <: AbstractArray && all(x -> typeof(x) <: Integer, dims)
        if any(x -> x == -1, dims)
            if haskey(ds.meta, "n_particles")
                n_particles = ds.meta["n_particles"]
                if typeof(n_particles) == String
                    lock(ds.lock) do
                        if endswith(ds.datafile, ".jld2")
                            file = jldopen(ds.datafile, "r")
                            traj = file[key]
                            n_particles = file[key][n_particles]
                        else
                            file = h5open(ds.datafile, "r")
                            traj = open_group(file, key)
                            n_particles = Base.read(traj, n_particles)
                        end
                        close(file)
                    end
                elseif !(typeof(n_particles) <: Integer)
                    throw(
                        ArgumentError(
                            "The metadata \"n_particles\" is invalid. Possible values are: [Integer (for specifying the number of nodes), String (as key inside the datafile)]",
                        ),
                    )
                end
            else
                throw(
                    ArgumentError(
                        "The metadata \"dims\" contains -1 (for inferring dimensions) but no metadata \"n_particles\" was provided. The number of nodes can only be inferred from a vector of dimensions with positive values. Either provide the number of nodes or use a vector of static positive dimensions.",
                    ),
                )
            end
            if haskey(ds.meta, "dims_key")
                lock(ds.lock) do
                    if endswith(ds.datafile, ".jld2")
                        file = jldopen(ds.datafile, "r")
                        traj = file[key]
                        dims_file = file[key]["dims_key"]
                    else
                        file = h5open(ds.datafile, "r")
                        traj = open_group(file, key)
                        dims_file = Base.read(traj, "dims_key")
                    end
                    close(file)
                end
                if length(dims_file) != length(dims)
                    throw(
                        ArgumentError(
                            "The size of the metadata \"dims\" vector is not equal the size of the dims inside the datafile: size(dims_meta) = $dims, size(dims_file) = $dims_file",
                        ),
                    )
                else
                    dims = dims_file
                end
            else
                throw(
                    ArgumentError(
                        "The metadata \"dims\" contains -1 (for inferring dimensions) but no metadata \"dims_key\" for reading the dimensions from the datafile was provided.",
                    ),
                )
            end
        else
            n_particles = prod(dims)
        end
    else
        throw(
            ArgumentError(
                "The metadata \"dims\" is invalid. Possible values are: [Integer (for specifying the dimensions), Vector{Integer} (for specifying nodes in each dimension)]",
            ),
        )
    end

    traj_dict["dt"] = Float32.(dt)
    traj_dict["trajectory_length"] = tl
    traj_dict["n_particles"] = n_particles
    traj_dict["dims"] = dims
end

"""
    alloc_traj!(traj_dict::Dict{String,Any}, ds::Dataset, fn::String)

Allocate a zero-initialized array in the data dictionary for a feature.

Creates and inserts a zero array with appropriate dimensions based on the feature type (static/dynamic)
and metadata. Does nothing if the feature is already present in the dictionary.

## Arguments
- `traj_dict::Dict{String,Any}`: Dictionary to store the allocated array (modified in-place).
- `ds::Dataset`: The dataset object containing feature metadata.
- `fn::String`: Feature name to allocate.

## Throws
- `ArgumentError`: If feature type is neither 'static' nor 'dynamic'.
"""
function alloc_traj!(traj_dict::Dict{String,Any}, ds::Dataset, fn::String)
    dim = haskey(ds.meta["features"][fn], "dim") ? ds.meta["features"][fn]["dim"] : 1
    if ds.meta["features"][fn]["type"] == "static"
        tl = 1
    elseif ds.meta["features"][fn]["type"] == "dynamic"
        tl = traj_dict["trajectory_length"]
    else
        throw(ArgumentError("feature type of feature \"$fn\" must be static or dynamic"))
    end
    if !haskey(traj_dict, fn)
        traj_dict[fn] = zeros(
            getfield(Base, Symbol(uppercasefirst(ds.meta["features"][fn]["dtype"]))),
            dim,
            traj_dict["n_particles"],
            tl,
        )
    end
end

"""
    set_traj_data!(traj_dict::Dict{String,Any}, ds::Dataset, fn::String, key)

Load feature data from file into the pre-allocated trajectory array.

Reads feature data from either .jld2 or .h5 file format. For dynamic features, loads data from all time steps.
For static features, loads a single time-independent array. Accesses files in a thread-safe manner using locks.

## Arguments
- `traj_dict::Dict{String,Any}`: Dictionary containing pre-allocated arrays (modified in-place).
- `ds::Dataset`: The dataset object.
- `fn::String`: Feature name to load.
- `key::String`: Trajectory key identifier.

"""
function set_traj_data!(traj_dict::Dict{String,Any}, ds::Dataset, fn::String, key)
    lock(ds.lock) do
        if endswith(ds.datafile, ".jld2")
            file = jldopen(ds.datafile, "r")
            traj = file[key]
            if ds.meta["features"][fn]["type"] == "dynamic"
                for t in 1:ds.meta["trajectory_length"]
                    dataset = ds.meta["features"][fn]["key"]
                    sp = split(dataset, "\$t")
                    dataset = sp[1] * "$t" * sp[2]

                    data = traj[dataset]

                    #concatenate into time series               
                    traj_dict[fn][:, :, t] = data
                end
            elseif ds.meta["features"][fn]["type"] == "static"
                dataset = ds.meta["features"][fn]["key"]
                data = traj[dataset]
                traj_dict[fn][1, :, :] .= data
            end
            dt = Float32.(file[key][ds.meta["dt"]])
        else
            file = h5open(ds.datafile, "r")
            traj = open_group(file, key)
            if ds.meta["features"][fn]["type"] == "dynamic"
                for t in 1:traj_dict["trajectory_length"]
                    dataset = ds.meta["features"][fn]["key"]
                    sp = split(dataset, "\$t")
                    dataset = sp[1] * "$t" * sp[2]

                    data = read_dataset(traj, dataset)
                    #concatenate into time series               
                    traj_dict[fn][:, :, t] = data
                end
            elseif ds.meta["features"][fn]["type"] == "static"
                dataset = ds.meta["features"][fn]["key"]
                data = read(traj, dataset)
                traj_dict[fn][1, :, :] .= data
            end
            dt = Float32.(Base.read(traj, ds.meta["dt"]))
        end
        close(file)
        traj_dict["dt"] = dt
    end
end

"""
    create_edges(traj_dict::Dict{String,Any}, meta::Dict{String,Any})

Construct graph edges based on spatial proximity and store edge data.

Computes sender-receiver pairs for all nodes within the connectivity radius at each time step.
Calculates relative displacement and distance for each edge. Supports both CPU and GPU computation.
Stores edge information (senders, receivers, features, bounds) in the trajectory dictionary.

## Arguments
- `traj_dict::Dict{String,Any}`: Dictionary containing trajectory data with 'position' and 'mask' fields (modified in-place to add edge information).
- `meta::Dict{String,Any}`: Metadata dictionary containing 'default_connectivity_radius', 'device', etc.

## Stores in traj_dict
- `senders`: List of sender node indices for each time step.
- `senders_old`: List of old sender indices (for potential filtering).
- `receivers`: List of receiver node indices for each time step.
- `edge_features`: Relative displacement and distance normalized by connectivity radius.
- `dist_bounds`: Distance bounds for each edge.
"""
function create_edges(traj_dict::Dict{String,Any}, meta::Dict{String,Any})
    local senders = []
    local senders_old = []
    local receivers = []
    local edge_features = []
    local dist_bounds = []
    sender = nothing
    receiver = nothing
    sender_old = []
    dist_bound = []
    position = traj_dict["position"]

    for i in axes(position)[3]
        fluid_position = position[:, traj_dict["mask"], i]
        current_position = position[:, :, i]
        if device == cpu_device()
            cur_pos_cpu = cpu_device()(current_position)
            tree = KDTree(cur_pos_cpu; reorder=false)
            receivers_list = inrange(
                tree, cur_pos_cpu, Float32(meta["default_connectivity_radius"]), false
            )
            sender = vcat(
                [repeat([i], length(j)) for (i, j) in enumerate(receivers_list)]...
            )
            receiver = vcat(receivers_list...)
            rel_displacement =
                (current_position[:, receiver] - current_position[:, sender]) ./
                Float32(meta["default_connectivity_radius"])
            rel_dist_norm = sqrt.(sum(abs2, rel_displacement; dims=1))
        else
            sender, receiver, rel_displacement, rel_dist_norm = point_neighbor_ns(
                current_position,
                Float32(meta["default_connectivity_radius"]),
                traj_dict["mask"],
            )
        end
        # if meta["features"]["node_type"]["data_max"] - meta["features"]["node_type"]["data_min"] != 0
        #     sender_old, receiver_old, sender, receiver, _, b_particle = check_and_delete_filtered(sender, receiver, size(fluid_position, 2), true)
        #     rel_displacement = (current_position[:, receiver_old] - current_position[:, sender_old]) ./ Float32(meta["default_connectivity_radius"])
        #     rel_dist_norm = sqrt.(sum(abs2, rel_displacement; dims = 1))
        #     dist_bound = compute_clostest_dist_bound(Array(sender_old), Array(receiver_old), Array(rel_dist_norm), Array(rel_displacement), size(fluid_position,2), length(unique(Array(b_particle))))
        # else
        #     dist_bound = ones(size(fluid_position))
        # end
        dist_bound = ones(size(fluid_position))
        push!(
            edge_features,
            Array(meta["device"](vcat(rel_displacement, rel_dist_norm) .+ 1.0f-8)),
        )
        push!(senders, Array(sender))
        push!(senders_old, Array(sender_old))
        push!(receivers, Array(receiver))
        push!(dist_bounds, dist_bound)
    end
    traj_dict["senders"] = meta["device"](senders)
    traj_dict["senders_old"] = senders_old
    traj_dict["receivers"] = meta["device"](receivers)
    traj_dict["edge_features"] = meta["device"](edge_features)
    traj_dict["dist_bounds"] = meta["device"](dist_bounds)
end

"""
    add_targets!(data::Dict{String,Any}, fields::Vector{Any}, device::Function)

Create target fields for supervised learning by duplicating specified fields.

For derivative  and other strategies requiring ground truth targets, this function creates "target|" prefixed
versions of specified fields. Skips internal metadata fields and only processes multi-dimensional temporal data.
Devices data appropriately except for node_type which remains on the original device.

## Arguments
- `data::Dict{String,Any}`: Dictionary containing trajectory data (modified in-place).
- `fields::Vector{Any}`: Field names for which to create target copies.
- `device::Function`: Device placement function (e.g., cpu_device, gpu_device).

"""
function add_targets!(data::Dict{String,Any}, fields::Vector{Any}, device::Function)
    new_data = deepcopy(data)
    for (key, value) in data
        if startswith(key, "target|") || key == "dt" || key == "n_particles"
            continue
        end
        if ndims(value) > 2 && size(value)[end] > 1
            if key == "node_type"
                new_data[key] = value
            else
                new_data[key] = device(value)
            end
            if key in fields
                new_data["target|" * key] = device(value)
            end
        end
    end
    for (key, value) in new_data
        data[key] = value
    end
end

"""
    preprocess!(data::Dict{String,Any}, noise_fields::Vector{Any}, noise_stddevs::Vector{Float32}, types_noisy::Vector, ts::Union{DerivativeStrategy,Nothing}, device::Function)

Apply preprocessing transformations including noise injection and optional shuffling.

Adds Gaussian noise to specified fields only for not-noisy node types. If using a derivative-based strategy with
random shuffling enabled, shuffles time indices within the window size. Uses a fixed random seed to be consisted during the shuffling.

## Arguments
- `data::Dict{String,Any}`: Dictionary containing trajectory data (modified in-place).
- `noise_fields::Vector{Any}`: Names of fields to which noise is added.
- `noise_stddevs::Union{Float32,Vector{Float32}}`: Standard deviation(s) of Gaussian noise. If length 1, broadcasted to all fields.
- `types_noisy::Vector`: Node types that should receive noise.
- `ts::Union{DerivativeStrategy,Nothing}`: Training strategy object; if DerivativeStrategy with random=true, enables shuffling.
- `device::Function`: Device placement function.

## Throws
- `DimensionMismatch`: If noise_stddevs length is neither 1 nor equal to noise_fields length.
"""
function preprocess!(
    data::Dict{String,Any},
    noise_fields::Vector{Any},
    noise_stddevs::Vector{Float32},
    types_noisy::Vector,
    ts::Union{DerivativeStrategy,Nothing},
    device::Function,
)
    if length(noise_stddevs) != 1 && length(noise_stddevs) != length(noise_fields)
        throw(
            DimensionMismatch(
                "dimension of noise must be 1 or match noise fields: noise has dim $(size(noise_stddevs)), noise fields has dim $(size(noise_fields))",
            ),
        )
    end
    for (i, nf) in enumerate(noise_fields)
        d = Normal(0.0f0, length(noise_stddevs) > 1 ? noise_stddevs[i] : noise_stddevs[1])
        noise = device(rand(d, size(data[nf])))

        mask = findall(x -> x ∉ types_noisy, data["node_type"][1, :, 1])
        noise[:, mask, :] .= 0
        data[nf] += noise
    end

    rng = MersenneTwister(1234)

    for key in keys(data)
        if length(data[key]) == 1 || size(data[key])[end] == 1
            continue
        end
        if typeof(ts) <: DerivativeStrategy && ts.random
            if key != "dt" && key != "n_particles"
                data[key] = data[key][
                    repeat([:], ndims(data[key]) - 1)...,
                    shuffle(
                        rng,
                        ts.window_size == 0 ? collect(1:end) : collect(1:(ts.window_size)),
                    ),
                ]
            end
        end
        seed!(rng, 1234)
    end
end

"""
    prepare_trajectory!(data::Dict{String,Any}, meta::Dict{String,Any}, device::Function)

Prepare trajectory data for model training by transferring to device and applying strategy-specific preprocessing.

For DerivativeStrategy training, calls add_targets! and preprocess! to prepare ground truth targets and apply noise/shuffling.
For other strategies, simply transfers feature data to the specified device.
Always skips node_type field during device transfer as it remains on original device.

## Arguments
- `data::Dict{String,Any}`: Dictionary containing trajectory data (modified in-place).
- `meta::Dict{String,Any}`: Metadata dictionary containing device, training_strategy, and feature configuration.
- `device::Function`: Device placement function (e.g., cpu_device, gpu_device).

## Returns
- `Tuple{Dict{String,Any},Dict{String,Any}}`: Tuple of (prepared_data, metadata).
"""
function prepare_trajectory!(
    data::Dict{String,Any}, meta::Dict{String,Any}, device::Function
)
    if !isnothing(meta["training_strategy"]) &&
        (typeof(meta["training_strategy"]) <: DerivativeStrategy)
        add_targets!(data, meta["derivative_target_features"], device)
        preprocess!(
            data,
            meta["input_features"],
            meta["noise_stddevs"],
            meta["types_noisy"],
            meta["training_strategy"],
            device,
        )
        for field in meta["feature_names"]
            if field == "node_type" || field in meta["derivative_target_features"]
                continue
            end
            data[field] = device(data[field])
        end
    else
        for field in meta["feature_names"]
            if field == "node_type"
                continue
            end
            data[field] = device(data[field])
        end
    end
    return data, meta
end
