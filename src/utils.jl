#
# Copyright (c) 2026 Josef Kircher, Julian Trommer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

import Printf: @sprintf
import JSON: print as json_print, parsefile as json_parsefile

"""
    n_node_types(meta)

Returns the number of distinct node types in the dataset.

## Arguments
- `meta`: Feature metadata dictionary containing `node_type` feature spec.

## Returns
- `Int`: `data_max - data_min + 1` for the `node_type` feature.
"""
function n_node_types(meta)
    meta["features"]["node_type"]["data_max"] - meta["features"]["node_type"]["data_min"] +
    1
end

"""
    n_node_types(meta)

Returns the number of distinct node types in the dataset.

## Arguments
- `meta`: Feature metadata dictionary containing `node_type` feature spec.

## Returns
- `Int`: `data_max - data_min + 1` for the `node_type` feature.
"""
function n_node_types(meta)
    meta["features"]["node_type"]["data_max"] - meta["features"]["node_type"]["data_min"] +
    1
end

"""
    isnumber(meta, f)

Checks whether the given feature is a numeric type (Int32 or Float32).

## Arguments
- `meta`: Feature metadata dictionary.
- `f`: Feature name to check.

## Returns
- `Bool`: True if feature is Int32 or Float32, false otherwise.
"""
function isnumber(meta, f)
    return getfield(Base, Symbol(uppercasefirst(meta["features"][f]["dtype"]))) == Int32 ||
           getfield(Base, Symbol(uppercasefirst(meta["features"][f]["dtype"]))) == Float32
end

"""
    der_minmax(path)

Calculates the minimum and maximum across training, validation, and test sets for each numeric feature.

Combines results from both training/validation and test data to compute overall min/max bounds.

## Arguments
- `path`: Path to the dataset files.

## Returns
- `Dict`: Dictionary mapping feature names to [min, max] value pairs across all datasets.
"""
function der_minmax(path)
    result = der_minmax(path, true)
    result_test = der_minmax(path, false)

    for (k, v) in result_test
        if v[1] < result[k][1]
            result[k][1] = v[1]
        end
        if v[2] > result[k][2]
            result[k][2] = v[2]
        end
    end
    return result
end

"""
    data_minmax(path)

Calculates the minimum and maximum values for each numeric feature across all dataset partitions.

Iterates through training, validation, and test datasets to compute global min/max bounds
for all numeric features (Int32 and Float32 types).

## Arguments
- `path`: Path to the dataset files.

## Returns
- `Dict`: Dictionary mapping feature names to [min, max] value pairs computed from all datasets.
"""
function data_minmax(path)
    args = Args()
    ds_train = Dataset(:train, path, args)
    ds_train.meta["types_updated"] = args.types_updated
    ds_train.meta["types_noisy"] = args.types_noisy
    ds_train.meta["noise_stddevs"] = args.noise_stddevs
    ds_train.meta["device"] = cpu_device()
    ds_train.meta["training_strategy"] = nothing
    train_loader = DataLoader(
        ds_train; batchsize=-1, buffer=false, parallel=true, shuffle=true
    )

    ds_valid = Dataset(:valid, path, args)
    ds_valid.meta["types_updated"] = args.types_updated
    ds_valid.meta["types_noisy"] = args.types_noisy
    ds_valid.meta["noise_stddevs"] = args.noise_stddevs
    ds_valid.meta["device"] = cpu_device()
    ds_valid.meta["training_strategy"] = nothing
    valid_loader = DataLoader(ds_valid; batchsize=-1, buffer=false, parallel=true)

    ds_test = Dataset(:test, path, args)
    ds_test.meta["device"] = cpu_device()
    ds_test.meta["training_strategy"] = nothing
    test_loader = DataLoader(ds_test; batchsize=-1, buffer=false, parallel=true)

    features = ds_train.meta["feature_names"]
    target_features = ds_train.meta["output_features"]

    result = Dict{String,Vector{Float32}}()
    for f in features
        if !haskey(ds_train.meta["features"][f], "onehot") && isnumber(ds_train.meta, f)
            result[f] = [Inf32, -Inf32]
        end
    end
    for tf in target_features
        if !haskey(ds_train.meta["features"][tf], "onehot") && isnumber(ds_train.meta, tf)
            result["target|$tf"] = [Inf32, -Inf32]
        end
    end

    function add_to_result(data)
        for f in features
            if !haskey(ds_train.meta["features"][f], "onehot") && isnumber(ds_train.meta, f)
                data_min = minimum(data[f])
                data_max = maximum(data[f])
                if data_min < result[f][1]
                    result[f][1] = data_min
                end
                if data_max > result[f][2]
                    result[f][2] = data_max
                end
            end
        end
        for tf in target_features
            if !haskey(ds_train.meta["features"][tf], "onehot") &&
                isnumber(ds_train.meta, tf)
                ddiff_min = minimum(data[tf])
                ddiff_max = maximum(data[tf])
                if ddiff_min < result["target|$tf"][1]
                    result["target|$tf"][1] = ddiff_min
                end
                if ddiff_max > result["target|$tf"][2]
                    result["target|$tf"][2] = ddiff_max
                end
            end
        end
    end
    p = Progress(
        length(train_loader) + length(valid_loader) + length(test_loader);
        desc="Calculating minmax-norm: ",
        dt=1.0,
        barlen=50,
    )
    for data in train_loader
        add_to_result(data)
        next!(p)
    end

    for data in valid_loader
        add_to_result(data)
        next!(p)
    end

    for data in test_loader
        add_to_result(data)
        next!(p)
    end
    finish!(p)

    return result
end

"""
    data_meanstd(path)

Calculates the mean and standard deviation for each feature in the given part of the dataset.

## Arguments
- `path`: Path to the dataset files.

## Returns
- Mean and standard deviation in training, validation and test set
"""
function data_meanstd(path)
    args = Args()
    ds_train = Dataset(:train, path, args)
    ds_train.meta["types_updated"] = args.types_updated
    ds_train.meta["types_noisy"] = args.types_noisy
    ds_train.meta["noise_stddevs"] = args.noise_stddevs
    ds_train.meta["device"] = cpu_device()
    ds_train.meta["training_strategy"] = nothing
    train_loader = DataLoader(
        ds_train; batchsize=-1, buffer=false, parallel=true, shuffle=true
    )

    ds_valid = Dataset(:valid, path, args)
    ds_valid.meta["types_updated"] = args.types_updated
    ds_valid.meta["types_noisy"] = args.types_noisy
    ds_valid.meta["noise_stddevs"] = args.noise_stddevs
    ds_valid.meta["device"] = cpu_device()
    ds_valid.meta["training_strategy"] = nothing
    valid_loader = DataLoader(ds_valid; batchsize=-1, buffer=false, parallel=true)

    ds_test = Dataset(:test, path, args)
    ds_test.meta["device"] = cpu_device()
    ds_test.meta["training_strategy"] = nothing
    test_loader = DataLoader(ds_test; batchsize=-1, buffer=false, parallel=true)

    features = ds_train.meta["feature_names"]
    target_features = ds_train.meta["output_features"]

    result = Dict{String,Any}()
    for f in features
        if !haskey(ds_train.meta["features"][f], "onehot") && isnumber(ds_train.meta, f)
            result["$f-acc_count"] = 0.0f0
            result["$f-acc_sum"] = zeros(Float32, ds_train.meta["features"][f]["dim"])
            result["$f-acc_sum_squared"] = zeros(
                Float32, ds_train.meta["features"][f]["dim"]
            )
        end
    end
    for tf in target_features
        if isnumber(ds_train.meta, tf)
            result["target|$tf-acc_count"] = 0.0f0
            result["target|$tf-acc_sum"] = zeros(
                Float32, ds_train.meta["features"][tf]["dim"]
            )
            result["target|$tf-acc_sum_squared"] = zeros(
                Float32, ds_train.meta["features"][tf]["dim"]
            )
        end
    end
    # result["edges-acc_count"] = 0.0f0
    # result["edges-acc_sum"] = zeros(Float32, ds_train.meta["dims"] + 1)
    # result["edges-acc_sum_squared"] = zeros(Float32, ds_train.meta["dims"] + 1)

    function add_to_result(data)
        for f in features
            if !haskey(ds_train.meta["features"][f], "onehot") && isnumber(ds_train.meta, f)
                result["$f-acc_count"] += prod(size(data[f]))
                for i in axes(data[f], 3)
                    result["$f-acc_sum"] += reduce(+, data[f][:, :, i]; dims=2)[:, 1]
                    result["$f-acc_sum_squared"] += reduce(+, data[f][:, :, i] .^ 2; dims=2)[
                        :, 1
                    ]
                end
                # result[f] = cat(
                #     result[f], [data[f][:, :, i] for i in axes(data[f], 3)]...; dims = 2)
            end
        end

        for tf in target_features
            if isnumber(ds_train.meta, tf)
                result["target|$tf-acc_count"] += prod(size(data[tf]))
                for i in axes(data[tf], 3)
                    result["target|$tf-acc_sum"] += reduce(+, data[tf][:, :, i]; dims=2)[
                        :, 1
                    ]
                    result["target|$tf-acc_sum_squared"] += reduce(
                        +, data[tf][:, :, i] .^ 2; dims=2
                    )[
                        :, 1
                    ]
                end
                # result["target|$tf"] = cat(result["target|$tf"],
                #     [tf_data[:, :, i] ./ Float32(data["dt"][i + 1] - data["dt"][i])
                #      for i in axes(tf_data, 3)]...;
                #     dims = 2)
            end
        end

        # result["edges-acc_count"] += prod(size(data["edge_features"]))
        # result["edges-acc_sum"] += reduce(+, data["edge_features"]; dims = 2)[:, 1]
        # result["edges-acc_sum"] += reduce(+, data["edge_features"] .^ 2; dims = 2)[:, 1]
    end

    p = Progress(
        length(train_loader) + length(valid_loader) + length(test_loader);
        desc="Calculating meanstd-norm: ",
        dt=1.0,
        barlen=50,
    )
    for data in train_loader
        add_to_result(data)
        next!(p)
    end
    for data in valid_loader
        add_to_result(data)
        next!(p)
    end
    for data in test_loader
        add_to_result(data)
        next!(p)
    end
    finish!(p)

    meanstd_dict = Dict{String,Any}()

    for f in features
        if !haskey(ds_train.meta["features"][f], "onehot") && isnumber(ds_train.meta, f)
            m = result["$f-acc_sum"] / max(result["$f-acc_count"], 1.0f0)
            s = sqrt.(
                max.(
                    Ref(0.0f0),
                    (
                        result["$f-acc_sum_squared"] / max(result["$f-acc_count"], 1.0f0) -
                        m .^ 2
                    ),
                ),
            )
            meanstd_dict[f] = (m, s)
        end
    end
    for f in target_features
        if isnumber(ds_train.meta, f)
            m = result["target|$f-acc_sum"] / max(result["target|$f-acc_count"], 1.0f0)
            s = sqrt.(
                max.(
                    Ref(0.0f0),
                    (
                        result["target|$f-acc_sum_squared"] /
                        max(result["target|$f-acc_count"], 1.0f0) - m .^ 2
                    ),
                ),
            )
            meanstd_dict["target|$f"] = (m, s)
        end
    end
    # m = result["edges-acc_sum"] / max(result["edges-acc_count"], 1.0f0)
    # s = sqrt.(max.(Ref(0.0f0),
    #     (result["edges-acc_sum_squared"] / max(result["edges-acc_count"], 1.0f0) -
    #      m .^ 2)))
    # meanstd_dict["edges"] = (m, s)

    return meanstd_dict
end

"""
    clear_line(move_up = true)

Deletes the content of the current line in the terminal.

## Arguments
- `move_up`: Whether the cursor should be moved up one line before deleting.
"""
function clear_line(move_up=true)
    if move_up
        print("\u1b[1F")    # Move cursor up one line and to start of line
    end
    print("\u1b[2K")    # Delete text in current line without moving cursor back
    print("\u1b[1G")    # Move cursor to start of line
end

"""
    clear_log(lines, move_up = true)

Deletes the content of the given number of lines in the terminal.

## Arguments
- `lines`: Number of lines to be deleted.
- `move_up`: Whether the cursor should be moved up one line before deleting.
"""
function clear_log(lines::Integer, move_up=true)
    @assert lines > 0 "The number of lines in the terminal to be cleared must be greater than 0 : lines == $lines"
    clear_line(move_up)
    for _ in 1:lines
        clear_line()
    end
end

"""
    update_meta!(path, norm_type)

Compute normalization statistics and write them into the dataset's `meta.json`.

Calls [`data_minmax`](@ref) or [`data_meanstd`](@ref) depending on `norm_type`,
then updates the per-feature entries in `meta.json` with the computed statistics.
Conflicting statistics from a different normalization type are removed.

## Arguments
- `path::String`: Dataset directory containing `meta.json`, `train.h5`, `valid.h5`, `test.h5`.
- `norm_type::Symbol`: One of `:online`, `:minmax`, or `:meanstd`.
  `:online` is a no-op (no precomputed statistics needed).

## Returns
- `String`: Path to the written `meta.json` file.
"""
function update_meta!(path::String, norm_type::Symbol)
    if norm_type ∉ (:online, :minmax, :meanstd)
        throw(
            ArgumentError(
                "Invalid norm_type=:$norm_type. Must be one of :online, :minmax, :meanstd."
            ),
        )
    end

    meta_path = joinpath(path, "meta.json")
    if !isfile(meta_path)
        throw(ArgumentError("meta.json not found at \"$meta_path\"."))
    end

    if norm_type == :online
        @info "norm_type=:online requires no precomputed statistics. meta.json unchanged."
        return meta_path
    end

    meta = json_parsefile(meta_path)

    conflicting_minmax = ("data_min", "data_max", "output_min", "output_max")
    conflicting_meanstd = ("data_mean", "data_std")

    if norm_type == :minmax
        stats = data_minmax(path)

        for (key, val) in stats
            if startswith(key, "target|")
                feat = key[8:end]
                if !haskey(meta["features"], feat)
                    continue
                end
                # Write output-specific keys only if they differ from input stats
                if haskey(stats, feat) && val != stats[feat]
                    meta["features"][feat]["output_min"] = Float64(val[1])
                    meta["features"][feat]["output_max"] = Float64(val[2])
                end
            else
                if !haskey(meta["features"], key)
                    continue
                end
                meta["features"][key]["data_min"] = Float64(val[1])
                meta["features"][key]["data_max"] = Float64(val[2])
                # Remove conflicting meanstd keys
                for ck in conflicting_meanstd
                    delete!(meta["features"][key], ck)
                end
            end
        end

    elseif norm_type == :meanstd
        stats = data_meanstd(path)

        for (key, val) in stats
            if startswith(key, "target|")
                # meanstd has no separate output keys; skip target entries
                continue
            end
            if !haskey(meta["features"], key)
                continue
            end
            meta["features"][key]["data_mean"] = Float64.(val[1])
            meta["features"][key]["data_std"] = Float64.(val[2])
            # Remove conflicting minmax keys
            for ck in conflicting_minmax
                delete!(meta["features"][key], ck)
            end
        end
    end

    open(meta_path, "w") do f
        json_print(f, meta, 2)
    end

    @info "Updated meta.json with $norm_type statistics at \"$meta_path\"."
    return meta_path
end
