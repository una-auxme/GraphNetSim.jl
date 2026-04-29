#
# Copyright (c) 2026 Josef Kircher, Julian Trommer, Simon Küchle
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using HDF5

"""
    visualize(inPath, outFolder, Position, subgroupTrajectory, Parameters, Trajectorys, NumberOfTimesteps)

Reads HDF5 file into dictionary and writes VTK HDF5 file format with dictToVTKHDF().

Input file must have a group for each trajectory and subgroups for all trajectories used.
Datasets are linked to each subgroup of trajectory.
If only certain trajectories should be written, specify them with a Vector{Int} (minimum is 1).
Validation contains the names of the subgroups which will be written and
each trajectory must have the same number of timesteps.

Timesteps can be automatically detected if dataset "timesteps" is linked to each trajectory group.
This dataset must contain the number for the largest timestep and timesteps 1:1:max will be read.
When using timesteps, all datasets to be read need "[Int]" appended.

## Arguments
- `inPath::String`: Complete path ending with .h5 file.
- `outFolder::String`: Complete path for output folder.
- `Position::String`: Name of HDF5 dataset for position data.
- `subgroupTrajectory::String`: Name of subgroup within trajectory groups.
- `Parameters::Vector{String}`: Names of other HDF5 datasets to read (optional).
- `Trajectorys::Union{Vector{Int},Nothing}`: Trajectory indices to write (optional, auto-detected if nothing).
- `NumberOfTimesteps::Union{Vector{Int},Nothing}`: Timesteps to write (optional, auto-detected if nothing).

## Returns
- `Dict`: Dictionary mapping (trajectory, dataset_name, timestep) tuples to numerical arrays.
"""
function visualize(
    inPath::String,
    outFolder::String,
    Position::String,
    subgroupTrajectory::String,
    Parameters::Vector{<:String}=[],
    Trajectorys::Union{Vector{Int},Nothing}=nothing,
    NumberOfTimesteps::Union{Vector{Int},Nothing}=nothing,
)
    readDict = Dict{Tuple{Int,String,Int},AbstractArray{<:Number}}()
    currentTrajectory = 1

    groupCheck = HDF5.h5open(inPath, "r")
    trajectory = keys(groupCheck)
    #get all trajectory groups
    n_timesteps = Array{Int}(undef, 0)
    for topGroup in trajectory
        inputFile = HDF5.h5open(inPath, "r")

        try
            traj_group = inputFile[topGroup]
            scalar_max = HDF5.read(traj_group, "timesteps")
            top = traj_group[subgroupTrajectory]

            # Determine actual step count in this subgroup by scanning its keys
            # for datasets of the form "Position[N]" and taking the max N.
            # The trajectory-level `timesteps` scalar serves only as an upper
            # bound — extrapolation rollouts store a shorter `gt/` group.
            prefix = Position * "["
            actual_max = 0
            for k in keys(top)
                (startswith(k, prefix) && endswith(k, "]")) || continue
                n = tryparse(Int, k[(length(prefix) + 1):(end - 1)])
                n === nothing && continue
                actual_max = max(actual_max, n)
            end
            step_count = min(scalar_max, actual_max)
            push!(n_timesteps, step_count)

            for t in 1:step_count
                readDict[(currentTrajectory, Position, t)] = HDF5.read(
                    top, Position * "[$t]"
                )
                for name in Parameters
                    readDict[(currentTrajectory, name, t)] = HDF5.read(top, name * "[$t]")
                end
            end
        finally
            HDF5.close(inputFile)
        end

        currentTrajectory += 1
    end
    dictToVTKHDF(
        readDict,
        outFolder,
        Position,
        subgroupTrajectory,
        Parameters,
        Trajectorys,
        n_timesteps,
    )
    return readDict
end

"""
    dictToVTKHDF(inputDict, outFolder, Position, subgroupTrajectory, Parameters, Trajectorys, NumberOfTimesteps)

Converts dictionary data to VTK HDF5 (.vtkhdf) file format and writes to disk.

Takes dictionary with (trajectory, dataset_name, timestep) keys and writes corresponding VTK HDF5 files.
Timesteps and trajectories can be selectively written using Vector{Int} arguments.
Automatically creates directory structure and handles 2D/3D coordinate conversions.

## Arguments
- `inputDict::Dict{Tuple{Int,String,Int},AbstractArray}`: Dictionary mapping (trajectory, dataset_name, timestep) to data arrays.
- `outFolder::String`: Complete path for output folder.
- `Position::String`: Name of dataset in dictionary for position data.
- `subgroupTrajectory::String`: Name of subgroup for organizing output files.
- `Parameters::Vector{String}`: Names of additional datasets in dictionary to include (optional).
- `Trajectorys::Union{Vector{Int},Nothing}`: Trajectory indices to write (optional, auto-detected if nothing).
- `NumberOfTimesteps::Union{Vector{Int},Nothing}`: Timesteps to write (optional, auto-detected if nothing).

## Returns
- `Nothing`: Writes VTK HDF5 files to `outFolder`/{trajectory}/{subgroups}/ directories.
"""

function dictToVTKHDF(
    inputDict::Dict{Tuple{Int,String,Int},<:AbstractArray{<:Number}},
    outFolder::String,
    Position::String,
    subgroupTrajectory::String,
    Parameters::Vector{<:String}=[],
    Trajectorys::Union{Vector{Int},Nothing}=nothing,
    NumberOfTimesteps::Union{Vector{Int},Nothing}=nothing,
)
    keysDict = keys(inputDict)

    checkMinMax = Matrix{Int64}(undef, 0, 2)
    for x in keysDict
        checkMinMax = vcat(checkMinMax, [x[1] x[3]])
    end

    maxTrajectory = maximum(checkMinMax[:, 1])
    minTrajectory = minimum(checkMinMax[:, 1])
    minTimestep = minimum(checkMinMax[:, 2])
    maxTimestep = maximum(checkMinMax[:, 2])

    print("
    Values found in Dict: 
    minTimestep: $minTimestep
    maxTimestep: $maxTimestep
    minTrajectory: $minTrajectory
    maxTrajectory: $maxTrajectory
    \n")
    #get NumberOfTrajectorys/Timesteps automatically

    if Trajectorys === nothing
        Trajectorys = minTrajectory:1:maxTrajectory
    end

    if NumberOfTimesteps === nothing
        NumberOfTimesteps = minTimestep:1:maxTimestep
    end

    if !isdir(outFolder)
        mkpath(outFolder)
    end

    for k in Trajectorys
        originalPath = pwd()
        cd(outFolder)
        try
            mkdir("$k" * "Trajectory")
        catch e
            println("dir $k" * "Trajectory already existing \n")
        finally
            cd(originalPath)
        end
        #make dir in outputFolder for each Trajectory

        originalPath = pwd()
        cd(outFolder * "/" * "$k" * "Trajectory")
        try
            mkdir("$subgroupTrajectory")
        catch e
            println("dir" * "$subgroupTrajectory" * "already existing \n")
        finally
            cd(originalPath)
        end
        #make dir in Trajectory for subgroubTrajectory

        steps = NumberOfTimesteps[k]
        steps = collect(1:1:steps)
        for t in steps
            output =
                outFolder *
                "/" *
                "$k" *
                "Trajectory" *
                "/" *
                "$subgroupTrajectory" *
                "/" *
                "$subgroupTrajectory" *
                "_" *
                "$k" *
                "Trajectory" *
                "_$t.vtkhdf"
            outputFile = HDF5.h5open(output, "w")

            try
                top = HDF5.create_group(outputFile, "VTKHDF")
                HDF5.attributes(top)["Version"] = [2, 2]

                let s = "UnstructuredGrid"
                    dtype = HDF5.datatype(s)
                    HDF5.API.h5t_set_cset(dtype.id, HDF5.API.H5T_CSET_ASCII)
                    dspace = HDF5.dataspace(s)
                    attr = HDF5.create_attribute(top, "Type", dtype, dspace)
                    HDF5.write_attribute(attr, dtype, s)
                end
                #attribute "UnstructuredGrid" must be datatype ASCII

                if maximum(size(inputDict[(k, Position, t)])) <= 3
                    NumberOfPoints = size(inputDict[(k, Position, t)])[2]
                else
                    NumberOfPoints = maximum(size(inputDict[(k, Position, t)]))
                end

                dimension = minimum(size(inputDict[(k, Position, t)]))

                if dimension < 3
                    inputDict[(k, Position, t)] = vcat(
                        inputDict[(k, Position, t)], zeros(3 - dimension, NumberOfPoints)
                    )
                end

                top["NumberOfPoints"] = [NumberOfPoints]
                top["NumberOfCells"] = [NumberOfPoints]                  #each point is one cell
                top["NumberOfConnectivityIds"] = [NumberOfPoints]
                top["Connectivity"] = collect(0:(NumberOfPoints - 1))
                top["Offsets"] = collect(0:NumberOfPoints)

                type_data = Int8.(ones(NumberOfPoints))
                dt = HDF5.API.h5t_copy(HDF5.API.H5T_STD_U8LE)
                dset = create_dataset(top, "Types", HDF5.Datatype(dt), dataspace(type_data))
                HDF5.write(dset, type_data)
                #Types must be encoded in Int8(char)

                top["Points"] = inputDict[(k, Position, t)]

                PointData = create_group(top, "PointData")

                for name in Parameters
                    PointData[name] = inputDict[k, name, t]
                end
                #all parameters stored in PointData group
            finally
                HDF5.close(outputFile)
            end
        end
    end
end

"""
    visualize_eval(inPath, outFolder; Position, gt_subgroup, gt_parameters,
                   pred_subgroup, pred_parameters, Trajectorys)

One-shot VTK export for evaluation outputs: writes both the rollout and the
ground-truth subgroup. Safe for extrapolation — GT files are produced for
every timestep present in the `gt/` group, even when the rollout is longer.

## Arguments
- `inPath::String`: Complete path ending with .h5 file.
- `outFolder::String`: Complete path for output folder.
- `Position::String`: HDF5 dataset name for position data (default `"pos"`).
- `gt_subgroup::String`: Subgroup name for ground truth (default `"gt"`).
- `gt_parameters::Vector{<:AbstractString}`: Additional GT datasets (default `["vel", "acc"]`).
- `pred_subgroup::String`: Subgroup name for rollout (default `"prediction"`).
- `pred_parameters::Vector{<:AbstractString}`: Additional rollout datasets (default `["vel", "acc", "err"]`).
- `Trajectorys::Union{Vector{Int},Nothing}`: Trajectory indices to write (optional, auto-detected if nothing).
"""
function visualize_eval(
    inPath::String,
    outFolder::String;
    Position::String="pos",
    gt_subgroup::String="gt",
    gt_parameters::Vector{<:AbstractString}=["vel", "acc"],
    pred_subgroup::String="prediction",
    pred_parameters::Vector{<:AbstractString}=["vel", "acc", "err"],
    Trajectorys::Union{Vector{Int},Nothing}=nothing,
)
    visualize(inPath, outFolder, Position, pred_subgroup, pred_parameters, Trajectorys)
    visualize(inPath, outFolder, Position, gt_subgroup, gt_parameters, Trajectorys)
    return nothing
end
