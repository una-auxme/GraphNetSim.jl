#
# Copyright (c) 2026 Josef Kircher, Julian Trommer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

module GraphNetSim

using GraphNetCore

using CUDA
using Lux, Zygote
using LuxCUDA
using MLUtils
using Optimisers
using Zygote
using HDF5
using Plots

import SciMLBase: ODEProblem, SecondOrderODEProblem
import OrdinaryDiffEq: OrdinaryDiffEqAlgorithm, Tsit5
import ProgressMeter: Progress

import Base: @kwdef
import SciMLBase: solve, remake
import NearestNeighbors: KDTree, inrange
import HDF5: h5open, create_group, open_group
import ProgressMeter: next!, update!, finish!
import Statistics: mean

include("utils.jl")
include("graph.jl")
include("solve.jl")
include("dataset.jl")
include("visualize.jl")
include("../convert_csv/csvToh5.jl")

export SingleShooting, MultipleShooting, DerivativeTraining, BatchingStrategy

export train_network, eval_network, data_minmax, data_meanstd
export visualize
export csv_to_hdf5

"""
    Args

Configuration structure for training and evaluating Graph Neural Network simulators.

## Fields

### Network Architecture
- `mps::Integer=15`: Number of message passing steps (higher = more expressive but slower)
- `layer_size::Integer=128`: Latent dimension for MLP hidden layers
- `hidden_layers::Integer=2`: Number of hidden layers in each MLP module

### Training Configuration
- `epochs::Integer=1`: Number of passes over the entire dataset
- `steps::Integer=10e6`: Total number of training steps
- `checkpoint::Integer=10000`: Interval (in steps) for saving checkpoints
- `norm_steps::Integer=1000`: Steps to accumulate normalization statistics before weight updates
- `batchsize::Integer=1`: Batch size (currently limited to 1 - full trajectory per batch)

### Normalization
- `max_norm_steps::Integer=10.0f6`: Maximum steps for online normalizer accumulation

### Data Augmentation
- `types_updated::Vector{Integer}=[1]`: Node types whose features are predicted
- `types_noisy::Vector{Integer}=[0]`: Node types to which noise is added during training
- `noise_stddevs::Vector{Float32}=[0.0f0]`: Standard deviations for Gaussian noise (per type or broadcast)

### Training Strategy
- `training_strategy::TrainingStrategy=DerivativeTraining()`: Method for computing loss

### Hardware and Optimization
- `use_cuda::Bool=true`: Enable CUDA GPU acceleration (if available)
- `gpu_device::Union{Nothing,CuDevice}`: CUDA device to use (auto-selected if CUDA functional)
- `optimizer_learning_rate_start::Float32=1.0f-4`: Initial learning rate
- `optimizer_learning_rate_stop::Union{Nothing,Float32}=nothing`: Final learning rate (for decay schedule)

### Validation
- `show_progress_bars::Bool=true`: Show training progress bars
- `use_valid::Bool=true`: Load validation checkpoint (best loss) instead of final checkpoint
- `solver_valid::OrdinaryDiffEqAlgorithm=Tsit5()`: ODE solver for validation rollouts
- `solver_valid_dt::Union{Nothing,Float32}=nothing`: Fixed timestep for validation solver
- `reset_valid::Bool=false`: Reset validation after loading checkpoint
- `save_step::Bool=false`: Save loss at every step (can create large log files)
"""
@kwdef mutable struct Args
    mps::Integer = 15
    layer_size::Integer = 128
    hidden_layers::Integer = 2
    batchsize::Integer = 1
    epochs::Integer = 1
    steps::Integer = 10e6
    checkpoint::Integer = 10000
    norm_steps::Integer = 1000
    max_norm_steps::Integer = 10.0f6
    types_updated::Vector{Integer} = [1]
    types_noisy::Vector{Integer} = [0]
    noise_stddevs::Vector{Float32} = [0.0f0]
    training_strategy::TrainingStrategy = DerivativeTraining()
    use_cuda::Bool = true
    gpu_device::Union{Nothing,CuDevice} = CUDA.functional() ? CUDA.device() : nothing
    show_progress_bars::Bool = true
    use_valid::Bool = true
    solver_valid::OrdinaryDiffEqAlgorithm = Tsit5()
    solver_valid_dt::Union{Nothing,Float32} = nothing
    reset_valid::Bool = false
    optimizer_learning_rate_start::Float32 = 1.0f-4
    optimizer_learning_rate_stop::Union{Nothing,Float32} = nothing
    save_step::Bool = false
end

"""
    calc_norms(dataset::Dataset, device::Function, args::Args)

Initialize and compute feature normalizers from dataset statistics.

Computes normalization statistics for edge features, node features, and output features
based on metadata specifications. Supports offline min/max and mean/std normalization,
boolean encoding, one-hot encoded features, and online accumulation strategies.

## Arguments
- `dataset::Dataset`: Dataset object containing feature metadata and specifications.
- `device::Function`: Device placement function (cpu_device or gpu_device).
- `args::Args`: Configuration struct with norm_steps for online normalizer accumulation.

## Returns
- `Tuple`: (quantities, e_norms, n_norms, o_norms)
  - `quantities::Int`: Total number of input feature dimensions
  - `e_norms`: Dictionary or single normalizer for edge features
  - `n_norms::Dict`: Dictionary mapping node feature names to normalizers
  - `o_norms::Dict`: Dictionary mapping output feature names to normalizers

## Normalizer Types
- `NormaliserOfflineMinMax`: Fixed min/max normalization with learnable target range
- `NormaliserOfflineMeanStd`: Fixed mean/standard deviation normalization
- `NormaliserOnline`: Accumulates statistics online during training

## Notes
- One-hot encoded Int32 features are expanded to multiple dimensions
- Boolean features are mapped to [0.0, 1.0] range
- Distance features add dimensions for domain boundary constraints
"""
function calc_norms(dataset, device, args)
    quantities = 0
    n_norms = Dict{String,Union{NormaliserOffline,NormaliserOnline}}()
    o_norms = Dict{String,Union{NormaliserOffline,NormaliserOnline}}()

    e_norms = NormaliserOnline(dataset.meta["dims"] + 1, device)

    input_features = dataset.meta["input_features"]
    output_features = dataset.meta["output_features"]

    for feature in dataset.meta["feature_names"]
        feature_dim = dataset.meta["features"][feature]["dim"]
        if feature in input_features
            quantities += feature_dim
        end

        if getfield(
            Base, Symbol(uppercasefirst(dataset.meta["features"][feature]["dtype"]))
        ) == Bool
            quantities += 1
            n_norms[feature] = NormaliserOfflineMinMax(0.0f0, 1.0f0)
            if feature in output_features
                o_norms[feature] = NormaliserOfflineMinMax(0.0f0, 1.0f0)
            end
        elseif getfield(
            Base, Symbol(uppercasefirst(dataset.meta["features"][feature]["dtype"]))
        ) == Int32
            if haskey(dataset.meta["features"][feature], "onehot") &&
                dataset.meta["features"][feature]["onehot"]
                quantities +=
                    dataset.meta["features"][feature]["data_max"] -
                    dataset.meta["features"][feature]["data_min"] + 1
                if haskey(dataset.meta["features"][feature], "target_min") &&
                    haskey(dataset.meta["features"][feature], "target_max")
                    n_norms[feature] = NormaliserOfflineMinMax(
                        0.0f0,
                        1.0f0,
                        Float32(dataset.meta["features"][feature]["target_min"]),
                        Float32(dataset.meta["features"][feature]["target_max"]),
                    )
                    if feature in output_features
                        o_norms[feature] = NormaliserOfflineMinMax(
                            0.0f0,
                            1.0f0,
                            Float32(dataset.meta["features"][feature]["target_min"]),
                            Float32(dataset.meta["features"][feature]["target_max"]),
                        )
                    end
                else
                    n_norms[feature] = NormaliserOfflineMinMax(0.0f0, 1.0f0)
                    if feature in output_features
                        o_norms[feature] = NormaliserOfflineMinMax(0.0f0, 1.0f0)
                    end
                end
            else
                throw(
                    ErrorException(
                        "Int32 types that are not onehot types are not supported yet."
                    ),
                )
            end
        else
            if haskey(dataset.meta["features"][feature], "data_min") &&
                haskey(dataset.meta["features"][feature], "data_max")
                if haskey(dataset.meta["features"][feature], "target_min") &&
                    haskey(dataset.meta["features"][feature], "target_max")
                    if feature in input_features
                        n_norms[feature] = NormaliserOfflineMinMax(
                            Float32(dataset.meta["features"][feature]["data_min"]),
                            Float32(dataset.meta["features"][feature]["data_max"]),
                            Float32(dataset.meta["features"][feature]["target_min"]),
                            Float32(dataset.meta["features"][feature]["target_max"]),
                        )
                    elseif feature in output_features
                        if haskey(dataset.meta["features"][feature], "output_min") &&
                            haskey(dataset.meta["features"][feature], "output_max")
                            o_norms[feature] = NormaliserOfflineMinMax(
                                Float32(dataset.meta["features"][feature]["output_min"]),
                                Float32(dataset.meta["features"][feature]["output_max"]),
                                Float32(dataset.meta["features"][feature]["target_min"]),
                                Float32(dataset.meta["features"][feature]["target_max"]),
                            )
                        else
                            o_norms[feature] = NormaliserOnline(
                                feature_dim, device; max_acc=Float32(args.norm_steps)
                            )
                        end
                    end
                else
                    if feature in input_features
                        n_norms[feature] = NormaliserOfflineMinMax(
                            Float32(dataset.meta["features"][feature]["data_min"]),
                            Float32(dataset.meta["features"][feature]["data_max"]),
                        )
                    elseif feature in output_features
                        if haskey(dataset.meta["features"][feature], "output_min") &&
                            haskey(dataset.meta["features"][feature], "output_max")
                            o_norms[feature] = NormaliserOfflineMinMax(
                                Float32(dataset.meta["features"][feature]["output_min"]),
                                Float32(dataset.meta["features"][feature]["output_max"]),
                            )
                        else
                            o_norms[feature] = NormaliserOnline(
                                feature_dim, device; max_acc=Float32(args.norm_steps)
                            )
                        end
                    end
                end
            elseif haskey(dataset.meta["features"][feature], "data_mean") &&
                haskey(dataset.meta["features"][feature], "data_std")
                if feature in input_features
                    n_norms[feature] = NormaliserOfflineMeanStd(
                        Float32.(dataset.meta["features"][feature]["data_mean"]),
                        Float32.(dataset.meta["features"][feature]["data_std"]),
                        device,
                    )
                elseif feature in output_features
                    o_norms[feature] = NormaliserOfflineMeanStd(
                        Float32.(dataset.meta["features"][feature]["data_mean"]),
                        Float32.(dataset.meta["features"][feature]["data_std"]),
                        device,
                    )
                end
            else
                if feature in input_features
                    n_norms[feature] = NormaliserOnline(
                        feature_dim, device; max_acc=Float32(args.norm_steps)
                    )
                elseif feature in output_features
                    o_norms[feature] = NormaliserOnline(
                        feature_dim, device; max_acc=Float32(args.norm_steps)
                    )
                end
            end
        end
    end
    if dataset.meta["features"]["node_type"]["data_max"] -
       dataset.meta["features"]["node_type"]["data_min"] + 1 > 1
        quantities += length(dataset.meta["bounds"]) * 2
    else
        quantities += length(dataset.meta["bounds"])
    end

    return quantities, e_norms, n_norms, o_norms
end

"""
    train_network(opt, ds_path::String, cp_path::String; kws...)

Train a Graph Neural Network simulator on trajectory data.

Initializes a graph network, loads dataset, computes normalization statistics, 
and performs supervised training using the specified training strategy. Validates 
periodically on validation set and saves checkpoints for the best model.

## Arguments
- `opt`: Optimizer configuration (e.g., `Optimisers.Adam(1f-4)`).
- `ds_path::String`: Path to dataset directory (must contain train, valid, test splits).
- `cp_path::String`: Path where checkpoints and logs are saved.

## Keyword Arguments
- `mps::Int=15`: Number of message passing steps in the network.
- `layer_size::Int=128`: Latent dimension of hidden MLP layers.
- `hidden_layers::Int=2`: Number of hidden layers in each MLP.
- `batchsize::Int=1`: Batch size for training (default uses full trajectories).
- `epochs::Int=1`: Number of training epochs.
- `steps::Int=10e6`: Total number of training steps.
- `checkpoint::Int=10000`: Create checkpoint every N steps.
- `norm_steps::Int=1000`: Steps for accumulating normalization statistics without updates.
- `max_norm_steps::Float32=10.0f6`: Maximum steps for online normalizers.
- `types_updated::Vector{Int}=[1]`: Node types whose features are predicted.
- `types_noisy::Vector{Int}=[0]`: Node types to which noise is added.
- `noise_stddevs::Vector{Float32}=[0.0f0]`: Standard deviations for Gaussian noise.
- `training_strategy::TrainingStrategy=DerivativeTraining()`: Training method to use.
- `use_cuda::Bool=true`: Use CUDA GPU if available.
- `solver_valid::OrdinaryDiffEqAlgorithm=Tsit5()`: ODE solver for validation rollouts.
- `solver_valid_dt::Union{Nothing,Float32}=nothing`: Fixed timestep for validation (if set).
- `optimizer_learning_rate_start::Float32=1.0f-4`: Initial learning rate.
- `optimizer_learning_rate_stop::Union{Nothing,Float32}=nothing`: Final learning rate for decay schedule.
- `show_progress_bars::Bool=true`: Show training progress bars.
- `use_valid::Bool=true`: Use validation checkpoint for early stopping.

## Returns
- `Float32`: Minimum validation loss achieved during training.

## Training Strategies
- `DerivativeTraining`: Train on current step derivatives (collocation)
- `BatchingStrategy`: Custom batching of trajectory segments

## Example #TODO add example with different training strategies

```julia
train_network(
    Optimisers.Adam(1f-4),
    "./data",
    "./checkpoints";
    epochs=10,
    steps=50000,
    mps=15,
    layer_size=128,
    training_strategy=DerivativeTraining()
)
```
"""
function train_network(opt, ds_path, cp_path; kws...)
    args = Args(; kws...)

    if CUDA.functional() && args.use_cuda
        @info "Training on CUDA GPU..."
        CUDA.device!(args.gpu_device)
        CUDA.allowscalar(false)
        device = gpu_device()
    else
        @info "Training on CPU..."
        device = cpu_device()
    end

    @info "Training with $(typeof(args.training_strategy))..."

    println("Loading training data...")
    ds_train = Dataset(:train, ds_path, args)
    ds_train.meta["types_updated"] = args.types_updated
    ds_train.meta["types_noisy"] = args.types_noisy
    ds_train.meta["noise_stddevs"] = args.noise_stddevs
    ds_train.meta["device"] = device
    ds_valid = Dataset(:valid, ds_path, args)
    ds_valid.meta["types_updated"] = args.types_updated
    ds_valid.meta["types_noisy"] = args.types_noisy
    ds_valid.meta["noise_stddevs"] = args.noise_stddevs
    ds_valid.meta["device"] = device
    ds_valid.meta["training_strategy"] = nothing

    @info "Training data loaded!"
    Threads.nthreads() < 2 &&
        @warn "Julia is currently running on a single thread! Start Julia with more threads to speed up data loading."

    println("Building model...")

    quantities, e_norms, n_norms, o_norms = calc_norms(ds_train, device, args)
    dims = ds_train.meta["dims"]
    outputs = 0
    for tf in ds_train.meta["output_features"]
        outputs += ds_train.meta["features"][tf]["dim"]
    end

    gns, opt_state, df_train, df_valid, df_step = load(
        quantities,
        typeof(dims) <: AbstractArray ? length(dims) : dims,
        e_norms,
        n_norms,
        o_norms,
        outputs,
        args.mps,
        args.layer_size,
        args.hidden_layers,
        opt,
        device,
        cp_path,
    ) # geht mit dims oder dimensions of array

    if isnothing(opt_state)
        opt_state = Optimisers.setup(opt, gns.ps)
    end

    Lux.trainmode(gns.st)

    @info "Model built!"
    print("Compiling code...")
    print("\u1b[1G")

    min_validation_loss = train_gns!(
        gns,
        opt_state,
        ds_train,
        ds_valid,
        df_train,
        df_valid,
        df_step,
        device,
        cp_path,
        args,
    )

    return min_validation_loss
end

"""
    train_gns!(gns::GraphNetwork, opt_state, ds_train::Dataset, ds_valid::Dataset, df_train, df_valid, df_step, device::Function, cp_path::String, args::Args)

Execute the main training loop for a Graph Neural Network simulator.

Performs supervised learning with periodic validation, checkpoint saving, and learning rate scheduling.
Supports various training strategies (derivative, batching) and handles
feature noise injection, gradient accumulation over multiple time steps, and online normalizer updates.

## Arguments
- `gns::GraphNetwork`: Graph network model containing parameters and normalizers.
- `opt_state`: Optimizer state from Optimisers.jl.
- `ds_train::Dataset`: Training dataset with trajectories and metadata.
- `ds_valid::Dataset`: Validation dataset for monitoring training progress.
- `df_train`: DataFrame storing training loss at checkpoints.
- `df_valid`: DataFrame storing best validation losses.
- `df_step`: DataFrame storing loss at each step (if save_step enabled).
- `device::Function`: Device placement function (cpu_device or gpu_device).
- `cp_path::String`: Path for saving checkpoints and logs.
- `args::Args`: Configuration including optimizer, strategy, and training parameters.

## Algorithm

For each training step:
1. Prepare input data and compute node features
2. For each time step in trajectory:
   - Build computation graph with neighborhood connections
   - Forward pass through network
   - Compute loss depending on training strategy
   - Accumulate gradients
3. Update parameters after accumulation window
4. Optionally decay learning rate
5. Every N steps: validate on full trajectories and save checkpoint if improved

## Returns
- `Float32`: Minimum validation loss achieved.

## Notes
- First `norm_steps` steps only accumulate normalization statistics
- Validation using long trajectory rollouts occurs at checkpoint intervals
- Checkpoints saved to `cp_path/valid` when validation loss improves
- Final checkpoint always saved at `cp_path`
"""
function train_gns!(
    gns::GraphNetwork,
    opt_state,
    ds_train::Dataset,
    ds_valid::Dataset,
    df_train,
    df_valid,
    df_step,
    device::Function,
    cp_path,
    args::Args,
)
    checkpoint = length(df_train.step) > 0 ? last(df_train.step) : 0
    step = checkpoint
    cp_progress = 0
    min_validation_loss = length(df_valid.loss) > 0 ? minimum(df_valid.loss) : Inf32
    last_validation_loss = min_validation_loss

    pr = Progress(
        args.epochs*args.steps;
        desc="Training progress: ",
        dt=1.0,
        barlen=50,
        start=checkpoint,
        showspeed=true,
        enabled=args.show_progress_bars,
    )

    local tmp_loss = 0.0f0
    local avg_loss = 0.0f0

    output_features = ds_train.meta["output_features"]

    if (typeof(args.training_strategy) <: DerivativeStrategy)
        target_features = ds_train.meta["derivative_target_features"]
    else
        target_features = ds_train.meta["solver_target_features"]
    end

    train_loader = DataLoader(
        ds_train; batchsize=-1, buffer=false, parallel=true, shuffle=true
    )
    valid_loader = DataLoader(ds_valid; batchsize=-1, buffer=false, parallel=true)

    while step < args.steps
        for data in train_loader
            delta = get_delta(args.training_strategy, data["trajectory_length"])

            node_type = device(
                Float32.(
                    GraphNetCore.one_hot(
                        vec(data["node_type"][:, :, 1]),
                        ds_train.meta["features"]["node_type"]["data_max"] -
                        ds_train.meta["features"]["node_type"]["data_min"] + 1,
                        1 - ds_train.meta["features"]["node_type"]["data_min"],
                    ),
                ),
            )

            batches = nothing
            if args.training_strategy isa BatchingStrategy
                batches = batchTrajectory(args.training_strategy, data)
            end
            for datapoint in 1:delta
                train_tuple = init_train_step(
                    args.training_strategy,
                    (
                        gns,
                        data,
                        ds_train.meta,
                        output_features,
                        target_features,
                        node_type,
                        data["mask"],
                        data["val_mask"],
                        ds_train.meta["device"],
                        datapoint,
                        batches,
                        args.show_progress_bars,
                    ),
                )
                gs, losses = train_step(args.training_strategy, train_tuple)
                tmp_loss += sum(losses)
                if args.save_step
                    push!(df_step, [datapoint, sum(losses)])
                end
                if step + datapoint > args.norm_steps
                    for i in eachindex(gs)
                        opt_state, ps = Optimisers.update(opt_state, gns.ps, gs[i])
                        gns.ps = ps

                        if !isnothing(args.optimizer_learning_rate_stop) &&
                            (step + datapoint) % 100 == 0
                            Optimisers.adjust!(
                                opt_state,
                                Float32(
                                    args.optimizer_learning_rate_start +
                                    (
                                        args.optimizer_learning_rate_start -
                                        args.optimizer_learning_rate_stop
                                    )*0.1^(step*5e6),
                                ),
                            ) # Learn rate decay from GNS paper
                        end
                    end
                    update!(
                        pr,
                        step + datapoint;
                        showvalues=[
                            (:train_step, "$(step + datapoint)/$(args.epochs*args.steps)"),
                            (:train_loss, sum(losses)),
                            (
                                :checkpoint,
                                length(df_train.step) > 0 ? last(df_train.step) : 0,
                            ),
                            (
                                :data_interval,
                                if delta isa Vector
                                    "$datapoint : [$(delta[1]),...,$(delta[end])]"
                                else
                                    delta
                                end,
                            ),
                            (:min_validation_loss, min_validation_loss),
                            (:last_validation_loss, last_validation_loss),
                        ],
                    )
                else
                    update!(
                        pr,
                        step + datapoint;
                        showvalues=[
                            (:train_step, "$(step + datapoint)/$(args.epochs*args.steps)"),
                            (:train_loss, "acc norm stats..."),
                            (:checkpoint, 0),
                        ],
                    )
                end
            end

            cp_progress += delta
            step += delta
            avg_loss += tmp_loss
            tmp_loss = 0.0f0

            if step > args.norm_steps && cp_progress >= args.checkpoint
                push!(df_train, [step, avg_loss / Float32(cp_progress)])

                traj_idx = 1
                valid_error = 0.0f0
                pr_valid = Progress(
                    ds_valid.meta["n_trajectories"];
                    desc="Validation progress: ",
                    barlen=50,
                    enabled=args.show_progress_bars,
                )

                if args.show_progress_bars
                    print("\n\n\n\n\n\n\n")
                end

                for data_valid in valid_loader
                    print("\n\n\n")
                    pr_solver = ProgressUnknown(;
                        desc="Trajectory $(traj_idx)/$(length(valid_loader)): ",
                        showspeed=true,
                        enabled=args.show_progress_bars,
                    )

                    node_type_valid = device(
                        Float32.(
                            GraphNetCore.one_hot(
                                vec(data_valid["node_type"][:, :, 1]),
                                ds_valid.meta["features"]["node_type"]["data_max"] -
                                ds_valid.meta["features"]["node_type"]["data_min"] + 1,
                                1 - ds_valid.meta["features"]["node_type"]["data_min"],
                            ),
                        ),
                    )

                    ve = validation_step(
                        args.training_strategy,
                        (
                            gns,
                            data_valid,
                            ds_valid.meta,
                            get_delta(
                                args.training_strategy, data_valid["trajectory_length"]
                            ),
                            args.solver_valid,
                            args.solver_valid_dt,
                            node_type_valid,
                            pr_solver,
                        ),
                    )

                    valid_error += ve

                    next!(
                        pr_valid;
                        showvalues=[
                            (:trajectory, "$traj_idx/$(ds_valid.meta["n_trajectories"])"),
                            (:valid_loss, "$(valid_error / traj_idx)"),
                        ],
                    )
                    traj_idx += 1
                end

                if valid_error / ds_valid.meta["n_trajectories"] < min_validation_loss
                    push!(df_valid, [step, valid_error / ds_valid.meta["n_trajectories"]])
                    save!(
                        gns,
                        opt_state,
                        df_train,
                        df_valid,
                        df_step,
                        step,
                        valid_error / ds_valid.meta["n_trajectories"],
                        joinpath(cp_path, "valid");
                        is_training=false,
                    )
                    min_validation_loss = valid_error / ds_valid.meta["n_trajectories"]
                    cp_progress = args.checkpoint
                end
                last_validation_loss = valid_error / ds_valid.meta["n_trajectories"]
                if !args.show_progress_bars
                    println("Train step: $(step)/$(args.epochs*args.steps)")
                    println(
                        "Checkpoint: $(length(df_train.step) > 0 ? last(df_train.step) : 0)"
                    )
                    println("min_validation_loss: $min_validation_loss")
                    println("last_validation_loss: $last_validation_loss")
                end
                save!(
                    gns,
                    opt_state,
                    df_train,
                    df_valid,
                    df_step,
                    step,
                    valid_error / ds_valid.meta["n_trajectories"],
                    cp_path;
                    is_training=false,
                )
                avg_loss = 0.0f0
                cp_progress = 0
            end
        end
    end
    finish!(pr)
    return min_validation_loss
end

"""
    eval_network(ds_path::String, cp_path::String, out_path::String, solver=nothing; start, stop, dt=nothing, saves, mse_steps, kws...)

Evaluate a trained Graph Neural Network simulator on test trajectories.

Loads a trained network from checkpoint, performs long-term trajectory rollouts,
computes error metrics against ground truth, and saves results to disk.

## Arguments
- `ds_path::String`: Path to dataset directory containing test split.
- `cp_path::String`: Path to checkpoint directory (contains model parameters).
- `out_path::String`: Path where evaluation results are saved.
- `solver`: ODE solver for long-term predictions (e.g., `Tsit5()`).



## Keyword Arguments
- `start::Real`: Start time for evaluation.
- `stop::Real`: End time for evaluation.
- `saves::AbstractVector`: Time points where solution is saved.
- `mse_steps::AbstractVector`: Time points where error metrics are computed.
- `dt::Union{Nothing,Real}=nothing`: Fixed timestep for solver (if applicable).
- `mps::Int=15`: Number of message passing steps (must match training config).
- `layer_size::Int=128`: Hidden layer size (must match training config).
- `hidden_layers::Int=2`: Number of hidden layers (must match training config).
- `types_updated::Vector{Int}=[1]`: Updated node types (must match training config).
- `use_cuda::Bool=true`: Use CUDA GPU if available.
- `use_valid::Bool=true`: Load from best validation checkpoint instead of final checkpoint.

## Output

Saves results to `out_path/{solver_name}/trajectories.h5`:
- Ground truth positions, velocities, accelerations
- Predicted positions, velocities, accelerations
- Prediction errors for each trajectory

## Example

```julia
eval_network(
    "./data",
    "./checkpoints",
    "./results";
    solver=Tsit5(),
    start=0.0f0,
    stop=1.0f0,
    dt=0.01f0,
    saves=0.0:0.01:1.0,
    mse_steps=0.0:0.1:1.0
)
```
"""
function eval_network(
    ds_path,
    cp_path::String,
    out_path::String,
    solver;
    start,
    stop,
    dt=nothing,
    saves,
    mse_steps,
    kws...,
)
    args = Args(; kws...)

    if CUDA.functional() && args.use_cuda
        @info "Evaluating on CUDA GPU..."
        CUDA.device!(args.gpu_device)
        CUDA.allowscalar(false)
        device = gpu_device()
    else
        @info "Evaluating on CPU..."
        device = cpu_device()
    end

    @info "Using Lux as backend..."

    println("Loading evaluation data...")
    ds_test = Dataset(:test, ds_path, args)
    ds_test.meta["device"] = device
    ds_test.meta["training_strategy"] = nothing

    # clear_log(1, false)
    @info "Evaluation data loaded!"
    Threads.nthreads() < 2 &&
        @warn "Julia is currently running on a single thread! Start Julia with more threads to speed up data loading."

    println("Building model...")

    quantities, e_norms, n_norms, o_norms = calc_norms(ds_test, device, args)

    dims = ds_test.meta["dims"]
    outputs = 0
    for tf in ds_test.meta["output_features"]
        outputs += ds_test.meta["features"][tf]["dim"]
    end

    gns, _, _, _ = load(
        quantities,
        typeof(dims) <: AbstractArray ? length(dims) : dims,
        e_norms,
        n_norms,
        o_norms,
        outputs,
        args.mps,
        args.layer_size,
        args.hidden_layers,
        nothing,
        device,
        args.use_valid ? joinpath(cp_path, "valid") : cp_path,
    )

    Lux.testmode(gns.st)

    # clear_log(1, false)
    @info "Model built!"

    eval_network!(
        solver, gns, ds_test, device, out_path, start, stop, dt, saves, mse_steps, args
    )
end

"""
    eval_network!(solver, gns::GraphNetwork, ds_test::Dataset, device::Function, out_path::String, start::Real, stop::Real, dt, saves, mse_steps, args::Args)

Perform evaluation loops and trajectory rollouts for all test samples.

Executes long-term predictions for each test trajectory, computes performance metrics
relative to ground truth, and saves trajectories and errors to HDF5 format.

## Arguments
- `solver`: ODE solver for trajectory rollouts (or `nothing` for collocation).
- `gns::GraphNetwork`: Trained graph network model.
- `ds_test::Dataset`: Test dataset with trajectories.
- `device::Function`: Device placement function.
- `out_path::String`: Output directory for results.
- `start::Real`: Evaluation start time.
- `stop::Real`: Evaluation end time.
- `dt`: Fixed timestep (or `nothing` for adaptive).
- `saves`: Time points to save solution.
- `mse_steps`: Time points to compute errors.
- `args::Args`: Configuration parameters.

## Algorithm

For each test trajectory:
1. Extract initial conditions from data
2. Create computation graph with graph network
3. Roll out trajectory using ODE solver for specified duration
4. Extract position, velocity, and acceleration from solution
5. Compute mean squared error against ground truth
6. Report cumulative error at specified time points

## Returns
- `Tuple`: (traj_ops, errors)
  - `traj_ops::Dict`: Dictionary of trajectories with ground truth and predictions
  - `errors::Dict`: Squared errors for each trajectory

## Output Files

Creates `{out_path}/{solver_name}/trajectories.h5` containing:
- Ground truth and predicted trajectories
- Error fields for each time step
- Properly indexed for easy post-processing

"""
function eval_network!(
    solver,
    gns::GraphNetwork,
    ds_test::Dataset,
    device::Function,
    out_path,
    start,
    stop,
    dt,
    saves,
    mse_steps,
    args::Args,
)
    local traj_ops = Dict{
        Tuple{Int,String},
        NamedTuple{
            (:pos, :vel, :acc),Tuple{Array{Float32,3},Array{Float32,3},Array{Float32,3}}
        },
    }()
    local errors = Dict{Tuple{Int,String},Array{Float32,3}}()
    local timesteps = Dict{Tuple{Int,String},Array{Float32,1}}()

    test_loader = DataLoader(ds_test; batchsize=-1, buffer=false, parallel=true)

    for (ti, data) in enumerate(test_loader)
        target_features = ds_test.meta["solver_target_features"]
        output_features = ds_test.meta["output_features"]
        println("Rollout trajectory $ti...")

        if length(test_loader) > 1
            start = 0.0f0
            dt = data["dt"] # TODO dt can be an array?
            stop = round((data["trajectory_length"] - 1) * dt; digits=6)
            saves = start:dt:stop
            mse_steps = saves
        end

        stepstart = round(Int, ((start/dt) + 1))

        initial_state = Dict(
            "position" => data["position"][:, :, stepstart],
            "velocity" => data["velocity"][:, :, stepstart],
        )

        node_type = device(
            Float32.(
                GraphNetCore.one_hot(
                    vec(data["node_type"][:, :, 1]),
                    ds_test.meta["features"]["node_type"]["data_max"] -
                    ds_test.meta["features"]["node_type"]["data_min"] + 1,
                    1 - ds_test.meta["features"]["node_type"]["data_min"],
                ),
            ),
        )

        pr = ProgressUnknown(;
            desc="Trajectory $ti/$(length(test_loader)): ",
            showspeed=true,
            enabled=args.show_progress_bars,
        )

        sol = rollout(
            solver,
            gns,
            initial_state,
            output_features,
            ds_test.meta,
            target_features,
            node_type,
            data["mask"],
            data["val_mask"],
            start,
            stop,
            dt,
            saves,
            device,
            pr,
        )

        sol_acc = sol(sol.t, Val{1})
        sol_pos = [u.x for u in sol.u]
        sol_vel = [u.dx for u in sol.u]
        sol_acc = [u.dx for u in sol_acc.u]

        #convert array of matix into multidimensional array
        sol_pos = cpu_device()(cat(sol_pos...; dims=3))
        sol_vel = cat(sol_vel...; dims=3)
        sol_acc = cat(sol_acc...; dims=3)

        timesteps[(ti, "timesteps")] = sol.t

        prediction = (pos=sol_pos, vel=sol_vel, acc=sol_acc)

        #select timestep for prediction
        gt_pos = device(data["position"])
        gt_pos = cpu_device()(
            gt_pos[:, :, stepstart:(stepstart + size(prediction.pos, 3) - 1)]
        )
        gt_vel = device(data["velocity"])
        gt_vel = cpu_device()(
            gt_vel[:, :, stepstart:(stepstart + size(prediction.vel, 3) - 1)]
        )
        gt_acc = device(data["acceleration"])
        gt_acc = cpu_device()(
            gt_acc[:, :, stepstart:(stepstart + size(prediction.acc, 3) - 1)]
        )

        gt = (pos=gt_pos, vel=gt_vel, acc=gt_acc)

        print("\r\u1b[K")
        @info "Rollout trajectory $ti completed!"

        println("MSE of state prediction:")
        error = mean(
            (
                prediction.pos[:, Array(data["mask"]), :] -
                gt.pos[:, Array(data["mask"]), :]
            ) .^ 2;
            dims=2,
        )

        for horizon in mse_steps
            err = mean(error[:, 1, findfirst(x -> x == horizon, saves)])
            cum_err = mean(error[:, 1, 1:findfirst(x -> x == horizon, saves)])
            println(
                "  Trajectory $ti | mse t=$(horizon): $err | cum_mse t=$(horizon): $cum_err | cum_rmse t=$(horizon): $(sqrt(cum_err))",
            )
        end

        accuracyError =
            (
                prediction.pos[:, Array(data["mask"]), :] .-
                gt.pos[:, Array(data["mask"]), :]
            ) .^ 2

        traj_ops[(ti, "gt")] = cpu_device()(gt)
        traj_ops[(ti, "prediction")] = cpu_device()(prediction)
        errors[(ti, "error")] = cpu_device()(accuracyError)
    end

    eval_path = joinpath(
        out_path, isnothing(solver) ? "collocation" : lowercase("$(nameof(typeof(solver)))")
    )
    mkpath(eval_path)

    h5open(joinpath(eval_path, "trajectories.h5"), "w") do f
        for i in 1:maximum(getfield.(keys(traj_ops), 1))
            currentTrajectory = create_group(f, string("trajectory_$i"))
            currentTrajectory["timesteps"] = size(timesteps[(i, "timesteps")])[1]

            sub_g = create_group(currentTrajectory, "prediction")
            prediction = traj_ops[i, "prediction"]

            for k in axes(prediction.pos, 3)
                sub_g["pos[$k]"] = prediction.pos[:, :, k]
                sub_g["vel[$k]"] = prediction.vel[:, :, k]
                sub_g["acc[$k]"] = prediction.acc[:, :, k]
                sub_g["err[$k]"] = errors[i, "error"][:, :, k]
            end

            sub_g = create_group(currentTrajectory, "gt")
            gt = traj_ops[i, "gt"]

            for k in axes(gt.pos, 3)
                sub_g["pos[$k]"] = gt.pos[:, :, k]
                sub_g["vel[$k]"] = gt.vel[:, :, k]
                sub_g["acc[$k]"] = gt.acc[:, :, k]
            end
        end
    end
    return traj_ops, errors
    @info "Evaluation completed!"
end

end
