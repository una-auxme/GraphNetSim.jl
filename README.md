# GraphNetSim.jl

[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://una-auxme.github.io/Optuna.jl/dev)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/JuliaDiff/BlueStyle)

## Overview

**GraphNetSim.jl** is a Julia package for training and evaluating Graph Neural Network (GNN) simulators for learning complex physical dynamics from trajectory data. It provides a comprehensive framework for:

- **Graph-based learning** of particle dynamics and fluid mechanics
- **Flexible training strategies** including batching and derivative-based approaches
- **Efficient GPU acceleration** via CUDA with automatic device management
- **Automatic normalization** with offline (min-max, mean-std) and online strategies
- **Long-term trajectory rollouts** using ODE solvers for generalization assessment
- **Feature noise injection** for robust model training

The package is build upon [**GraphNetCore.jl**](https://github.com/una-auxme/GraphNetCore.jl) for the underlying graph neural network architecture.

## Installation

To add GraphNetSim.jl to your Julia environment, use:

```julia
using Pkg
Pkg.add("GraphNetSim")
```

Or in the Julia REPL, press `]` to enter package mode and type:

```
pkg> add GraphNetSim
```

## Quick Start

Here's a minimal example of training a Graph Neural Network simulator on trajectory data:

```julia
using GraphNetSim
using Optimisers  # For optimization
using OrdinaryDiffEq  # For ODE solvers

# Path to your dataset containing train/valid/test splits
ds_path = "./path/to/dataset"

# Directory where checkpoints will be saved
cp_path = "./checkpoints"

# Train the network with default configuration
min_loss = train_network(
    Optimisers.Adam(1.0f-4),  # Optimizer and learning rate
    ds_path,
    cp_path;
    epochs=10,
    steps=50000,
    mps=15,                    # Message passing steps
    layer_size=128,            # Hidden layer dimension
    hidden_layers=2,           # Number of hidden layers per MLP module
    checkpoint=1000,           # Save checkpoint every N steps
    use_cuda=true,             # Enable GPU acceleration if available
    training_strategy=DerivativeTraining()  # Training strategy
)

# Evaluate the trained network with long-term rollouts
eval_network(
    ds_path,
    cp_path,
    "./results";               # Output directory
    solver=Tsit5(),            # ODE solver
    start=0.0f0,
    stop=1.0f0,
    saves=0.0:0.01:1.0,        # Time points to save
    mse_steps=0.0:0.1:1.0      # Time points for error metrics
)
```

Further examples will be added soon.

## Dataset Format

Datasets should be organized as:

```
dataset/
├── meta.json           # Metadata (feature specs, topology, etc.)
├── train.h5            # Training trajectories
├── valid.h5            # Validation trajectories
└── test.h5             # Test trajectories
```

The metadata file defines feature dimensions, node types, graph connectivity, and normalization settings.

## Key Features

- **Multiple Training Strategies**: Choose between `BatchingTraining` and `DerivativeTraining` to suit your problem
- **GPU-accelerated Training**: Automatic CUDA detection and memory management
- **Flexible Architecture**: Configurable message passing steps, layer sizes, and hidden layers
- **Progress Monitoring**: Built-in progress bars and logging for training and validation

## Architecture Overview

The package implements a graph-based approach to physics simulation:

1. **Graph Construction**: Particles and boundaries are represented as nodes; interactions are represented as edges based on spatial proximity
2. **Message Passing**: Graph neural networks aggregate information from neighboring particles through multiple message passing iterations
3. **Dynamics Prediction**: The network predicts accelerations, velocities, or other output features based on learned interactions
4. **ODE Integration**: Predicted dynamics are integrated forward in time using standard ODE solvers

## Configuration

Training is controlled via the `Args` structure, which accepts keyword arguments:

```julia
args = GraphNetSim.Args(
    mps=15,                                  # Message passing steps
    layer_size=128,                          # Hidden dimension
    hidden_layers=2,                         # Layers per MLP
    epochs=1,                                # Training epochs
    steps=10e6,                              # Total steps
    checkpoint=10000,                        # Checkpoint interval
    norm_steps=1000,                         # Online norm accumulation
    types_updated=[1],                       # Updated node types
    types_noisy=[0],                         # Noisy node types
    noise_stddevs=[0.0f0],                   # Noise levels
    training_strategy=DerivativeTraining(),  # Training strategy
    use_cuda=true,                           # GPU acceleration
    optimizer_learning_rate_start=1.0f-4,    # Initial learning rate
    optimizer_learning_rate_stop=nothing,    # Final learning rate (for scheduling)
    use_valid=true,                          # Use best validation checkpoint
    show_progress_bars=true                  # Show progress
)
```

See the [full API documentation](https://una-auxme.github.io/GraphNetSim.jl/dev/api/) for complete parameter details.

## Visualization

Export predicted trajectories as VTK files for visualization:

```julia
visualize(
    "trajectories.h5",           # Results file from eval_network
    "./vtk_output",              # Output directory
    "pos",                        # Position dataset name
    "prediction";                 # Subgroup to visualize
    Trajectorys=1:5              # Trajectory indices
)
```

## Related Packages

- [**PointNeighbors.jl**](https://github.com/una-auxme/PointNeighbors.jl): Efficient spatial indexing for neighbor queries
- [**GraphNetCore.jl**](https://github.com/una-auxme/GraphNetCore.jl): Core GNN architecture and normalization strategies
- [**DifferentialEquations.jl**](https://github.com/SciML/DifferentialEquations.jl): ODE solvers for trajectory integration

## References

This package is inspired by the Graph Network-based Simulator (GNS) framework:

- Sanchez-Gonzalez, A., Godwin, J., Pfaff, T., et al. (2020). "Learning to Simulate Complex Physics with Graph Networks." *Proceedings of the 37th International Conference on Machine Learning (ICML)*.

## Contributing

We welcome contributions to GraphNetSim.jl! Please follow the [ColPrac](https://github.com/SciML/ColPrac) guidelines for collaborative practices.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
