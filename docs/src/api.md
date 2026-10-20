# API Reference
## Configuration and Data Structures

### Args

```@docs
GraphNetSim.Args
```

### Dataset

```@docs
GraphNetSim.Dataset
```

### Model Configuration

```@docs
GraphNetSim.ModelConfig
GraphNetSim.save_model_config
GraphNetSim.load_model_config
```

#### Dataset Constructors

```@docs
GraphNetSim.Dataset(::String, ::String, ::Any)
GraphNetSim.Dataset(::Symbol, ::String, ::Any)
GraphNetSim.get_file
```

## Training and Evaluation

### Main Training Function

```@docs
GraphNetSim.train_network
```

### Internal Training Functions

```@docs
GraphNetSim.train_gns!
```

### Normalization Setup

```@docs
GraphNetSim.calc_norms
```

### Main Evaluation Function

```@docs
GraphNetSim.eval_network
GraphNetSim.eval_network!
```

### Extrapolation Beyond Ground Truth

```@docs
GraphNetSim.extrapolate_network
GraphNetSim.extrapolate_network!
```

## Training Strategies

### Abstract Base Type

```@docs
GraphNetSim.prepare_training
GraphNetSim.get_delta
init_train_step
train_step
validation_step
GraphNetSim._validation_step
GraphNetSim.batchTrajectory
```

### Concrete Strategies

```@docs
GraphNetSim.SingleShooting
GraphNetSim.MultipleShooting
GraphNetSim.DerivativeTraining
GraphNetSim.BatchingStrategy
```

### Default Sensitivity Algorithm

`BatchingStrategy`, `SingleShooting` and `MultipleShooting` default their `sense`
keyword to the algorithm returned here.

```@docs
GraphNetSim._default_solver_sense
```

### Schedulers

Per-pass selection rules used by `DerivativeTraining` and `BatchingStrategy`
to choose which training unit (timestep or batch) is processed at each
inner iteration of a trajectory pass.

```@docs
GraphNetSim.Scheduler
```

#### Stateless and loss-buffer schedulers

```@docs
GraphNetSim.Sequential
GraphNetSim.Shuffled
GraphNetSim.WorstLoss
GraphNetSim.UniqueWorst
GraphNetSim.WeightedWorst
GraphNetSim.TopKWorst
GraphNetSim.EpsilonGreedy
GraphNetSim.PercentileWorst
```

#### Stateful schedulers (per-pass state)

These keep per-unit state in mutable fields and rebuild it at the start of every
trajectory pass, mirroring the training loop's fresh losses buffer.

```@docs
GraphNetSim.UCB
GraphNetSim.LearningProgress
GraphNetSim.Staleness
```

#### Annealing schedulers (global step counter)

These keep a step counter across the whole run instead of resetting per pass, so
their behaviour shifts as training progresses.

```@docs
GraphNetSim.ProgressiveHorizon
GraphNetSim.AnnealedWeighted
```

#### Gradient-pooling scheduler

Pools a window of neighbouring timesteps into a single gradient step. Supported by
`DerivativeTraining` only; `BatchingStrategy` rejects it because it already pools
temporally over its ODE interval.

```@docs
GraphNetSim.TemporalWindow
GraphNetSim.window_indices
```

## Normalization and Data Statistics

### Computing Normalization Statistics

```@docs
GraphNetSim.data_minmax
GraphNetSim.data_meanstd
GraphNetSim.der_minmax
GraphNetSim.update_meta!
```

## Graph Construction and ODE Solving

### Building Computation Graphs

```@docs
GraphNetSim.build_graph
```

### ODE Integration

```@docs
GraphNetSim.rollout
```

## Data Utilities and Conversion

### Dataset Loading

```@docs
GraphNetSim.keystraj
GraphNetSim.MLUtils.getobs!
```

### Format Conversion

```@docs
GraphNetSim.csv_to_hdf5
GraphNetSim.vtk_to_hdf5
```

## Visualization

### VTK Export

```@docs
visualize
visualize_eval
```
