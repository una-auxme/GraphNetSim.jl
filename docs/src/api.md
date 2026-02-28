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

## Normalization and Data Statistics

### Computing Normalization Statistics

```@docs
GraphNetSim.data_minmax
GraphNetSim.data_meanstd
GraphNetSim.der_minmax
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
```

## Visualization

### VTK Export

```@docs
visualize
```