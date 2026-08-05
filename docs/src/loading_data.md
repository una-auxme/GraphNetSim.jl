# Loading Data

There are two independent questions when getting data into GraphNetSim.jl:

1. **What on-disk format is the data in?** — HDF5, JLD2, or a CSV that you convert first.
2. **How do you hand it to the trainer?** — point `train_network` at a directory, or build a
   [`Dataset`](@ref GraphNetSim.Dataset) yourself.

The [`meta.json`](#The-meta.json-schema) file is the glue: it describes *where* each feature lives
inside the data file and *how* to interpret it, so the same loader adapts to many different data
layouts without reshaping the raw arrays.

## 1. Point `train_network` at a dataset directory (standard path)

The usual way. Organize the dataset as a directory containing a `meta.json` and one data file per
split, then pass the directory to [`train_network`](@ref GraphNetSim.train_network):

```
dataset/
├── meta.json     # feature specs, normalization stats, node types, topology
├── train.h5      # training trajectories   (.jld2 also accepted)
├── valid.h5      # validation trajectories
└── test.h5       # test trajectories
```

```julia
train_network(Optimisers.Adam(1f-4), "./dataset", "./checkpoints"; steps=50_000)
```

Internally this constructs `Dataset(:train, ds_path, args)` and `Dataset(:valid, ds_path, args)`
for you. [`eval_network`](@ref GraphNetSim.eval_network) does the same for the `:test` split.

## 2. Construct a `Dataset` directly

For custom loops, non-standard filenames, or inspecting data, build the [`Dataset`](@ref
GraphNetSim.Dataset) yourself. Two constructors are available.

**By split + directory** — expects `meta.json` plus `train`/`valid`/`test` in the directory (this is
what `train_network` calls under the hood):

```julia
args = GraphNetSim.Args(; training_strategy=DerivativeTraining())
ds   = GraphNetSim.Dataset(:train, "./dataset", args)
```

**By explicit file paths** — use any filenames/locations you like for the data and metadata files:

```julia
ds = GraphNetSim.Dataset("./somewhere/run42.h5", "./somewhere/spec.json", args)
```

See [`GraphNetSim.get_file`](@ref) for the split-name → filename lookup rules.

## Supported file formats: HDF5 and JLD2

Both `.h5` (via HDF5.jl) and `.jld2` (via JLD2.jl) are accepted interchangeably for every data
file. When a split is loaded by symbol, [`get_file`](@ref GraphNetSim.get_file) prefers `.jld2`
and falls back to `.h5`. In both formats each **top-level group/key is one trajectory**
(enumerated by [`keystraj`](@ref GraphNetSim.keystraj)); the per-feature datasets live inside each
trajectory group.

## The `meta.json` schema

`meta.json` drives all loading and normalization decisions. Each entry under `"features"` names
the dataset key inside the data file and how to read it:

```json
"features": {
  "position": {
    "key": "pos[$t]",   "type": "dynamic", "dtype": "float32", "dim": 2,
    "data_mean": [...],  "data_std": [...]
  },
  "node_type": {
    "key": "type",      "type": "static",  "dtype": "int32",   "dim": 1,
    "onehot": true,      "data_min": 1,     "data_max": 2
  }
}
```

Key points that make the loader flexible:

- **`key`** is the dataset name inside each trajectory group. For `"type": "dynamic"` features the
  literal `$t` is substituted with the 1-based timestep, so per-timestep arrays like `pos[1]`,
  `pos[2]`, … are stitched into a time series. `"type": "static"` features (e.g. `node_type`) are
  read once and broadcast across time.
- **Normalization is selected per feature** from the stats present: `data_min`/`data_max` →
  `NormaliserOfflineMinMax`, `data_mean`/`data_std` → `NormaliserOfflineMeanStd`, and if neither is
  given the feature uses `NormaliserOnline` (stats accumulated over the first `norm_steps` steps).
- **Trajectory metadata can be literal or a datafile key.** `dt`, `trajectory_length`,
  `n_particles`, and `dims` may each be given as a fixed value **or** as a string naming a dataset
  inside the trajectory group, so ragged datasets (varying length/particle count per trajectory)
  are supported. Use `trajectory_length = -1` together with a `dims_key` to infer length/shape
  from the file.

The remaining top-level fields (`feature_names`, `input_features`, `output_features`,
`derivative_target_features`, `default_connectivity_radius`, `bounds`, …) select which features are
inputs, targets, and how the graph is built.

## Importing from CSV

If your simulation exports particle trajectories as CSV (e.g. a ParaView/SPH export with per-row
particles and `Points:0`, `Vel:0`, `Type`, … columns), convert it to an HDF5 file with
[`csv_to_hdf5`](@ref GraphNetSim.csv_to_hdf5). It groups rows by particle id, selects the requested
spatial dimensions, and **computes accelerations** from velocity/position using a choice of
difference or interpolation schemes (`pchip`, `central_diff`, `cubic_spline`, …):

```julia
using GraphNetSim

# 3D, PCHIP-interpolated accelerations
csv_to_hdf5("data/dam_break.csv", "data/train.h5";
            dt=0.01, dims=[1, 2, 3], interpolation_scheme="pchip")

# 2D (skip the y-dimension), copy extra per-particle fields through
csv_to_hdf5("data/input.csv", "data/train.h5";
            dims=[1, 3], interpolation_scheme="cubic_spline",
            extra_fields=[:Mass, :Pressure])
```

The result is written in the timestep-based layout (`pos[$t]`, `vel[$t]`, `acc[$t]`, plus
`type`, `dt`, `n_particles`, `trajectory_length`) that the `meta.json` keys above expect. Write a
`meta.json` alongside the converted `train.h5`/`valid.h5`/`test.h5` and you are back on the standard
path.

## Importing from VTK (ParaView / SPH)

Particle simulators (DualSPHysics, ParaView exports, …) usually emit VTK time
series: a `.pvd` collection referencing per-timestep `.pvtu`/`.vtu` pieces.
[`vtk_to_hdf5`](@ref GraphNetSim.vtk_to_hdf5) converts these into the same HDF5
layout as `csv_to_hdf5` — one trajectory group per source:

```julia
using GraphNetSim

vtk_to_hdf5(
    "sim/OUTPUT/particles.pvd",     # a .pvd, a directory, a .pvtu/.vtu, a legacy .vtk, or a vector of these
    "dataset/train.h5";
    dims          = [1, 2, 3],
    velocity_field = "Vel",         # PointData arrays; auto-detected from aliases if absent
    type_field     = "phaseID",     # → node_type (remapped to 1…k)
    id_field       = "prtlID",      # particles are sorted by this to stay row-aligned over time
    acc_field      = "Acc",         # read acceleration from the file …
    write_meta     = true,          # … or emit a computed skeleton meta.json (Tier B)
)
```

Implementation notes that matter in practice:

- **Wrapper formats are parsed internally.** `ReadVTK.jl` cannot open `.pvd`
  (`Collection`) or `.pvtu` (`PUnstructuredGrid`) files, so `vtk_to_hdf5` reads
  those XML wrappers itself and hands only the leaf `.vtu` pieces to ReadVTK for
  the raw/zlib/appended binary decode.
- **Legacy `.vtk` is supported too.** ReadVTK handles only the XML/VTKHDF
  formats, so a built-in reader decodes legacy BINARY `POLYDATA` files (e.g.
  DualSPHysics `PartAll_*.vtk`). Point a `vtk_to_hdf5` call at a directory of
  such files (sorted lexicographically = time order) or a single `.vtk`. Since
  these carry no acceleration, it is recomputed automatically.
- **Empty frames are skipped.** Trailing 0-point frames (common once particles
  leave the domain) are detected from the leaf header and dropped before ReadVTK
  is called (it otherwise throws on empty compressed blocks). The trajectory is
  truncated to the contiguous non-empty prefix.
- **`recompute_acc=true`** ignores the file's acceleration and derives it from
  velocity/position via `interpolation_scheme` (reusing the `csv_to_hdf5`
  machinery). Useful when the stored `Acc` is a raw solver force rather than the
  kinematic acceleration the ODE integrates — for the WashTec SPH data the file
  `Acc` has magnitudes ~10⁴ while the recomputed kinematic acceleration is O(1).
- **Field names vary between codes.** `velocity_field`/`type_field`/`id_field`/
  `acc_field` are matched case-insensitively, falling back to a small alias table
  (`Velocity`/`v`, `Type`/`Mk`, `Idp`/`id`, …). An unresolved *required* field
  errors and lists the arrays actually present so you can map it explicitly.
- **Variable particle counts are rejected** (for now): if the count changes
  between kept frames the converter errors — trim the source to a constant-count
  window.

### The `meta.json` for a converted dataset

A full training dataset still needs a `meta.json`. Two options:

- **Provide your own** — pass `meta = "path/to/meta.json"`; it is validated
  against the written HDF5 and copied next to the output.
- **Auto-skeleton** — pass `write_meta = true` to emit a `meta.json` with every
  derivable field filled in (feature keys/types/dims, `dims`, `bounds`, node-type
  range, and computed mean/std normalization stats). Review the feature *roles*
  (`input_features`/`output_features`/…) and especially
  `default_connectivity_radius`, which is only *estimated* from mean particle
  spacing and must be set correctly for real training.

## MLUtils integration

[`Dataset`](@ref GraphNetSim.Dataset) implements the MLUtils data-container interface
([`numobs`](https://juliaml.github.io/MLUtils.jl/stable/api/#MLUtils.numobs) /
[`getobs!`](@ref GraphNetSim.MLUtils.getobs!)), where **one observation is one trajectory**. This is
how the training loop batches and shuffles trajectories, and it means you can drop a `Dataset` into
a standard `MLUtils.DataLoader` in your own code:

```julia
using MLUtils
loader = MLUtils.DataLoader(ds; batchsize=1, shuffle=true)
for traj in loader
    # traj is a Dict of feature arrays on the configured device
end
```

See the [API reference](api.md) for full signatures of every function mentioned here.
