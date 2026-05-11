# Duese: merge per-trajectory HDF5 datasets into a single VTK time series

**Status:** approved
**Date:** 2026-05-11

## Goal

Produce one VTK HDF5 file *per timestep* containing the merged point cloud of
**all** Duese trajectories under `data/Duese/batch2_clustered_traj*/`, with a
per-point `TrajectoryID` field so overlapping particles can be colored by
source trajectory in ParaView.

## Input

- 954 directories: `data/Duese/batch2_clustered_traj<N>/datasets/train.h5`
  (train/valid/test in each dir are byte-identical; train is canonical).
- Each `train.h5` contains a single group `trajectory_1` with:
  - `Points[t]`, `Vel[t]`, `Acc[t]` for `t = 1..trajectory_length` —
    Float32 arrays of shape `(3, n_particles)`.
  - `n_particles :: Int32`, `trajectory_length :: Int32`, `dt :: Float32`,
    `type :: (n_particles,) Int32`.
- `trajectory_length = 12` and `dt` are uniform across all trajectories
  (verified for the sampled subset; verified at runtime per file).
- `n_particles` varies per trajectory (observed: ~9k to ~46k).

## Output

- Directory: `data/Duese/merged_vtk/` (created if missing).
- Files: `merged_t{NN}.vtkhdf` for `NN = 01..12`. ParaView opens the directory
  as a 12-step time series.
- Format: `VTKHDF` v2.2 `UnstructuredGrid`, mirroring the layout produced by
  `dictToVTKHDF` in [src/visualize.jl](../../../src/visualize.jl):
  - ASCII attribute `Type = "UnstructuredGrid"`.
  - `NumberOfPoints`, `NumberOfCells`, `NumberOfConnectivityIds` all equal to
    `N_total = Σ n_particles[i]`.
  - `Connectivity = 0:N_total-1`, `Offsets = 0:N_total`.
  - `Types` = vector of `UInt8(1)` (VTK_VERTEX), one cell per point, written
    with `H5T_STD_U8LE`.
  - `Points` = `3 × N_total` Float32 matrix, concatenated in trajectory-index
    order.
  - `PointData/TrajectoryID` = length-`N_total` Int32 vector, value = parsed
    trajectory index `N`.

## Algorithm

1. **Discover.** `readdir("data/Duese")` → filter entries matching
   `batch2_clustered_traj<N>`, parse `N`, sort ascending.
2. **First pass (metadata).** For each dir, open `datasets/train.h5`, read
   `trajectory_1/n_particles` and `trajectory_1/trajectory_length`, close.
   Skip the entire trajectory (don't include in `counts`/`offsets`/`traj_ids`)
   with a warning if `trajectory_length != 12`. Build:
   - `counts :: Vector{Int}` (one entry per kept trajectory)
   - `offsets :: Vector{Int}` (exclusive prefix sum of `counts`)
   - `total_pts :: Int`
   - `traj_ids :: Vector{Int32}` of length `total_pts`, filled with the
     parsed trajectory index.
3. **Per-timestep emit.** For `t = 1..12`:
   - Allocate `pts = Matrix{Float32}(undef, 3, total_pts)`.
   - For each kept trajectory `i`: open its h5, read `Points[t]` directly
     into the slice `pts[:, offsets[i]+1 : offsets[i]+counts[i]]`, close.
   - Write `merged_vtk/merged_t$(lpad(t,2,'0')).vtkhdf` using a small local
     writer (see "Writer" below).
   - Free `pts` (`GC.gc()` not required — Julia handles it).
   - Print one progress line: `[t/12] wrote merged_tNN.vtkhdf (N_total pts)`.

## Writer

A local function `write_merged_vtkhdf(path, pts, traj_ids)` that emits exactly
the structure listed in **Output** above, copying the ASCII-attribute pattern
and `H5T_STD_U8LE` `Types` dataset from `dictToVTKHDF`. Not exposed via the
package API — defined inline in the script.

## Memory budget

Per timestep, peak transient allocation is the merged `pts` matrix:
`3 × N_total × 4 B` ≈ 360 MB for `N_total ≈ 30M`. The `traj_ids` Int32 vector
(`~120 MB`) is allocated once and reused across all 12 outputs. Output `.vtkhdf`
files are ~480 MB each (`pts + traj_ids + headers`); 12 files ≈ 5.7 GB on disk.

## Location & invocation

- Script: `data/Duese/merge_trajectories_vtk.jl`.
- Not added to package API or `src/visualize.jl`.
- Run: `julia --project data/Duese/merge_trajectories_vtk.jl`.
- Hard-coded paths relative to the repo root (`data/Duese` in / `data/Duese/merged_vtk` out);
  if invoked from elsewhere, the script `cd`s to the repo root first via
  the script's own `@__DIR__` (script lives in `data/Duese/`, so
  repo root = `joinpath(@__DIR__, "..", "..")`).

## Non-goals

- No velocity / acceleration / type fields in the output. Adding them later
  is a one-line change in the per-timestep loop.
- No transient single-file VTKHDF (Steps extension) — explicitly rejected
  during brainstorming for compatibility / writer simplicity.
- No CLI flags or subset selection — explicitly rejected; all 954 every run.

## Verification

After running the script, manual verification:

1. `ls data/Duese/merged_vtk/` shows exactly 12 files named
   `merged_t01.vtkhdf` … `merged_t12.vtkhdf`.
2. `h5ls -r data/Duese/merged_vtk/merged_t01.vtkhdf` lists
   `VTKHDF/{Points,Connectivity,Offsets,Types,NumberOfPoints,
   NumberOfCells,NumberOfConnectivityIds,PointData/TrajectoryID}`.
3. `NumberOfPoints[1]` matches `sum(counts)` from the first pass (i.e.
   the total over kept trajectories, after any skip-warnings).
4. Opening one file in ParaView and coloring by `TrajectoryID` shows
   distinct colors per source trajectory.
