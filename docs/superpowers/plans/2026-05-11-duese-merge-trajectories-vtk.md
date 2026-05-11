# Duese: Merge Trajectories into VTK Time Series — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `data/Duese/merge_trajectories_vtk.jl` so it emits 12 `.vtkhdf` files (one per timestep) under `data/Duese/merged_vtk/`, each containing the merged point cloud of all 954 `batch2_clustered_traj*` trajectories with a per-point `TrajectoryID` field for color-coding in ParaView.

**Architecture:** Single standalone Julia script. Three functions plus a top-level driver call: (1) `discover_trajectories` enumerates trajectory dirs and parses indices; (2) `read_metadata` opens each h5 once to read `n_particles`/`trajectory_length` and builds `counts`/`offsets`/`traj_ids`; (3) `write_merged_vtkhdf` emits a single VTKHDF UnstructuredGrid file matching the layout used by `dictToVTKHDF` in [src/visualize.jl](../../../src/visualize.jl). The driver loops `t = 1..12`, reads `Points[t]` from every trajectory into a preallocated `3 × total_pts` matrix, and writes one file per timestep.

**Tech Stack:** Julia ≥1.10 (project env), `HDF5.jl` (already in `Project.toml` — required transitively by GraphNetSim and used directly by `src/visualize.jl`). No new dependencies.

**Spec:** [docs/superpowers/specs/2026-05-11-duese-merge-trajectories-vtk-design.md](../specs/2026-05-11-duese-merge-trajectories-vtk-design.md)

---

## File Structure

- **Create:** `data/Duese/merge_trajectories_vtk.jl` — the entire deliverable. ~120 lines containing imports, three named functions, and a top-level `merge_all(...)` call guarded by `if abspath(PROGRAM_FILE) == @__FILE__`.
- **No modifications** to `src/visualize.jl` or any other package file.
- **Output (at runtime, not committed):** `data/Duese/merged_vtk/merged_t01.vtkhdf` … `merged_t12.vtkhdf`.

The script mirrors the style of the existing `data/Duese/akima_batch2.jl` and `data/Duese/cluster_droplets.jl`: bare `using HDF5`, top-level functions, no module wrapper, hard-coded base dir resolved via `@__DIR__` so it works regardless of `pwd()`.

---

### Task 1: Discovery & metadata pass

**Files:**
- Create: `data/Duese/merge_trajectories_vtk.jl`

Adds the imports, the `BASE_DIR` / `OUT_DIR` constants (both resolved as absolute paths from `@__DIR__`), `discover_trajectories`, and `read_metadata`. No writer yet — this task is verified by calling the two functions interactively and inspecting the returned vectors.

- [ ] **Step 1: Create the script with the discovery + metadata functions**

Write the following to `data/Duese/merge_trajectories_vtk.jl`:

```julia
#
# Copyright (c) 2026 Josef Kircher
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
# Merge every Duese trajectory in data/Duese/batch2_clustered_traj*/ into a
# single VTK HDF5 time series under data/Duese/merged_vtk/. Each output file
# (one per timestep) contains the concatenated point cloud of all trajectories
# plus a per-point TrajectoryID field so overlapping particles can be colored
# by source trajectory in ParaView.

using HDF5
using Printf

const SCRIPT_DIR = @__DIR__                  # .../data/Duese
const BASE_DIR   = SCRIPT_DIR                # input root
const OUT_DIR    = joinpath(SCRIPT_DIR, "merged_vtk")
const TRAJ_REGEX = r"^batch2_clustered_traj(\d+)$"
const EXPECTED_T = 12                        # uniform trajectory_length

"""
    discover_trajectories(base_dir) -> Vector{Tuple{Int,String}}

Scan `base_dir` for `batch2_clustered_traj<N>` subdirectories that contain a
readable `datasets/train.h5`. Return a vector of `(N, abs_path_to_train_h5)`
pairs sorted by `N` ascending.
"""
function discover_trajectories(base_dir::AbstractString)
    out = Tuple{Int,String}[]
    for entry in readdir(base_dir)
        m = match(TRAJ_REGEX, entry)
        m === nothing && continue
        n = parse(Int, m.captures[1])
        h5path = joinpath(base_dir, entry, "datasets", "train.h5")
        isfile(h5path) || continue
        push!(out, (n, h5path))
    end
    sort!(out; by = first)
    return out
end

"""
    read_metadata(traj_paths; expected_T) -> (kept, counts, offsets, traj_ids, total_pts)

For each `(n, h5path)` in `traj_paths`, open the file once, read
`trajectory_1/n_particles` and `trajectory_1/trajectory_length`. Skip the
trajectory (with a printed warning) if `trajectory_length != expected_T`.

Returns:
- `kept :: Vector{Tuple{Int,String}}`   — surviving subset of `traj_paths`
- `counts :: Vector{Int}`               — `n_particles` per kept trajectory
- `offsets :: Vector{Int}`              — exclusive prefix sum of `counts`
- `traj_ids :: Vector{Int32}`           — length `total_pts`; entry j holds
                                          the trajectory index that owns the
                                          j-th merged point
- `total_pts :: Int`                    — `sum(counts)`
"""
function read_metadata(
    traj_paths::Vector{Tuple{Int,String}};
    expected_T::Integer = EXPECTED_T,
)
    kept = Tuple{Int,String}[]
    counts = Int[]
    for (n, path) in traj_paths
        n_particles, tl = h5open(path, "r") do fid
            g = fid["trajectory_1"]
            Int(read(g, "n_particles")), Int(read(g, "trajectory_length"))
        end
        if tl != expected_T
            @warn "Skipping trajectory $n: trajectory_length=$tl (expected $expected_T)" path
            continue
        end
        push!(kept, (n, path))
        push!(counts, n_particles)
    end
    offsets = Vector{Int}(undef, length(counts))
    acc = 0
    for i in eachindex(counts)
        offsets[i] = acc
        acc += counts[i]
    end
    total_pts = acc
    traj_ids = Vector{Int32}(undef, total_pts)
    for i in eachindex(kept)
        n = Int32(kept[i][1])
        @inbounds for j in (offsets[i] + 1):(offsets[i] + counts[i])
            traj_ids[j] = n
        end
    end
    return kept, counts, offsets, traj_ids, total_pts
end
```

- [ ] **Step 2: Smoke-test discovery and metadata by hand**

Run from the repo root:

```bash
julia --project -e '
include("data/Duese/merge_trajectories_vtk.jl")
trajs = discover_trajectories(BASE_DIR)
println("found ", length(trajs), " trajectories; first=", trajs[1], " last=", trajs[end])
kept, counts, offsets, traj_ids, total_pts = read_metadata(trajs)
println("kept=", length(kept), " total_pts=", total_pts,
        " ids[1]=", traj_ids[1],
        " ids[counts[1]]=", traj_ids[counts[1]],
        " ids[counts[1]+1]=", traj_ids[counts[1]+1])
'
```

Expected:
- `found 954 trajectories; first=(1, ".../batch2_clustered_traj1/datasets/train.h5") last=(954, ...)` — count must be 954 and ordering must be by `N` (not lexicographic, so 10 should come *after* 9, not after 1).
- `kept=954 total_pts=<some big number, expected order of tens of millions>`.
- `ids[1] == 1` (first point belongs to trajectory 1).
- `ids[counts[1]] == 1` (last point of trajectory 1 still tagged 1).
- `ids[counts[1]+1] == 2` (first point of trajectory 2 tagged 2).

If `found` is not 954, check whether some `batch*` dirs lack `datasets/train.h5` and adjust expectations rather than the code.

- [ ] **Step 3: Commit**

```bash
git add data/Duese/merge_trajectories_vtk.jl
git commit -m "$(cat <<'EOF'
feat(duese): discover trajectories and build metadata for VTK merge

Adds discover_trajectories and read_metadata to the new merge script.
discover_trajectories scans data/Duese for batch2_clustered_traj<N> dirs
and returns the train.h5 paths sorted numerically by N. read_metadata
opens each file once to read n_particles/trajectory_length, skips any
trajectory whose length deviates from the expected 12, and builds the
counts / offsets / per-point TrajectoryID vector used by the writer.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: VTKHDF writer

**Files:**
- Modify: `data/Duese/merge_trajectories_vtk.jl` (append a new function)

Adds `write_merged_vtkhdf(path, pts, traj_ids)` that emits a single VTKHDF v2.2 `UnstructuredGrid` file. Layout matches `dictToVTKHDF` in [src/visualize.jl:122-257](../../../src/visualize.jl#L122-L257) — same ASCII `Type` attribute pattern, same `H5T_STD_U8LE` `Types` dataset — but writes one merged cloud instead of one cloud per trajectory.

- [ ] **Step 1: Append `write_merged_vtkhdf` to the script**

Append to `data/Duese/merge_trajectories_vtk.jl`:

```julia
"""
    write_merged_vtkhdf(path, pts, traj_ids)

Write a VTK HDF5 v2.2 UnstructuredGrid file at `path` containing
`size(pts, 2)` points (each a VTK_VERTEX cell) with a `PointData/TrajectoryID`
Int32 field. `pts` must be a `3 × N` Float32 matrix; `traj_ids` must be a
length-`N` Int32 vector.
"""
function write_merged_vtkhdf(
    path::AbstractString,
    pts::AbstractMatrix{Float32},
    traj_ids::AbstractVector{Int32},
)
    @assert size(pts, 1) == 3 "pts must be 3 x N"
    @assert size(pts, 2) == length(traj_ids) "pts columns must match traj_ids length"
    n = size(pts, 2)
    HDF5.h5open(path, "w") do fid
        top = HDF5.create_group(fid, "VTKHDF")
        HDF5.attributes(top)["Version"] = [2, 2]

        # "Type" attribute must be ASCII-encoded, matching dictToVTKHDF.
        let s = "UnstructuredGrid"
            dtype = HDF5.datatype(s)
            HDF5.API.h5t_set_cset(dtype.id, HDF5.API.H5T_CSET_ASCII)
            dspace = HDF5.dataspace(s)
            attr = HDF5.create_attribute(top, "Type", dtype, dspace)
            HDF5.write_attribute(attr, dtype, s)
        end

        top["NumberOfPoints"]          = [n]
        top["NumberOfCells"]           = [n]
        top["NumberOfConnectivityIds"] = [n]
        top["Connectivity"]            = collect(0:(n - 1))
        top["Offsets"]                 = collect(0:n)

        # Types: one VTK_VERTEX (=1) per point, encoded as UInt8.
        type_data = Int8.(ones(n))
        dt = HDF5.API.h5t_copy(HDF5.API.H5T_STD_U8LE)
        dset = HDF5.create_dataset(top, "Types", HDF5.Datatype(dt), HDF5.dataspace(type_data))
        HDF5.write(dset, type_data)

        top["Points"] = pts

        point_data = HDF5.create_group(top, "PointData")
        point_data["TrajectoryID"] = traj_ids
    end
    return path
end
```

- [ ] **Step 2: Smoke-test the writer with a synthetic 4-point file**

Run from the repo root:

```bash
julia --project -e '
include("data/Duese/merge_trajectories_vtk.jl")
tmp = tempname() * ".vtkhdf"
pts = Float32[0 1 2 3; 0 0 0 0; 0 0 0 0]
ids = Int32[1, 1, 2, 2]
write_merged_vtkhdf(tmp, pts, ids)
println("wrote ", tmp, " size=", filesize(tmp))

# Verify structure with HDF5 directly.
HDF5.h5open(tmp, "r") do fid
    top = fid["VTKHDF"]
    println("Version           = ", read(HDF5.attributes(top)["Version"]))
    println("Type              = ", read(HDF5.attributes(top)["Type"]))
    println("NumberOfPoints    = ", read(top, "NumberOfPoints"))
    println("NumberOfCells     = ", read(top, "NumberOfCells"))
    println("Connectivity      = ", read(top, "Connectivity"))
    println("Offsets           = ", read(top, "Offsets"))
    println("Types             = ", read(top, "Types"))
    println("Points            = ", read(top, "Points"))
    println("TrajectoryID      = ", read(top["PointData"], "TrajectoryID"))
end
rm(tmp)
'
```

Expected stdout (exact values):
- `Version           = [2, 2]`
- `Type              = UnstructuredGrid`
- `NumberOfPoints    = [4]`
- `NumberOfCells     = [4]`
- `Connectivity      = [0, 1, 2, 3]`
- `Offsets           = [0, 1, 2, 3, 4]`
- `Types             = Int8[1, 1, 1, 1]` (or `UInt8[1,1,1,1]` — both are acceptable per the H5T_STD_U8LE storage type)
- `Points            = Float32[0.0 1.0 2.0 3.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0]`
- `TrajectoryID      = Int32[1, 1, 2, 2]`

If `Type` reads back as bytes/garbage instead of the string `"UnstructuredGrid"`, the ASCII charset attribute path is broken — fix by re-checking the `let s = ...` block matches `dictToVTKHDF` byte-for-byte.

- [ ] **Step 3: Commit**

```bash
git add data/Duese/merge_trajectories_vtk.jl
git commit -m "$(cat <<'EOF'
feat(duese): write_merged_vtkhdf emits one VTKHDF UnstructuredGrid file

Adds the per-file writer used by the merge driver: one VTK_VERTEX cell
per point, ASCII Type="UnstructuredGrid" attribute, and a
PointData/TrajectoryID Int32 field. Layout matches dictToVTKHDF in
src/visualize.jl, just merged into a single cloud instead of one cloud
per trajectory.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Merge driver + entry point + full run

**Files:**
- Modify: `data/Duese/merge_trajectories_vtk.jl` (append driver + entry guard)

Adds `merge_all(base_dir, out_dir)` which: discovers trajectories, calls `read_metadata`, allocates a `3 × total_pts` Float32 buffer, and for each `t ∈ 1:EXPECTED_T` reads `Points[t]` from each trajectory into the appropriate slice, then writes `merged_t<NN>.vtkhdf`. A bottom-of-file `if abspath(PROGRAM_FILE) == @__FILE__` guard calls `merge_all(BASE_DIR, OUT_DIR)` so the script runs when invoked directly but is silent when `include`'d (which Tasks 1 and 2 already relied on).

- [ ] **Step 1: Append the driver and entry guard**

Append to `data/Duese/merge_trajectories_vtk.jl`:

```julia
"""
    merge_all(base_dir, out_dir; expected_T=EXPECTED_T)

End-to-end driver: discover trajectories under `base_dir`, allocate one
`3 × total_pts` Float32 buffer, and for each timestep `t in 1:expected_T`
fill the buffer with `Points[t]` from every trajectory (in trajectory-index
order) and write the merged cloud to
`out_dir/merged_t<lpad(t,2,'0')>.vtkhdf`.
"""
function merge_all(
    base_dir::AbstractString,
    out_dir::AbstractString;
    expected_T::Integer = EXPECTED_T,
)
    isdir(out_dir) || mkpath(out_dir)

    trajs = discover_trajectories(base_dir)
    isempty(trajs) && error("No trajectories found under $base_dir")
    kept, counts, offsets, traj_ids, total_pts = read_metadata(
        trajs; expected_T = expected_T
    )
    @info "Merge ready" n_kept=length(kept) total_pts=total_pts out_dir=out_dir

    pts = Matrix{Float32}(undef, 3, total_pts)
    for t in 1:expected_T
        for (i, (_, path)) in enumerate(kept)
            ci = counts[i]
            oi = offsets[i]
            HDF5.h5open(path, "r") do fid
                g = fid["trajectory_1"]
                dset = g["Points[$t]"]
                # Read directly into the preallocated slice. HDF5.jl indexed
                # read returns a fresh array, so copy into the view.
                slab = read(dset)
                @assert size(slab) == (3, ci) "trajectory $(kept[i][1]) Points[$t] has size $(size(slab)), expected (3, $ci)"
                @inbounds pts[:, (oi + 1):(oi + ci)] = slab
            end
        end
        out_path = joinpath(out_dir, @sprintf("merged_t%02d.vtkhdf", t))
        write_merged_vtkhdf(out_path, pts, traj_ids)
        @info @sprintf("[%2d/%d] wrote %s (%d pts)", t, expected_T, basename(out_path), total_pts)
    end
    return out_dir
end

if abspath(PROGRAM_FILE) == @__FILE__
    merge_all(BASE_DIR, OUT_DIR)
end
```

- [ ] **Step 2: Run end-to-end**

From the repo root:

```bash
julia --project data/Duese/merge_trajectories_vtk.jl
```

Expected:
- `[ Info: Merge ready n_kept=954 total_pts=<N>` printed first.
- Twelve `[ Info: [tt/12] wrote merged_tNN.vtkhdf (<N> pts)` lines, one per timestep.
- No warnings other than possible `Skipping trajectory <n>` lines from `read_metadata` (none expected with the current dataset).
- Runtime: order of minutes (≈11 500 HDF5 opens + ~360 M Float32 reads + 12 file writes); on a modern SSD expect 1–5 minutes.

If the run dies with `OutOfMemoryError`, the design assumes ~480 MB working set — verify nothing else is consuming RAM and that `total_pts` is in the expected tens-of-millions range (not billions).

- [ ] **Step 3: Verify the output directory and a sample file**

```bash
ls -1 data/Duese/merged_vtk/
```

Expected exactly:
```
merged_t01.vtkhdf
merged_t02.vtkhdf
merged_t03.vtkhdf
merged_t04.vtkhdf
merged_t05.vtkhdf
merged_t06.vtkhdf
merged_t07.vtkhdf
merged_t08.vtkhdf
merged_t09.vtkhdf
merged_t10.vtkhdf
merged_t11.vtkhdf
merged_t12.vtkhdf
```

Verify file structure on `merged_t01.vtkhdf`:

```bash
julia --project -e '
using HDF5
HDF5.h5open("data/Duese/merged_vtk/merged_t01.vtkhdf", "r") do fid
    top = fid["VTKHDF"]
    println("Type           = ", read(HDF5.attributes(top)["Type"]))
    println("NumberOfPoints = ", read(top, "NumberOfPoints"))
    println("Points size    = ", size(top["Points"]))
    println("TrajectoryID   length = ", length(top["PointData"]["TrajectoryID"]))
    println("TrajectoryID[1]              = ", read(top["PointData"]["TrajectoryID"])[1])
    println("max TrajectoryID             = ", maximum(read(top["PointData"]["TrajectoryID"])))
end
'
```

Expected:
- `Type           = UnstructuredGrid`
- `NumberOfPoints` length 1 and equal to the `total_pts` printed by the run.
- `Points size    = (3, <total_pts>)`.
- `TrajectoryID   length = <total_pts>`.
- `TrajectoryID[1] = 1` (first point owned by trajectory 1).
- `max TrajectoryID = 954` (or whatever the largest discovered trajectory index was).

Cross-check that `Points[:, 1:counts[1]]` in the first merged file equals `Points[1]` from `batch2_clustered_traj1/datasets/train.h5`:

```bash
julia --project -e '
using HDF5
src = HDF5.h5open("data/Duese/batch2_clustered_traj1/datasets/train.h5", "r") do f
    read(f["trajectory_1"], "Points[1]")
end
dst = HDF5.h5open("data/Duese/merged_vtk/merged_t01.vtkhdf", "r") do f
    read(f["VTKHDF"]["Points"])
end
n = size(src, 2)
println("first-traj slice matches: ", src == dst[:, 1:n])
'
```

Expected: `first-traj slice matches: true`. If false, the offset bookkeeping is off.

- [ ] **Step 4: Commit**

```bash
git add data/Duese/merge_trajectories_vtk.jl
git commit -m "$(cat <<'EOF'
feat(duese): merge_all driver writes 12-file VTK time series

Adds merge_all(base_dir, out_dir): one preallocated 3xN buffer reused
across timesteps; per timestep, fills the buffer with Points[t] from
each trajectory (in trajectory-index order) and writes
merged_vtk/merged_t<NN>.vtkhdf via write_merged_vtkhdf. Script entry
guard runs merge_all on the standard Duese paths when invoked as
`julia --project data/Duese/merge_trajectories_vtk.jl`.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

**1. Spec coverage:**
- Input discovery of 954 trajs from `batch2_clustered_traj*/datasets/train.h5` → Task 1.
- One file per timestep, 12 files, named `merged_t<NN>.vtkhdf` under `data/Duese/merged_vtk/` → Task 3 driver + verification.
- VTKHDF v2.2 UnstructuredGrid layout (`Connectivity`, `Offsets`, `Types`=VTK_VERTEX, ASCII `Type` attr) mirroring `dictToVTKHDF` → Task 2.
- Per-point `TrajectoryID` Int32 in `PointData` → Task 2 writer signature + Task 1 metadata.
- First-pass metadata: `counts`, `offsets`, `traj_ids`, `total_pts`, skip-with-warning on `trajectory_length != 12` → Task 1 `read_metadata`.
- Per-timestep emit with preallocated `pts` matrix → Task 3 driver.
- Memory budget (~480 MB transient) → respected by single shared `pts` buffer in Task 3.
- Script location `data/Duese/merge_trajectories_vtk.jl`, no package API change → Tasks 1–3 all touch only that one file.
- Invokable as `julia --project data/Duese/merge_trajectories_vtk.jl` → Task 3 entry guard.
- Verification (12 files; `h5ls`-equivalent structure check; first-trajectory slice equality) → Task 3 Step 3.

The spec's "non-goals" (no vel/acc/type fields, no transient single-file, no CLI flags) are honored — none of the tasks add those.

**2. Placeholder scan:** No "TBD", no "implement later", no "similar to Task N", no abstract "handle edge cases". Every code-bearing step contains the full code block. Verification steps name exact expected values (4-point synthetic test, `TrajectoryID[1] == 1`, `max == 954`, etc.).

**3. Type consistency:**
- `discover_trajectories` returns `Vector{Tuple{Int,String}}` — used in `read_metadata` signature, used in `merge_all` (`for (i, (_, path)) in enumerate(kept)`).
- `read_metadata` returns `(kept, counts, offsets, traj_ids, total_pts)` — same 5-tuple consumed in `merge_all` Step 1.
- `traj_ids :: Vector{Int32}` declared in Task 1, consumed by `write_merged_vtkhdf(path, pts, traj_ids::AbstractVector{Int32})` in Task 2.
- `pts :: Matrix{Float32}` declared in Task 3 driver, consumed by `write_merged_vtkhdf(path, pts::AbstractMatrix{Float32}, ...)` in Task 2.
- `EXPECTED_T` constant referenced by both `read_metadata`'s default kwarg and `merge_all`'s loop bound.
- All file paths spelled out: `data/Duese/batch2_clustered_traj<N>/datasets/train.h5` (input), `data/Duese/merged_vtk/merged_t<NN>.vtkhdf` (output).

No inconsistencies found.
