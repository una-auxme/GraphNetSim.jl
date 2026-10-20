#
# Copyright (c) 2026 Josef Jouaux, Julian Trommer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
# This file contains work derived from DeepMind's "learning_to_simulate"
# (https://github.com/google-deepmind/deepmind-research), modified from the original:
#
#   Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# See THIRD_PARTY_NOTICES.md for details.
#

using CUDA
import Statistics: norm
using JLD2
using PointNeighbors
using Octopus: Octopus
using ChainRulesCore

"""
    build_graph(gns::GraphNetCore.GraphNetwork, data::Dict{String,Any}, datapoint::Integer, meta, node_type, device)

Construct a [FeatureGraph](https://una-auxme.github.io/MeshGraphNets.jl/dev/graph_net_core/#GraphNetCore.FeatureGraph) from trajectory data at a specific time step.

Extracts position and velocity data from the trajectory dictionary at the given time point,
then delegates to the second method to construct the graph with edge connectivity and normalized features.

## Arguments
- `gns::GraphNetCore.GraphNetwork`: Graph network model containing normalizers for features.
- `data::Dict{String,Any}`: Dictionary containing trajectory data (position, velocity, etc.).
- `datapoint::Integer`: Time step index to extract from the trajectory.
- `meta::Dict{String,Any}`: Metadata dictionary with connectivity and feature settings.
- `node_type`: One-hot encoded node type features.
- `device::Function`: Device placement function (cpu or gpu).

## Returns
- `GraphNetCore.FeatureGraph`: Constructed graph with normalized node and edge features.
"""
function build_graph(
    gns::GraphNetCore.GraphNetwork,
    data::Dict{String,Any},
    datapoint::Integer,
    meta,
    node_type,
    device,
)

    # fluid_position = data["position"][:, data["mask"], datapoint]
    current_position = data["position"][:, :, datapoint]

    velocity = data["velocity"][:, :, datapoint]

    build_graph(gns, current_position, velocity, meta, node_type, data["mask"], device)
end

# Materialise `position` into a concrete, device-native matrix for neighbor search.
#
# In `build_graph`, `position === x.x` is a view into the ODE-state ComponentArray
# (on CPU a `ReshapedArray` over a 1-D `SubArray`). Neighbor search dispatches on the
# concrete array type (`point_neighbor_ns(::Array)` vs `(::CuArray)`), and the
# SingleShooting/MultipleShooting backward pass differentiates through this call, so the
# result must be (a) a concrete `Array`/`CuArray` and (b) produced by a
# Zygote-differentiable op. On GPU, `device(x)` yields a `CuArray` (unchanged from the
# original code). On CPU, `collect(x)` yields a dense `Array` via a differentiable copy —
# unlike `adapt_structure(::CPUDevice, ::SubArray)`, whose `SubArray` constructor has no
# adjoint (`Need an adjoint for constructor SubArray`).
_neighbor_positions(device, x) = device(x)
_neighbor_positions(::CPUDevice, x) = collect(x)

"""
    build_graph(gns::GraphNetCore.GraphNetwork, position, velocity, meta, node_type, mask, device)

Construct a [FeatureGraph](https://una-auxme.github.io/MeshGraphNets.jl/dev/graph_net_core/#GraphNetCore.FeatureGraph) from position and velocity data with edge connectivity.

Computes edges based on spatial proximity using GPU-accelerated neighborhood search, calculates relative displacements
and normalized distances. Node features are constructed from position, velocity, node type, and distance bounds to domain boundaries.
All features are normalized using the normalizers stored in the model.

## Arguments
- `gns::GraphNetCore.GraphNetwork`: Graph network model containing normalizers.
- `position::AbstractArray`: Particle positions with shape (dims, n_particles).
- `velocity::AbstractArray`: Particle velocities with shape (dims, n_particles).
- `meta::Dict{String,Any}`: Metadata with default_connectivity_radius, bounds, dims, input_features, and device settings.
- `node_type::AbstractArray`: One-hot encoded node type features.
- `mask::AbstractVector`: Indices of particles to include in the graph (fluid particles).
- `device::Function`: Device placement function.

## Returns
- `GraphNetCore.FeatureGraph`: Graph with normalized node features, normalized edge features, sender and receiver indices.
"""
function build_graph(
    gns::GraphNetCore.GraphNetwork, position, velocity, meta, node_type, mask, device
) # TODO check ODE solve and if this is really repeatedly done
    senders, receivers, rel_displacement, rel_dist_norm = neighbor_search(
        _neighbor_positions(device, position),
        Float32(meta["default_connectivity_radius"]),
        get(meta, "neighbor_backend", :pointneighbors),
    )

    multi_type = n_node_types(meta) > 1
    # Wall distance is driven purely by the presence of a domain box in the dataset meta
    # (`meta["bounds"]`) — independent of the node-type count and of `input_features`. A
    # bounded dataset gives every particle a distance-to-walls block; an unbounded one
    # (no `bounds`) yields a correspondingly shorter input vector. `multi_type` only
    # controls the `node_type` one-hot block below.
    use_wall = haskey(meta, "bounds")
    use_position = "position" in meta["input_features"]

    vel_norm = gns.n_norm["velocity"](velocity)

    edge_features = device(vcat(rel_displacement, rel_dist_norm) .+ 1.0f-8)

    # Node features are the vertical concatenation of, in order:
    #   [position?, velocity, wall_distance?, node_type?]
    # Build this as a single `vcat` over the present blocks rather than chaining one
    # `vcat` per block. Chaining re-allocates and re-copies the growing feature matrix
    # at every link — it allocated 1.7-2.1x the final array in transient GPU garbage,
    # whereas a single `vcat` allocates exactly the output once (~4.8x faster
    # forward, ~1/3 less total fwd+bwd allocation; values/gradients bit-identical).
    # The block tuple uses immutable splats (no `push!`) so the expression stays
    # Zygote-differentiable on the training RHS; a lone present block returns untouched.
    nf_blocks = (
        (use_position ? (gns.n_norm["position"](position),) : ())...,
        vel_norm,
        (use_wall ? (_wall_distance(position, meta, device),) : ())...,
        (multi_type ? (node_type,) : ())...,
    )
    nf = length(nf_blocks) == 1 ? nf_blocks[1] : vcat(nf_blocks...)
    node_features = device(nf)

    return GraphNetCore.FeatureGraph(
        node_features, gns.e_norm(edge_features), senders, receivers
    )
end

# Clipped per-particle distance to the domain box `meta["bounds"]`, following
# DeepMind's "distance to walls" node feature. This depends only on `meta["bounds"]`
# and positions — NOT on node types or on the presence of boundary particles — so it
# is computed for every particle whenever the feature is enabled. Returns `2 * dims`
# rows (low + high bound per spatial dimension), matching the width reserved by
# `calc_norms`. An implicit domain box with no boundary particles is fully supported;
# the only hard requirement is that `meta["bounds"]` is defined.
function _wall_distance(position, meta, device)
    haskey(meta, "bounds") || throw(
        ArgumentError("wall_distance requires meta[\"bounds\"] to be defined."),
    )
    boundaries = device(Float32.(vcat(permutedims.(meta["bounds"])...)))
    dist_low_bound = position .- boundaries[:, 1]
    dist_up_bound = boundaries[:, 2] .- position
    return clamp.(
        vcat(dist_low_bound, dist_up_bound) ./ Float32(meta["default_connectivity_radius"]),
        -1.0f0,
        1.0f0,
    )
end

"""
    check_and_delete_filtered(arr1, arr2, value, boundary_last::Bool=true)

Filter edge pairs based on sender indices and optionally reorganize indices.

Uses value as a threshold to split sender indices (typically separating fluid particles from boundary particles).
Creates a mask to keep edges based on the receiver filtering condition, then renumbers indices to account for removed particles.

## Arguments
- `arr1`: Sender indices array.
- `arr2`: Receiver indices array.
- `value`: Threshold index for splitting particles (typically n_fluid_particles).
- `boundary_last::Bool=true`: If true, keeps receivers ≤ value after the split; otherwise, keeps receivers > value before split.

## Returns
- `Tuple`: (arr1_masked, arr2_masked, arr1_renumbered, arr2_renumbered, split_index, boundary_particles)
  - Original arrays masked to keep valid edges, and renumbered versions with adjusted indices.
"""
function check_and_delete_filtered(arr1, arr2, value, boundary_last=true)
    idx = findfirst(x -> x > value, arr1)
    if isnothing(idx)
        # println("No element in arr1 is ≥ $value.")
        return arr1, arr2, arr1, arr2, value, []
    end
    if !boundary_last
        # Step 3: From index `idx` in arr2, get values > value
        keep = arr2[1:idx] .> value
        indices_to_keep = findall(keep)  # Convert to indices in full arr2
        # println(indices_to_keep)
        mask = vcat(indices_to_keep, collect((idx + 1):length(arr1)))
    else
        keep = arr2[idx:end] .<= value # Receiver is a fluid particle
        # println(findall(keep))
        indice_to_keep = findall(keep) .+ (idx-1) # indize in receiver die behalten werden
        mask = vcat(collect(1:(idx - 1)), Array(indice_to_keep)) #TODO ERROR here. too little different senders. Problem in neighbour hood search
        mask = CuArray(mask)
    end

    arr1_new = arr1[mask]

    boundary_particle_indice = arr1_new[idx:end]
    arr1_new, arr2_new = replace_with_indices(
        Array(arr1_new), Array(arr2[mask]), unique(Array(boundary_particle_indice)), value
    )

    return arr1[mask], arr2[mask], arr1_new, arr2_new, idx, arr1_new[idx:end]
end

"""
    find_missing(array, max_value)

Find all missing values in a range from minimum array value to max_value.

Identifies integer values that are not present in the input array within the specified range.
Useful for finding gaps in particle or cell indices.

## Arguments
- `array`: Array of values to check.
- `max_value`: Maximum value in the range to search.

## Returns
- `CuArray{Int32}`: Array of missing values not present in input array.
"""
function find_missing(array, max_value)
    # Determine the starting point: for example, from the minimum value in the array
    start_value = minimum(array)
    # Create a set of the array for quick lookup
    array_set = Set(Array(array))
    # Generate the full range from start_value to max_value
    full_range = start_value:max_value
    # Use set difference to find missing values
    missing_values = setdiff(full_range, array_set)
    return CuArray(missing_values)
end

"""
    replace_with_indices(arr1, arr2, refs, start_boundary)

Replace indexed values with sequential numbering starting from offset.

Replaces all occurrences of reference indices in both arrays with new sequential indices
starting from start_boundary. Useful for renumbering arrays after filtering removed indices.

## Arguments
- `arr1`: First array to modify (typically senders).
- `arr2`: Second array to modify (typically receivers).
- `refs`: Vector of indices to replace.
- `start_boundary`: Starting value for replacement indices.

## Returns
- `Tuple{CuArray,CuArray}`: (arr1_modified, arr2_modified) with replaced indices on GPU.
"""
function replace_with_indices(arr1, arr2, refs, start_boundary)
    for i in eachindex(refs)
        replace!(arr1, refs[i] => i+start_boundary)
        replace!(arr2, refs[i] => i+start_boundary)
    end
    return CuArray(arr1), CuArray(arr2)
end

"""
    compute_clostest_dist_bound(senders, receivers, rel_dist_norm, rel_displacement, len_particle, len_b_particle)::AbstractArray

Compute distance bounds for particles in contact with boundaries.

For each fluid particle in contact with boundary particles, finds the closest boundary contact
and computes a distance bound field. Returns ones for fluid-fluid interactions and the negative
relative displacement normalized for fluid-boundary interactions.

## Arguments
- `senders`: Source particle indices from edge connectivity.
- `receivers`: Receiver particle indices from edge connectivity.
- `rel_dist_norm`: Normalized distances for each edge.
- `rel_displacement`: Relative displacement vectors (dims × n_edges).
- `len_particle`: Number of fluid particles (boundary indices start after this).
- `len_b_particle`: Number of boundary particles.

## Returns
- `AbstractArray`: Distance bound field with shape (dims, len_particle + len_b_particle),
  where 1.0 for fluid-fluid and -rel_displacement for closest fluid-boundary interactions.
"""
function compute_clostest_dist_bound(
    senders, receivers, rel_dist_norm, rel_displacement, len_particle, len_b_particle
)
    indice_boundary = findall(x -> x .> len_particle, senders)
    fluid_with_boundary = receivers[indice_boundary]

    save = unique(fluid_with_boundary)
    euclid = zeros(Int, length(save))
    for idx in indice_boundary
        part = receivers[idx]
        search_idx = findfirst(x -> x == part, save)
        if euclid[search_idx] == 0
            euclid[search_idx] = idx
        elseif rel_dist_norm[euclid[search_idx]] > rel_dist_norm[idx]
            euclid[search_idx] = idx
        end
    end

    z = hcat(
        ones(size(rel_displacement, 1), len_particle),
        zeros(size(rel_displacement, 1), len_b_particle),
    )
    z[:, receivers[euclid]] = - rel_displacement[:, euclid]
    return z
end

"""
    point_neighbor_ns(pos::CuArray, radius::Float32)

GPU particle neighbor search backed by [Octopus.jl](../Octopus.jl) — the
fast octree neighborhood search of Fernández-Fernández et al. (SIGGRAPH Asia
2022). The octree is O(N) and adapts to clustered/anisotropic point sets, so on
the GPU it matches the old PointNeighbors `GridNeighborhoodSearch` on speed while
using roughly **half** the GPU memory (the old dense `FullGridCellList` scales
with domain/radius, not particle count — wasteful for the small connectivity
radii used here). The CPU `Array` path keeps the old PointNeighbors search (see
the method below), which is faster on the CPU than the octree build.

## Output convention (identical to the CPU path)
- `receivers[k] = i` (the query point), `senders[k] = j` (its neighbor).
- `rel_displacement[:, k] = (pos[:, i] - pos[:, j]) / radius`.
- `rel_dist_norm[1, k]    = ‖pos[:, i] - pos[:, j]‖ / radius`.

## Self-loops
`build_edges` excludes the self-pair `j == i`; PointNeighbors includes one
self-loop per particle (zero displacement / distance). We re-append those
self-loops so the GPU and CPU paths return identical edge sets.

## Gradients
Differentiation w.r.t. `pos` flows through `Octopus.build_edges_diff` (via
its ChainRulesCore extension); the tree topology and radius are
non-differentiable. The appended self-loops contribute zero gradient
(`sender == receiver` cancels, and their zero distance is skipped by the
`d_norm > 1e-8` guard). `build_edges_diff` computes the
finite-difference-verified `∂L/∂pos`; the original hand-written GPU rrule had a
flipped sign (see the corrected CPU rrule below for the right convention).

## Arguments
- `pos::CuArray`: Particle positions with shape (dims, n_particles).
- `radius::Float32`: Search radius for neighbor detection.

## Returns
- `Tuple`: (senders, receivers, rel_displacement, rel_dist_norm)
  - `senders::CuArray{Int32}`: Source particle indices for each edge.
  - `receivers::CuArray{Int32}`: Neighbor particle indices for each edge.
  - `rel_displacement::CuArray{Float32}`: Relative displacements normalized by radius (dims × n_edges).
  - `rel_dist_norm::CuArray{Float32}`: Euclidean distances normalized by radius (1 × n_edges).

## Notes
- Uses PointNeighbors.jl GridNeighborhoodSearch for efficient GPU computation.
- All distances and displacements are normalized by the search radius.
- Supports arbitrary dimension (2D, 3D, etc.).
"""
function point_neighbor_ns(pos::CuArray, radius::Float32)
    system = pos
    min_corner = minimum(pos; dims=2)
    max_corner = maximum(pos; dims=2)
    nhs = GridNeighborhoodSearch{size(pos, 1)}(;
        search_radius=radius,
        n_points=size(pos, 2),
        cell_list=FullGridCellList(; min_corner, max_corner, search_radius=radius),
        update_strategy=ParallelUpdate(),
    )
    # Build the cell list on-device: adapt the (empty) nhs to the GPU, then `initialize!` with the
    # CuArray positions so PointNeighbors dispatches to the parallel atomic init
    # (`default_backend(::CuArray)` => GPU; `ParallelUpdate`'s `initialize_grid!` is the parallel one).
    # Avoids the GPU->CPU->GPU round trip of the old `initialize!(nhs, Array(...), Array(...))` +
    # adapt-back, whose serial CPU cell-list build was ~16 ms at 33k particles vs ~2 ms here (~8x).
    # Edge set is identical; edge order may differ (atomic push), perturbing the downstream scatter
    # by ~eps only. Benchmarked in example/RuntimeBenchmark/.
    nhs_gpu = adapt(CUDABackend(), nhs)
    initialize!(nhs_gpu, pos, pos)

    n_neighbors_gpu = CuArray(zeros(Int, size(pos, 2)))
    foreach_point_neighbor(system, pos, nhs_gpu) do i, _, _, _
        n_neighbors_gpu[i] += 1
    end
    # n_edges = CUDA.reduce(+,n_neighbors_gpu)
    n_edges = sum(n_neighbors_gpu)
    # println("Number of edges: $n_edges")
    senders = CuArray{Int32}(undef, n_edges)
    receivers = CuArray{Int32}(undef, n_edges)
    rel_displacement = CuArray{Float32}(undef, size(pos, 1), n_edges)
    rel_dist_norm = CuArray{Float32}(undef, 1, n_edges)

    offset = CUDA.cumsum(n_neighbors_gpu) .- n_neighbors_gpu .+ 1
    foreach_point_neighbor(system, pos, nhs_gpu) do i, j, pos_diff, distance
        receivers[offset[i]] = i # switched it for different subsets in boundary situation
        senders[offset[i]] = j
        # senders[offset[i]] = i
        # receivers[offset[i]] = j
        for d in 1:size(pos, 1)
            rel_displacement[d, offset[i]] = pos_diff[d] / radius
        end
        rel_dist_norm[offset[i]] = distance/radius
        offset[i] += 1
    end
    # rel_displacement = rel_displacement ./ (Float32(radius))
    # rel_dist_norm = rel_dist_norm ./ (Float32(radius))
    senders, receivers, rel_displacement, rel_dist_norm
end

"""
    point_neighbor_ns(pos::Array, radius::Float32)

CPU version of particle neighbor search using PointNeighbors.jl GridNeighborhoodSearch.

Mirrors the GPU version but operates on plain Julia Arrays instead of CuArrays.
Together with the CuArray method, this allows `build_graph` to call
`point_neighbor_ns` unconditionally and rely on multiple dispatch.

## Arguments
- `pos::Array`: Particle positions, shape (dims, n_particles).
- `radius::Float32`: Search radius (connectivity radius).

## Returns
- `Tuple`: (senders, receivers, rel_displacement, rel_dist_norm) — all plain Arrays.
"""
function point_neighbor_ns(pos::Array, radius::Float32; max_points_per_cell::Integer=100)
    system = pos
    min_corner = minimum(pos; dims=2)
    max_corner = maximum(pos; dims=2)
    nhs = GridNeighborhoodSearch{size(pos, 1)}(;
        search_radius=radius,
        n_points=size(pos, 2),
        cell_list=FullGridCellList(;
            min_corner, max_corner, search_radius=radius, max_points_per_cell
        ),
    )
    initialize!(nhs, system, pos)

    # First pass: count neighbors per particle
    n_neighbors = zeros(Int, size(pos, 2))
    foreach_point_neighbor(system, pos, nhs) do i, _, _, _
        n_neighbors[i] += 1
    end

    n_edges = sum(n_neighbors)
    senders = Vector{Int32}(undef, n_edges)
    receivers = Vector{Int32}(undef, n_edges)
    rel_displacement = Array{Float32}(undef, size(pos, 1), n_edges)
    rel_dist_norm = Array{Float32}(undef, 1, n_edges)

    # Second pass: populate edge arrays
    offset = cumsum(n_neighbors) .- n_neighbors .+ 1
    foreach_point_neighbor(system, pos, nhs) do i, j, pos_diff, distance
        receivers[offset[i]] = i
        senders[offset[i]] = j
        for d in 1:size(pos, 1)
            rel_displacement[d, offset[i]] = pos_diff[d] / radius
        end
        rel_dist_norm[offset[i]] = distance / radius
        offset[i] += 1
    end

    return senders, receivers, rel_displacement, rel_dist_norm
end

"""
    octopus_ns(pos, radius::Float32)

Neighbor search backend using [Octopus.jl](https://github.com/una-auxme/Octopus.jl)'s fast octree
(`TNS`). Dispatches on the array type internally (CPU `Array` or `CuArray`), so the same code serves
both devices. Returns `(senders, receivers, rel_displacement, rel_dist_norm)` in exactly the format
of [`point_neighbor_ns`](@ref) — same receiver/sender convention, displacement sign, radius
normalization, and appended self-edges — so the two backends are interchangeable in [`build_graph`](@ref).
Gradients flow through `pos` via Octopus's differentiable `build_edges_diff` rrule; the tree build is
non-differentiable (`@ignore_derivatives`). The octree is O(N) and ~18x leaner than PointNeighbors'
dense grid on the GPU, so it's the backend for large point clouds where the grid OOMs.
"""
function octopus_ns(pos, radius::Float32)
    D = size(pos, 1)
    n = size(pos, 2)
    tns = ChainRulesCore.@ignore_derivatives begin
        t = Octopus.TNS(eltype(pos); ndims=D)
        Octopus.set_search_radius!(t, radius)
        pid = Octopus.add_point_set!(t, pos)
        Octopus.set_active_search!(t, pid, pid)
        Octopus.run!(t)
        t
    end
    e = Octopus.build_edges_diff(pos, tns, 1, radius)

    self_s, self_r, self_disp, self_dist = ChainRulesCore.@ignore_derivatives begin
        s = similar(e.senders, n)
        copyto!(s, Int32.(1:n))
        (
            s,
            copy(s),
            fill!(similar(e.rel_displacement, D, n), zero(eltype(e.rel_displacement))),
            fill!(similar(e.rel_dist_norm, 1, n), zero(eltype(e.rel_dist_norm))),
        )
    end

    senders = vcat(e.senders, self_s)
    receivers = vcat(e.receivers, self_r)
    rel_displacement = hcat(e.rel_displacement, self_disp)
    rel_dist_norm = hcat(e.rel_dist_norm, self_dist)
    return senders, receivers, rel_displacement, rel_dist_norm
end

"""
    neighbor_search(pos, radius::Float32, backend::Symbol)

Select the neighborhood-search implementation for graph construction. `backend` comes from
`Args.neighbor_backend` (threaded via `meta["neighbor_backend"]`):

- `:pointneighbors` — [`point_neighbor_ns`](@ref), PointNeighbors.jl grid search (default).
- `:octopus`        — [`octopus_ns`](@ref), Octopus.jl octree search (memory-lean on GPU; needed for
                      large clouds where the dense grid OOMs).
- `:auto`           — PointNeighbors on CPU (`Array`), Octopus on GPU (`CuArray`).

All branches return the identical `(senders, receivers, rel_displacement, rel_dist_norm)` format and
are differentiable in `pos`.
"""
function neighbor_search(pos, radius::Float32, backend::Symbol)
    if backend === :pointneighbors
        return point_neighbor_ns(pos, radius)
    elseif backend === :octopus
        return octopus_ns(pos, radius)
    elseif backend === :auto
        return pos isa CuArray ? octopus_ns(pos, radius) : point_neighbor_ns(pos, radius)
    else
        throw(
            ArgumentError(
                "unknown neighbor_backend $(repr(backend)); " *
                "use :pointneighbors, :octopus, or :auto",
            ),
        )
    end
end

"""
    ChainRulesCore.rrule(::typeof(point_neighbor_ns), pos::Array, radius::Float32)

Define the reverse-mode automatic differentiation rule for `point_neighbor_ns` on CPU.

Mirrors the GPU rrule but operates on plain Arrays. Enables gradient computation
through the neighbor search for CPU-based ODE training (SingleShooting, MultipleShooting).

## Arguments
- `::typeof(point_neighbor_ns)`: Function identifier.
- `pos::Array`: Particle positions.
- `radius::Float32`: Search radius.

## Returns
- `Tuple`: (primal_output, pullback_function)
"""
function ChainRulesCore.rrule(::typeof(point_neighbor_ns), pos::Array, radius::Float32)
    senders, receivers, rel_displacement, rel_dist_norm = point_neighbor_ns(pos, radius)

    function point_neighbor_ns_cpu_pullback(Δ)
        Δrel_disp_raw = Δ[3]
        Δrel_dist_raw = Δ[4]

        grad_pos = zeros(eltype(pos), size(pos))

        Δdisp = if Δrel_disp_raw isa ChainRulesCore.AbstractZero
            zeros(Float32, size(rel_displacement))
        elseif Δrel_disp_raw isa AbstractArray
            Δrel_disp_raw
        else
            convert(Array{Float32}, Δrel_disp_raw)
        end

        Δdist = if Δrel_dist_raw isa ChainRulesCore.AbstractZero
            zeros(Float32, size(rel_dist_norm))
        elseif Δrel_dist_raw isa AbstractArray
            Δrel_dist_raw
        else
            convert(Array{Float32}, Δrel_dist_raw)
        end

        if !(
            Δrel_disp_raw isa ChainRulesCore.NoTangent &&
            Δrel_dist_raw isa ChainRulesCore.NoTangent
        )
            for idx in eachindex(senders)
                i = Int(receivers[idx])
                j = Int(senders[idx])

                for d in 1:size(grad_pos, 1)
                    val_disp = Δdisp[d, idx] / radius
                    # rel_displacement[d,idx] = (pos[d,i] - pos[d,j]) / radius, with
                    # i = receiver, j = sender, so ∂/∂pos_i = +1/r and ∂/∂pos_j = -1/r:
                    # +receiver, -sender (FD-verified; see test_neighbor_backend.jl).
                    grad_pos[d, i] += val_disp
                    grad_pos[d, j] -= val_disp
                end

                d_norm = rel_dist_norm[1, idx]
                if d_norm > 1.0f-8
                    for d in 1:size(grad_pos, 1)
                        grad_val =
                            (Δdist[1, idx] * rel_displacement[d, idx]) / (d_norm * radius)
                        grad_pos[d, i] += grad_val
                        grad_pos[d, j] -= grad_val
                    end
                end
            end
        end

        return (NoTangent(), grad_pos, NoTangent(), NoTangent())
    end

    return (senders, receivers, rel_displacement, rel_dist_norm),
    point_neighbor_ns_cpu_pullback
end

"""
    ChainRulesCore.rrule(::typeof(point_neighbor_ns), pos::CuArray, radius::Float32)

Define the reverse-mode automatic differentiation rule for `point_neighbor_ns`.

Enables gradient computation through the neighbor search operation for backpropagation.
Gradients flow only through position; radius is treated as constant.

## Arguments
- `::typeof(point_neighbor_ns)`: Function identifier.
- `pos::CuArray`: Particle positions.
- `radius::Float32`: Search radius.

## Returns
- `Tuple`: (primal_output, pullback_function)
  - `primal_output`: (senders, receivers, rel_displacement, rel_dist_norm).
  - `pullback_function`: Function that computes gradients with respect to position.

## Notes
- Only position gradients are computed; radius gradient is NoTangent.
- Uses GPU kernel for efficient gradient computation.
- Handles various tangent types from Zygote (AbstractZero, AbstractArray, etc.).
"""
function ChainRulesCore.rrule(::typeof(point_neighbor_ns), pos::CuArray, radius::Float32)
    # Forward Pass
    senders, receivers, rel_displacement, rel_dist_norm = point_neighbor_ns(pos, radius)

    function point_neighbor_ns_pullback(Δ)
        # Δ = (f̄, pos̄, radius̄)
        # Note: Δ is a Tuple, so Δ[3] is rel_displacement gradient, Δ[4] is rel_dist_norm
        Δrel_disp_raw = Δ[3]
        Δrel_dist_raw = Δ[4]

        # Initialize gradient for pos
        grad_pos = CUDA.zeros(eltype(pos), size(pos))

        # Helper to convert Zygote Tangents/Nothing to CuArray
        # This is critical to prevent "dynamic invocation"
        function ensure_cuda(amt, dims)
            if amt isa AbstractArray
                return amt
            elseif amt isa ChainRulesCore.AbstractZero
                return CUDA.zeros(Float32, dims...)
            else
                # Handle cases where Zygote might wrap the array in a Fill or NamedTuple
                return convert(CuArray{Float32}, amt)
            end
        end

        Δdisp = ensure_cuda(Δrel_disp_raw, size(rel_displacement))
        Δdist = ensure_cuda(Δrel_dist_raw, size(rel_dist_norm))

        # Only compute if we have non-zero gradients
        if !(
            Δrel_disp_raw isa ChainRulesCore.NoTangent &&
            Δrel_dist_raw isa ChainRulesCore.NoTangent
        )
            n_edges = length(senders)
            threads = 256
            blocks = cld(n_edges, threads)

            @cuda threads=threads blocks=blocks pullback_kernel!(
                grad_pos,
                Δdisp,
                Δdist,
                senders,
                receivers,
                rel_displacement,
                rel_dist_norm,
                radius,
            )
        end
        # Return gradients for: (::typeof(point_neighbor_ns), pos, radius, mask)    
        return (NoTangent(), grad_pos, NoTangent(), NoTangent())
    end

    return (senders, receivers, rel_displacement, rel_dist_norm), point_neighbor_ns_pullback
end

"""
    pullback_kernel!(grad_pos, Δrel_disp, Δrel_dist, senders, receivers, rel_disp, rel_dist, radius)

GPU kernel for computing position gradients during backpropagation through neighbor search.

Computes gradients with respect to particle positions from gradients of relative displacements
and distances. Uses atomic operations to safely accumulate gradients for each particle.

## Arguments
- `grad_pos`: Output gradient array with shape (dims, n_particles).
- `Δrel_disp`: Gradient w.r.t. relative displacements (dims × n_edges).
- `Δrel_dist`: Gradient w.r.t. normalized distances (1 × n_edges).
- `senders`: Source particle indices.
- `receivers`: Receiver particle indices.
- `rel_disp`: Relative displacement values (dims × n_edges).
- `rel_dist`: Normalized distance values (1 × n_edges).
- `radius`: Search radius used in normalization.

## Implementation Details
- Launched as CUDA kernel with threads=256.
- Uses atomic addition to handle gradient contributions to shared particles.
- Gradient contribution splits between displacement and distance terms with proper normalization.
- Returns nothing (modifies grad_pos in-place).
"""
function pullback_kernel!(
    grad_pos, Δrel_disp, Δrel_dist, senders, receivers, rel_disp, rel_dist, radius
)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if idx <= length(senders)
        # Use Int64 for indexing to avoid overflows and ensure compatibility
        i = Int32(receivers[idx])
        j = Int32(senders[idx])

        # 1. Gradient from rel_displacement (Matrix: dim x n_edges)
        for d in 1:size(grad_pos, 1)
            val_disp = Δrel_disp[d, idx] / radius
            # rel_displacement = (pos_i - pos_j)/radius (i=receiver, j=sender), so the
            # position gradient is +receiver, -sender (FD-verified).
            CUDA.atomic_add!(pointer(grad_pos, (i-1)*size(grad_pos, 1) + d), val_disp)
            CUDA.atomic_add!(pointer(grad_pos, (j-1)*size(grad_pos, 1) + d), -val_disp)
        end

        # 2. Gradient from rel_dist_norm (Matrix: 1 x n_edges)
        # Match your allocation: rel_dist_norm[1, offset[i]]
        d_norm = rel_dist[1, idx]

        if d_norm > 1.0f-8
            for d in 1:size(grad_pos, 1)
                # rel_disp is also [d, idx]
                grad_val = (Δrel_dist[1, idx] * rel_disp[d, idx]) / (d_norm * radius)

                CUDA.atomic_add!(pointer(grad_pos, (i-1)*size(grad_pos, 1) + d), grad_val)
                CUDA.atomic_add!(pointer(grad_pos, (j-1)*size(grad_pos, 1) + d), -grad_val)
            end
        end
    end
    return nothing
end
