#
# Copyright (c) 2026 Josef Kircher, Julian Trommer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using CUDA
import Statistics: norm
using JLD2
using PointNeighbors
using ChainRulesCore

"""
    GraphCache{S,W}

Mutable cache holding the current graph topology and the list of particle
pairs whose normalized distance is within the "watch band" `[0.9, 1.1]` of
the search radius. Used by the graph-reconstruction callback: topology is
held constant between callback events so the ODE RHS is differentiable with
a fixed edge set, and `rebuild_topology!` refreshes it whenever a watched
pair crosses the radius.

## Fields
- `senders`, `receivers`: current edge index arrays (device-resident).
- `watched_pairs`: `2 × n_watched_max` host-side `Int32` matrix of
  undirected pairs `(i, j)` with `i < j`. Only columns `1:n_watched` are
  active; the rest are ignored.
- `n_watched`: number of currently active watched pairs.
- `n_watched_max`: preallocated capacity. If a rebuild would exceed this,
  `overflow` is set and the caller is expected to terminate+retry the solve
  with a larger capacity.
- `overflow`: set to `true` by `rebuild_topology!` if more than
  `n_watched_max` pairs fell into the watch band.
- `radius`: search radius (same units as positions).
"""
mutable struct GraphCache{S,W}
    senders::S
    receivers::S
    watched_pairs::W
    n_watched::Int
    n_watched_max::Int
    overflow::Bool
    radius::Float32
end

"""
    GraphCache(position, radius; safety_factor=4.0f0)

Build an empty `GraphCache` sized from the initial particle positions.

Runs one neighbor search at `1.1 * radius` (the upper edge of the watch
band), counts how many undirected pairs land in `[0.9, 1.1] * radius`, and
sets `n_watched_max = max(ceil(safety_factor * initial_watched), 16)`.
Does **not** populate `senders`/`receivers`/`watched_pairs` — call
`rebuild_topology!` next.

The wider search radius is essential: at the true radius `R`, neighbor
search only returns pairs with `d ≤ R`, so a particle about to *enter* the
neighborhood (currently at `d ∈ (R, 1.1R]`) would never be watched and
would silently join the graph without a callback firing. Searching at
`1.1R` ensures both entering and exiting candidates are seen.
"""
function GraphCache(position, radius::Float32; safety_factor::Float32=4.0f0)
    search_radius = 1.1f0 * radius
    senders, receivers, _, rel_dist_norm = point_neighbor_ns(position, search_radius)

    # rel_dist_norm from point_neighbor_ns is distance/search_radius.
    # Convert to true distance for band comparison.
    dists_host = vec(Array(rel_dist_norm)) .* search_radius

    s_host = Array(senders)
    r_host = Array(receivers)
    initial_watched = 0
    @inbounds for k in eachindex(s_host)
        s_host[k] < r_host[k] || continue  # undirected, skip self-edges
        d = dists_host[k]
        if 0.9f0 * radius <= d <= 1.1f0 * radius
            initial_watched += 1
        end
    end
    n_watched_max = max(ceil(Int, safety_factor * initial_watched), 16)

    is_gpu = position isa CuArray
    senders_init = is_gpu ? CUDA.zeros(Int32, 0) : Int32[]
    receivers_init = is_gpu ? CUDA.zeros(Int32, 0) : Int32[]
    watched_pairs = zeros(Int32, 2, n_watched_max)
    return GraphCache(
        senders_init, receivers_init, watched_pairs, 0, n_watched_max, false, radius
    )
end

"""
    rebuild_topology!(cache::GraphCache, position) -> Bool

Refresh `cache.senders`, `cache.receivers`, and `cache.watched_pairs` from
the current particle `position`.

Runs `point_neighbor_ns` at the **widened** radius `1.1 * cache.radius` so
both in-neighborhood pairs (`d ≤ R`) and about-to-enter pairs (`d ∈ (R,
1.1R]`) are visible. Distances are recomputed directly from `position` and
used to split the edge list into:

- **Topology**: pairs with `d ≤ R` → stored in `cache.senders`/`receivers`
  (device-resident, in both directions as `point_neighbor_ns` returns them).
- **Watched**: undirected pairs (`i < j`) with `0.9R ≤ d ≤ 1.1R` → stored
  host-side in `cache.watched_pairs[:, 1:n_watched]` so the callback's
  `condition!` can index them with plain scalar Julia.

If the number of watched pairs exceeds `cache.n_watched_max`,
`cache.overflow` is set to `true`, `n_watched` is clamped to
`n_watched_max`, and the function returns `false`. Callers should then
terminate the solve, grow `n_watched_max`, and retry.

Returns `true` on success (no overflow).
"""
function rebuild_topology!(cache::GraphCache, position)
    radius = cache.radius
    search_radius = 1.1f0 * radius
    senders_wide, receivers_wide, _, rel_dist_norm = point_neighbor_ns(
        position, search_radius
    )

    # rel_dist_norm from point_neighbor_ns is distance/search_radius (1×n_edges).
    # Topology filtering on GPU: keep pairs whose true distance ≤ R,
    # i.e. (dist_norm * search_radius) ≤ R ⟺ dist_norm ≤ R/search_radius.
    topo_threshold = radius / search_radius  # = 1/1.1 ≈ 0.909
    topo_mask = vec(rel_dist_norm) .<= topo_threshold
    cache.senders = senders_wide[topo_mask]
    cache.receivers = receivers_wide[topo_mask]

    # Watched pairs: undirected (i < j), true distance in [0.9R, 1.1R].
    # Convert thresholds to normalized units: d/search_radius.
    lo = 0.9f0 * radius / search_radius
    hi = 1.0f0  # 1.1R / search_radius = 1.0 (by definition)
    dist_norm_vec = vec(rel_dist_norm)

    # Transfer only senders/receivers/dists to host for the watched-pair
    # scalar loop. This is a single bulk transfer rather than per-element
    # scalar indexing, and avoids recomputing distances.
    s_host = Array(senders_wide)
    r_host = Array(receivers_wide)
    d_host = Array(dist_norm_vec)

    n_watched = 0
    overflow = false
    @inbounds for k in eachindex(s_host)
        i = s_host[k]
        j = r_host[k]
        i < j || continue  # undirected
        dn = d_host[k]
        if lo <= dn <= hi
            if n_watched < cache.n_watched_max
                n_watched += 1
                cache.watched_pairs[1, n_watched] = i
                cache.watched_pairs[2, n_watched] = j
            else
                overflow = true
                break
            end
        end
    end

    cache.n_watched = n_watched
    cache.overflow = overflow
    return !overflow
end

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

    build_graph(
        gns,
        current_position,
        data["velocity"][:, :, datapoint],
        meta,
        node_type,
        data["mask"],
        device,
    )
end

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
    senders, receivers, rel_displacement, rel_dist_norm = point_neighbor_ns(
        position, Float32(meta["default_connectivity_radius"])
    )
    return _assemble_feature_graph(
        gns,
        position,
        velocity,
        rel_displacement,
        rel_dist_norm,
        senders,
        receivers,
        meta,
        node_type,
        mask,
        device,
    )
end

"""
    build_graph_cached(gns, cache::GraphCache, position, velocity, meta, node_type, mask, device)

Construct a `FeatureGraph` using the topology stored in `cache` instead of
running a fresh neighbor search.

Computes `rel_displacement` and `rel_dist_norm` directly from
`cache.senders`/`cache.receivers` via plain (Zygote-differentiable) array
indexing, then delegates feature assembly to `_assemble_feature_graph`.
This is the path used inside the ODE RHS when the graph-reconstruction
callback is active: between callback events the topology is held constant
so gradients flow cleanly through the cached edge set without ever calling
the custom `point_neighbor_ns` rrule.

The caller is responsible for keeping `cache` in sync with `position`
(typically via `maybe_rebuild_topology!` called from the ODE RHS).
"""
function build_graph_cached(
    gns::GraphNetCore.GraphNetwork,
    cache::GraphCache,
    position,
    velocity,
    meta,
    node_type,
    mask,
    device,
)
    radius = cache.radius
    senders = cache.senders
    receivers = cache.receivers

    # Differentiable edge features from cached topology. Direction matches
    # `point_neighbor_ns`: pos_diff = pos[receiver] - pos[sender].
    rel_displacement = (position[:, receivers] .- position[:, senders]) ./ radius
    rel_dist_norm = sqrt.(sum(abs2, rel_displacement; dims=1) .+ 1.0f-12)

    return _assemble_feature_graph(
        gns,
        position,
        velocity,
        rel_displacement,
        rel_dist_norm,
        senders,
        receivers,
        meta,
        node_type,
        mask,
        device,
    )
end

"""
    _assemble_feature_graph(gns, position, velocity, rel_displacement, rel_dist_norm,
                            senders, receivers, meta, node_type, mask, device)

Internal helper that builds a `FeatureGraph` from already-computed edge
topology and edge geometry. Shared between `build_graph` (which sources its
topology from `point_neighbor_ns`) and `build_graph_cached` (which sources
it from a `GraphCache`). All node-feature/edge-feature normalization and
the boundary distance bound logic live here.
"""
function _assemble_feature_graph(
    gns::GraphNetCore.GraphNetwork,
    position,
    velocity,
    rel_displacement,
    rel_dist_norm,
    senders,
    receivers,
    meta,
    node_type,
    mask,
    device,
)
    if n_node_types(meta) > 1
        if length(mask) == size(position, 2)
            dist_bound = device(ones(Float32, size(position)...))
        else
            boundaries = device(Float32.(vcat(permutedims.(meta["bounds"])...)))
            dist_low_bound = position .- boundaries[:, 1]
            dist_up_bound = boundaries[:, 2] .- position
            dist_bound = clamp.(
                vcat(dist_low_bound, dist_up_bound) ./
                Float32(meta["default_connectivity_radius"]),
                -1.0f0,
                1.0f0,
            )
        end
    end

    edge_features = device(vcat(rel_displacement, rel_dist_norm) .+ 1.0f-8)

    if n_node_types(meta) == 1
        if length(meta["input_features"]) == 2
            node_features = device(
                vcat(gns.n_norm["position"](position), gns.n_norm["velocity"](velocity))
            )
        else
            node_features = device(gns.n_norm["velocity"](velocity))
        end
    else
        if length(meta["input_features"]) == 2
            node_features = device(
                vcat(
                    gns.n_norm["position"](position),
                    gns.n_norm["velocity"](velocity),
                    dist_bound,
                    node_type,
                ),
            )
        else
            node_features = device(
                vcat(gns.n_norm["velocity"](velocity), dist_bound, node_type)
            )
        end
    end

    return GraphNetCore.FeatureGraph(
        node_features, gns.e_norm(edge_features), senders, receivers
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

Perform GPU-accelerated neighbor search using PointNeighbors grid-based acceleration.

Constructs a grid neighborhood search structure from particle positions and radius,
then efficiently finds all particle pairs within the search radius using grid-based acceleration.
Returns normalized relative displacements and distances.

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
    system = pos#[:,mask]
    min_corner = minimum(pos; dims=2)
    max_corner = maximum(pos; dims=2)
    nhs = GridNeighborhoodSearch{size(pos, 1)}(;
        search_radius=radius,
        n_points=size(pos, 2),
        cell_list=FullGridCellList(; min_corner, max_corner, search_radius=radius),
    )
    initialize!(nhs, Array(system), Array(pos))
    backend = CUDABackend()
    # Simple example: just count the neighbors of each particle
    n_neighbors_gpu = CuArray(zeros(Int, size(pos, 2)))
    nhs_gpu = adapt(backend, nhs)

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
function point_neighbor_ns(pos::Array, radius::Float32)
    system = pos
    min_corner = minimum(pos; dims=2)
    max_corner = maximum(pos; dims=2)
    nhs = GridNeighborhoodSearch{size(pos, 1)}(;
        search_radius=radius,
        n_points=size(pos, 2),
        cell_list=FullGridCellList(; min_corner, max_corner, search_radius=radius),
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
                    grad_pos[d, j] += val_disp
                    grad_pos[d, i] -= val_disp
                end

                d_norm = rel_dist_norm[1, idx]
                if d_norm > 1.0f-8
                    for d in 1:size(grad_pos, 1)
                        grad_val =
                            (Δdist[1, idx] * rel_displacement[d, idx]) / (d_norm * radius)
                        grad_pos[d, j] += grad_val
                        grad_pos[d, i] -= grad_val
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
            # Use linear indexing for grad_pos [d, j] and [d, i]
            CUDA.atomic_add!(pointer(grad_pos, (j-1)*size(grad_pos, 1) + d), val_disp)
            CUDA.atomic_add!(pointer(grad_pos, (i-1)*size(grad_pos, 1) + d), -val_disp)
        end

        # 2. Gradient from rel_dist_norm (Matrix: 1 x n_edges)
        # Match your allocation: rel_dist_norm[1, offset[i]]
        d_norm = rel_dist[1, idx]

        if d_norm > 1.0f-8
            for d in 1:size(grad_pos, 1)
                # rel_disp is also [d, idx]
                grad_val = (Δrel_dist[1, idx] * rel_disp[d, idx]) / (d_norm * radius)

                CUDA.atomic_add!(pointer(grad_pos, (j-1)*size(grad_pos, 1) + d), grad_val)
                CUDA.atomic_add!(pointer(grad_pos, (i-1)*size(grad_pos, 1) + d), -grad_val)
            end
        end
    end
    return nothing
end
