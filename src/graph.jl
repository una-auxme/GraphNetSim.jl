#
# Copyright (c) 2026 Josef Kircher, Julian Trommer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using CUDA
import Statistics: norm
using JLD2
using PointNeighbors  # CPU neighbor search (Array path); GPU path uses TreeNSearch
using TreeNSearch      # GPU neighbor search (CuArray path) — fast octree, half the GPU memory
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
    # if size(boundaries,2)== 0
    #     current_position = position
    # else
    #     current_position = hcat(position, boundaries)
    #     velocity = hcat(velocity, zeros(Float32, meta["dims"], size(boundaries,2)))
    # end

    # Materialize `position` onto the active device's *concrete* array type
    # before the neighbor search. In the ODE solver path `position` is `x.x`, a
    # GPU-backed ComponentArray view (`<: AbstractArray` but not `<: CuArray`),
    # which would otherwise miss `point_neighbor_ns(::CuArray)` and fall back to
    # the host search. `device(...)` yields a `CuArray` on GPU / `Array` on CPU,
    # selecting the matching method so the search AND its gradient
    # (TreeNSearch `build_edges_diff`) stay on-device. `device` differentiates
    # (MLDataDevices rrule), so ∂L/∂pos flows back to the view.
    senders, receivers, rel_displacement, rel_dist_norm = point_neighbor_ns(
        device(position), Float32(meta["default_connectivity_radius"])
    )
    # if size(boundaries,2) != 0
    # #     # sender_old, receiver_old, senders, receivers, _, b_particle = check_and_delete_filtered(senders, receivers, size(position, 2), true)
    # #     # rel_displacement = (position[:, receiver_old] - position[:, sender_old]) ./ Float32(meta["default_connectivity_radius"])
    # #     # rel_dist_norm = sqrt.(sum(abs2, rel_displacement; dims = 1))
    #     dist_bound = compute_clostest_dist_bound(Array(senders), Array(receivers), Array(rel_dist_norm), Array(rel_displacement), size(position,2), length(unique(Array(b_particle))))
    #     particles = unique(Array(senders))
    # else
    #     dist_bound = cu(ones(Float32, size(position)...))
    #     particles = Colon()
    # end

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

GPU particle neighbor search backed by [TreeNSearch.jl](../fast_octree) — the
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
Differentiation w.r.t. `pos` flows through `TreeNSearch.build_edges_diff` (via
its ChainRulesCore extension); the tree topology and radius are
non-differentiable. The appended self-loops contribute zero gradient
(`sender == receiver` cancels, and their zero distance is skipped by the
`d_norm > 1e-8` guard). `build_edges_diff` computes the
finite-difference-verified `∂L/∂pos`; the original hand-written GPU rrule had a
flipped sign (see the corrected CPU rrule below for the right convention).

## Arguments
- `pos::CuArray`: particle positions, shape `(dims, n_particles)`.
- `radius::Float32`: search (connectivity) radius.

## Returns
- `Tuple`: `(senders, receivers, rel_displacement, rel_dist_norm)` as CuArrays.
"""
function point_neighbor_ns(pos::CuArray, radius::Float32)
    D = size(pos, 1)
    n = size(pos, 2)

    # Build the octree and run the search. These setup ops mutate `tns` and are
    # non-differentiable; only `build_edges_diff` carries the gradient to `pos`.
    tns = ChainRulesCore.@ignore_derivatives begin
        t = TNS(eltype(pos); ndims=D)
        set_search_radius!(t, radius)
        id = add_point_set!(t, pos)
        set_active_search!(t, id, id)
        run!(t)
        t
    end
    e = build_edges_diff(pos, tns, 1, radius)

    # Re-append the self-loops that `build_edges` drops (j == i), matching the
    # historical PointNeighbors behavior. They are constant w.r.t. `pos`.
    self_s, self_r, self_disp, self_dist = ChainRulesCore.@ignore_derivatives begin
        s = similar(e.senders, n)
        copyto!(s, Int32.(1:n))
        (
            s,
            copy(s),
            fill!(similar(e.rel_displacement, D, n), 0),
            fill!(similar(e.rel_dist_norm, 1, n), 0),
        )
    end

    senders = vcat(e.senders, self_s)
    receivers = vcat(e.receivers, self_r)
    rel_displacement = hcat(e.rel_displacement, self_disp)
    rel_dist_norm = hcat(e.rel_dist_norm, self_dist)

    return senders, receivers, rel_displacement, rel_dist_norm
end

"""
    point_neighbor_ns(pos::Array, radius::Float32)

CPU particle neighbor search using PointNeighbors.jl `GridNeighborhoodSearch`
(`FullGridCellList`). Mirrors the GPU `CuArray` method's interface and output
convention exactly — query = receiver `i`, neighbor = sender `j`,
`rel_displacement = (pos_i - pos_j)/radius`, one self-loop per particle — so
`build_graph` can call `point_neighbor_ns` unconditionally and dispatch on the
array type.

The grid search is kept on the CPU because building the TreeNSearch octree is
markedly slower there (~10× on a 46k-point droplet); TreeNSearch is used only on
the GPU, where it matches the grid on speed and uses ~half the GPU memory.
Gradients flow through the corrected rrule below.
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

Reverse-mode AD rule for the CPU (`Array`) neighbor search. Computes `∂L/∂pos`
from the upstream tangents on `rel_displacement` / `rel_dist_norm`; the edge set
and the radius are non-differentiable.

**Sign convention — corrected vs the original implementation** (verified against
finite differences and matching `TreeNSearch.build_edges_diff`): since
`rel_displacement[:, k] = (pos[:, i] - pos[:, j]) / radius` with `i = receiver`
and `j = sender`, the gradient accumulates **+ on the receiver** and **− on the
sender**. The original CPU/GPU rrules had these swapped, which negated `∂L/∂pos`
and affected only the ODE/solver strategies (the sole paths that backprop through
positions). Self-loops (`i == j`) cancel, and their zero distance is skipped by
the `> 1e-8` guard.
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
                i = Int(receivers[idx])   # query point
                j = Int(senders[idx])     # neighbor

                for d in 1:size(grad_pos, 1)
                    val_disp = Δdisp[d, idx] / radius
                    grad_pos[d, i] += val_disp   # ∂rel_disp/∂pos_i = +1/r
                    grad_pos[d, j] -= val_disp   # ∂rel_disp/∂pos_j = -1/r
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
