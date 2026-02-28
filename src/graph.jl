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

    if device == cpu_device()
        cur_pos_cpu = cpu_device()(position)
        tree = KDTree(cur_pos_cpu; reorder=false)
        receivers_list = device(inrange(
            tree, cur_pos_cpu, Float32(meta["default_connectivity_radius"]), false
        ))
        senders = device(vcat(
            [repeat([i], length(j)) for (i, j) in enumerate(receivers_list)]...
        ))
        receivers = vcat(receivers_list...)
        rel_displacement =
            (position[:, receivers] - position[:, senders]) ./
            Float32(meta["default_connectivity_radius"])
        rel_dist_norm = sqrt.(sum(abs2, rel_displacement; dims=1))
    else
        senders, receivers, rel_displacement, rel_dist_norm = point_neighbor_ns(
            position, Float32(meta["default_connectivity_radius"])
        ) # experimental
    end
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

    if length(mask) == size(position, 2) # TODO val_mask is always position size long, this needs something smarter
        dist_bound = cu(ones(Float32, size(position)...))
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
    # dist_bound = cu(ones(Float32, size(position)...))
    edge_features = device(vcat(rel_displacement, rel_dist_norm) .+ 1.0f-8)
    input_dim = 0
    for feature in meta["input_features"]
        input_dim += meta["features"][feature]["dim"]
    end

    if length(meta["input_features"]) == 2
        node_features = device(vcat(
            gns.n_norm["position"](position),
            gns.n_norm["velocity"](velocity),
            dist_bound,
            gns.n_norm["node_type"](node_type),
        ))
    else
        node_features = device(vcat(
            gns.n_norm["velocity"](velocity),
            dist_bound,
            gns.n_norm["node_type"](node_type),
        ))
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
