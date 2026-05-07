#
# Copyright (c) 2026 Josef Kircher, Julian Trommer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

"""
    rollout_history(gns, initial_state, output_fields, meta, target_fields,
                    node_type, mask, val_mask, sim_interval, device, pr=nothing)

Hand-written fixed-step Euler rollout that maintains a `(dim, particles, C)` velocity
history buffer for `meta["history_size"] > 1`.

Each step:
1. Build graph from current `position` and the C-step velocity buffer.
2. Run the GNN, denormalize outputs, apply `val_mask`.
3. Integrate: `velocity += dt * accel`, `position += dt * velocity`.
4. Slide the buffer one step (drop index 1, append new velocity at index C).

# Buffer warmup

If `initial_state` contains a 3D `"velocity_window"` of shape `(dim, particles, C)`,
the buffer is seeded with those C ground-truth velocities (paper-faithful: the model
sees an in-distribution history at step 1). The caller is responsible for pairing
`"position"` with the last frame of the window so the newest buffer slot and the
position correspond to the same time step (matching the training-time pairing).

If `"velocity_window"` is absent the buffer falls back to `repeat(initial_state["velocity"], C)`,
which is OOD vs. the training distribution — only intended as a debug/legacy path.

Returns `(t = times, u = states, acc = accelerations)` mirroring the subset of the
`sol.u` interface that `_validation_step` and `_extract_trajectory_arrays` consume.
"""
function rollout_history(
    gns::GraphNetCore.GraphNetwork,
    initial_state,
    output_fields,
    meta,
    target_fields,
    node_type,
    mask,
    val_mask,
    sim_interval,
    device,
    pr=nothing,
)
    C = get(meta, "history_size", 1)
    pos = initial_state["position"]
    vh = if haskey(initial_state, "velocity_window")
        vw = initial_state["velocity_window"]
        size(vw, 3) == C || throw(
            ArgumentError(
                "velocity_window must have last dim == history_size = $C; got " *
                "size $(size(vw)).",
            ),
        )
        device(vw)
    else
        v0 = initial_state["velocity"]
        d, n = size(v0, 1), size(v0, 2)
        device(repeat(reshape(v0, d, n, 1); outer=(1, 1, C)))
    end
    vel = vh[:, :, end]
    dim, np = size(vel, 1), size(vel, 2)
    dt = Float32(sim_interval[2] - sim_interval[1])

    indices = [meta["features"][tf]["dim"] for tf in target_fields]
    saved = [(; x=copy(pos), dx=copy(vel))]
    accs = [device(zeros(Float32, dim, np))]
    times = Float32[Float32(sim_interval[1])]

    for k in 1:(length(sim_interval) - 1)
        graph = build_graph(gns, pos, vh, meta, node_type, mask, device)
        output, st = gns.model(graph, gns.ps, gns.st)
        gns.st = st

        denorm = similar(output)
        for i in eachindex(output_fields)
            r = (sum(indices[1:(i - 1)]) + 1):sum(indices[1:i])
            denorm[r, :] = inverse_data(gns.o_norm[output_fields[i]], output[r, :])
        end
        accel = denorm .* val_mask

        vel = vel .+ dt .* accel
        pos = pos .+ dt .* vel
        vh = cat(vh[:, :, 2:end], reshape(vel, dim, np, 1); dims=3)

        push!(saved, (; x=copy(pos), dx=copy(vel)))
        push!(accs, copy(accel))
        push!(times, Float32(sim_interval[k + 1]))
        if !isnothing(pr)
            next!(pr; showvalues=[(:t, "$(length(saved))")])
        end
    end

    if !isnothing(pr)
        finish!(pr)
    end

    return (t=times, u=saved, acc=accs)
end

function _extract_trajectory_arrays(
    sol::NamedTuple{(:t, :u, :acc)}
)
    sol_pos = cpu_device()(cat([u.x for u in sol.u]...; dims=3))
    sol_vel = cat([u.dx for u in sol.u]...; dims=3)
    sol_acc = cat(sol.acc...; dims=3)
    return sol.t, (pos=sol_pos, vel=sol_vel, acc=sol_acc)
end
