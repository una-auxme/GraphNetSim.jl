#
# Copyright (c) 2026 Josef Kircher
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using CUDA

"""
    maybe_rebuild_topology!(cache::GraphCache, position)

Check whether any watched particle pair has left the watch band
`[0.9R, 1.1R]` and, if so, rebuild the cached graph topology via
`rebuild_topology!`.

This function is designed to be called from the ODE RHS inside an
`@ignore_derivatives` block. Because it never touches the ODE state `u`
(only the external `GraphCache`), the adjoint correction is zero and
Zygote can safely ignore it. This avoids the need for a
`VectorContinuousCallback`, which would force a non-Zygote VJP backend
(ReverseDiffVJP / EnzymeVJP) — neither of which works on GPU.

During the adjoint backward pass the RHS is re-evaluated at interpolated
forward-solution points. The topology check runs again with those
positions, producing the correct (or very close) topology for each time
point. This is actually slightly better than the callback approach, where
the backward pass would use the final-forward-solve topology everywhere.

## When a rebuild triggers

A watched pair `(i, j)` was in `[0.9R, 1.1R]` at the last rebuild. If
its current distance has left this band (`< 0.9R` or `> 1.1R`), it means
the pair has crossed the radius boundary and the topology is stale.

## Overflow handling

If `rebuild_topology!` reports overflow (more watched pairs than
`cache.n_watched_max`), the `cache.overflow` flag is set. The caller
(training-step wrapper) should inspect this flag after `solve` returns
and retry with a larger capacity.

## GPU note

Copies the full position matrix to host via `Array(position)` for scalar
indexing. For typical particle counts this is negligible compared to the
per-step GNN forward pass.
"""
function maybe_rebuild_topology!(cache::GraphCache, position)
    n = cache.n_watched
    n == 0 && return nothing

    # Transfer position to host for scalar distance checks. For typical
    # particle counts this single bulk copy (dims × n_particles × 4 bytes)
    # is much cheaper than launching multiple GPU kernels for a handful of
    # watched pairs.
    pos_host = Array(position)
    radius = cache.radius
    needs_rebuild = false
    @inbounds for k in 1:n
        i = cache.watched_pairs[1, k]
        j = cache.watched_pairs[2, k]
        d2 = 0.0f0
        for d in 1:size(pos_host, 1)
            Δ = pos_host[d, i] - pos_host[d, j]
            d2 += Δ * Δ
        end
        dist = sqrt(d2)
        if dist < 0.9f0 * radius || dist > 1.1f0 * radius
            needs_rebuild = true
            break
        end
    end
    if needs_rebuild
        rebuild_topology!(cache, position)
    end
    return nothing
end
