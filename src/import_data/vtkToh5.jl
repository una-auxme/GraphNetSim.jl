#
# Copyright (c) 2026 Josef Jouaux
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
# VTK -> HDF5 importer for particle trajectories (DualSPHysics / ParaView style).
#
# ReadVTK.jl cannot open the collection/parallel wrapper formats these datasets
# ship in (`Collection` .pvd, `PUnstructuredGrid` .pvtu) and crashes on empty
# (0-point) pieces. We therefore parse the .pvd/.pvtu wrappers ourselves (plain
# XML, LightXML) and hand *only* the leaf .vtu pieces to ReadVTK, which decodes
# their raw/zlib/appended binary correctly. 0-point frames are detected from the
# leaf header and skipped before ReadVTK is ever called.
#

using HDF5
# Import (not `using`) both VTK libraries: several of their exports (`attribute`,
# `get_data`, …) collide with CUDA/GraphNetCore names inside this module, so we
# qualify every call explicitly.
using ReadVTK: ReadVTK
using LightXML: LightXML
using JSON: JSON
using Statistics: Statistics

# Common name aliases for each logical role, matched case-insensitively when the
# explicitly requested field name is absent from a file's PointData arrays.
const _VTK_FIELD_ALIASES = Dict{Symbol,Vector{String}}(
    :velocity => ["vel", "velocity", "v", "u"],
    :acceleration => ["acc", "acceleration", "accel", "a"],
    :type => ["type", "phaseid", "mk", "mat", "material", "phase", "typ"],
    :id => ["prtlid", "idp", "id", "particleid", "pointid"],
)

# ---------------------------------------------------------------------------
# Wrapper parsing (.pvd / .pvtu)
# ---------------------------------------------------------------------------

"""
    _resolve_source_timesteps(source) -> (times, piece_files)

Turn a `source` (a `.pvd`, a directory, or a single `.pvtu`/`.vtu`) into an
ordered list of timesteps. Returns `times::Vector{Float64}` and
`piece_files::Vector{Vector{String}}`, where `piece_files[t]` is the list of
leaf `.vtu` paths making up timestep `t`.
"""
function _resolve_source_timesteps(source::AbstractString)
    if isfile(source) && endswith(source, ".pvd")
        return _parse_pvd(source)
    elseif isdir(source)
        pvds = filter(f -> endswith(f, ".pvd"), readdir(source; join=true))
        if !isempty(pvds)
            return _parse_pvd(first(sort(pvds)))
        end
        pvtus = sort(filter(f -> endswith(f, ".pvtu"), readdir(source; join=true)))
        if !isempty(pvtus)
            return collect(Float64, 0:(length(pvtus) - 1)),
            [_expand_pieces(p) for p in pvtus]
        end
        # A directory of one-file-per-timestep leaves (`.vtu` or legacy `.vtk`).
        leaves = sort(
            filter(
                f -> endswith(f, ".vtu") || endswith(f, ".vtk"), readdir(source; join=true)
            ),
        )
        isempty(leaves) && throw(
            ArgumentError("No .pvd, .pvtu, .vtu or .vtk files found in directory: $source"),
        )
        return collect(Float64, 0:(length(leaves) - 1)), [[v] for v in leaves]
    elseif isfile(source) && (
        endswith(source, ".pvtu") || endswith(source, ".vtu") || endswith(source, ".vtk")
    )
        return [0.0], [_expand_pieces(source)]
    else
        throw(
            ArgumentError(
                "Invalid VTK source (expected .pvd, directory, .pvtu, .vtu or .vtk): $source",
            ),
        )
    end
end

"""Parse a `.pvd` Collection into `(times, piece_files)`."""
function _parse_pvd(pvd_path::AbstractString)
    doc = LightXML.parse_file(pvd_path)
    times = Float64[]
    piece_files = Vector{Vector{String}}()
    try
        r = LightXML.root(doc)
        colls = LightXML.get_elements_by_tagname(r, "Collection")
        isempty(colls) && throw(ArgumentError("No <Collection> in .pvd: $pvd_path"))
        base = dirname(pvd_path)
        for ds in LightXML.get_elements_by_tagname(first(colls), "DataSet")
            tstr = LightXML.attribute(ds, "timestep")
            fstr = LightXML.attribute(ds, "file")
            fstr === nothing &&
                throw(ArgumentError("<DataSet> without `file` in $pvd_path"))
            fpath = isabspath(fstr) ? fstr : joinpath(base, fstr)
            push!(
                times, tstr === nothing ? Float64(length(times)) : Base.parse(Float64, tstr)
            )
            push!(piece_files, _expand_pieces(fpath))
        end
    finally
        LightXML.free(doc)
    end
    return times, piece_files
end

"""Expand a `.pvtu` into its leaf `.vtu` `<Piece Source=...>` files; pass `.vtu` through."""
function _expand_pieces(path::AbstractString)
    endswith(path, ".pvtu") || return [path]
    doc = LightXML.parse_file(path)
    pieces = String[]
    try
        r = LightXML.root(doc)
        grids = LightXML.get_elements_by_tagname(r, "PUnstructuredGrid")
        isempty(grids) && throw(ArgumentError("No <PUnstructuredGrid> in .pvtu: $path"))
        base = dirname(path)
        for pc in LightXML.get_elements_by_tagname(first(grids), "Piece")
            src = LightXML.attribute(pc, "Source")
            src === nothing && continue
            push!(pieces, isabspath(src) ? src : joinpath(base, src))
        end
    finally
        LightXML.free(doc)
    end
    return pieces
end

# ---------------------------------------------------------------------------
# Leaf reading (empty-frame guard + ReadVTK)
# ---------------------------------------------------------------------------

function _find_subseq(haystack::Vector{UInt8}, needle::Vector{UInt8})
    nlen = length(needle)
    nlen == 0 && return nothing
    @inbounds for i in 1:(length(haystack) - nlen + 1)
        ok = true
        for j in 1:nlen
            if haystack[i + j - 1] != needle[j]
                ok = false
                break
            end
        end
        ok && return i
    end
    return nothing
end

"""
    _vtu_npoints(path) -> Int

Read `NumberOfPoints` from a leaf `.vtu` header cheaply and *without* ReadVTK, so
0-point frames can be skipped before ReadVTK's compressed reader is invoked (it
throws `ZlibError` on empty appended blocks). Scans only the ASCII header bytes.
"""
function _vtu_npoints(path::AbstractString)
    n = min(filesize(path), 8192)
    head = read(open(path, "r"), n)
    key = Vector{UInt8}("NumberOfPoints=\"")
    idx = _find_subseq(head, key)
    idx === nothing &&
        throw(ArgumentError("Could not locate NumberOfPoints in header of $path"))
    i = idx + length(key)
    digits = UInt8[]
    while i <= length(head) && (head[i] >= UInt8('0') && head[i] <= UInt8('9'))
        push!(digits, head[i])
        i += 1
    end
    return Base.parse(Int, String(digits))
end

# --- Legacy binary VTK (`.vtk`) reader --------------------------------------
# ReadVTK handles only the XML formats + VTKHDF, so legacy `.vtk` files (e.g.
# DualSPHysics `PartAll_*.vtk`: BINARY, big-endian, POLYDATA) get their own
# minimal parser below. Only the POINTS + point-data arrays we need are decoded.

function _vtk_dtype(s::AbstractString)
    s == "float" && return Float32
    s == "double" && return Float64
    s == "int" && return Int32
    s == "unsigned_int" && return UInt32
    s == "long" && return Int64
    s == "unsigned_long" && return UInt64
    s == "short" && return Int16
    s == "unsigned_short" && return UInt16
    (s == "unsigned_char" || s == "char") && return UInt8
    return throw(ArgumentError("Unsupported legacy VTK datatype: $s"))
end

"""Read one keyword line, skipping leading newlines left after a binary blob."""
function _read_line!(bytes::Vector{UInt8}, pos::Int)
    n = length(bytes)
    while pos <= n && (bytes[pos] == UInt8('\n') || bytes[pos] == UInt8('\r'))
        pos += 1
    end
    start = pos
    while pos <= n && bytes[pos] != UInt8('\n')
        pos += 1
    end
    line = rstrip(String(bytes[start:min(pos - 1, n)]))
    pos <= n && (pos += 1)   # consume the trailing '\n' so binary blobs start at `pos`
    return String(line), pos
end

"""Read `count` big-endian values of type `T` starting at `pos`."""
function _read_be!(bytes::Vector{UInt8}, pos::Int, count::Int, ::Type{T}) where {T}
    nb = count * sizeof(T)
    vals = reinterpret(T, bytes[pos:(pos + nb - 1)])
    out = T <: Union{UInt8,Int8} ? Array(vals) : ntoh.(vals)
    return Array(out), pos + nb
end

"""
Minimal reader for legacy BINARY VTK POLYDATA/UNSTRUCTURED_GRID particle files.
Returns `(points::(3,N), names, data::Dict)` like [`_read_leaf`](@ref).
"""
function _read_legacy_vtk(path::AbstractString)
    bytes = read(path)
    pos = 1
    magic, pos = _read_line!(bytes, pos)
    startswith(magic, "# vtk") ||
        throw(ArgumentError("Not a legacy VTK file (bad magic): $path"))
    _title, pos = _read_line!(bytes, pos)
    fmt, pos = _read_line!(bytes, pos)
    fmt == "BINARY" ||
        throw(ArgumentError("Only BINARY legacy VTK is supported (got \"$fmt\"): $path"))
    _dataset, pos = _read_line!(bytes, pos)   # DATASET POLYDATA / UNSTRUCTURED_GRID

    pts = Matrix{Float32}(undef, 3, 0)
    names = String[]
    data = Dict{String,Array}()
    npts = 0
    while pos <= length(bytes)
        line, pos = _read_line!(bytes, pos)
        isempty(line) && continue
        toks = split(line)
        kw = toks[1]
        if kw == "POINTS"
            npts = Base.parse(Int, toks[2])
            vals, pos = _read_be!(bytes, pos, npts * 3, _vtk_dtype(toks[3]))
            pts = Float32.(reshape(vals, 3, npts))
        elseif kw in ("VERTICES", "LINES", "POLYGONS", "TRIANGLE_STRIPS", "CELLS")
            _, pos = _read_be!(bytes, pos, Base.parse(Int, toks[3]), Int32)   # skip topology
        elseif kw == "POINT_DATA"
            npts = Base.parse(Int, toks[2])
        elseif kw == "SCALARS"
            name = toks[2]
            ncomp = length(toks) >= 4 ? Base.parse(Int, toks[4]) : 1
            _lut, pos = _read_line!(bytes, pos)                               # LOOKUP_TABLE line
            vals, pos = _read_be!(bytes, pos, npts * ncomp, _vtk_dtype(toks[3]))
            data[name] = ncomp == 1 ? vals : reshape(vals, ncomp, npts)
            push!(names, name)
        elseif kw in ("VECTORS", "NORMALS")
            name = toks[2]
            vals, pos = _read_be!(bytes, pos, npts * 3, _vtk_dtype(toks[3]))
            data[name] = reshape(vals, 3, npts)
            push!(names, name)
        elseif kw == "FIELD"
            for _ in 1:Base.parse(Int, toks[3])
                aline, pos = _read_line!(bytes, pos)
                at = split(aline)
                ncomp, ntup = Base.parse(Int, at[2]), Base.parse(Int, at[3])
                vals, pos = _read_be!(bytes, pos, ncomp * ntup, _vtk_dtype(at[4]))
                data[at[1]] = ncomp == 1 ? vals : reshape(vals, ncomp, ntup)
                push!(names, at[1])
            end
        else
            break   # CELL_DATA or anything we don't consume: stop before misreading binary
        end
    end
    return pts, names, data
end

"""Number of points in a leaf, without decoding the payload (empty-frame guard)."""
function _leaf_npoints(path::AbstractString)
    endswith(lowercase(path), ".vtk") || return _vtu_npoints(path)
    n = min(filesize(path), 8192)
    head = read(open(path, "r"), n)
    idx = _find_subseq(head, Vector{UInt8}("POINTS "))
    idx === nothing && throw(ArgumentError("Could not locate POINTS in header of $path"))
    i = idx + 7
    digits = UInt8[]
    while i <= length(head) && (head[i] >= UInt8('0') && head[i] <= UInt8('9'))
        push!(digits, head[i])
        i += 1
    end
    return Base.parse(Int, String(digits))
end

"""Read one leaf piece (`.vtu` via ReadVTK, or legacy `.vtk`): `(points::(3,N), names, data)`."""
function _read_leaf(path::AbstractString)
    endswith(lowercase(path), ".vtk") && return _read_legacy_vtk(path)
    vtk = ReadVTK.VTKFile(path)
    pts = Float32.(Array(ReadVTK.get_points(vtk)))
    pd = ReadVTK.get_point_data(vtk)
    names = collect(keys(pd))
    data = Dict{String,Array}()
    for nm in names
        data[nm] = Array(ReadVTK.get_data(pd[nm]))
    end
    return pts, names, data
end

"""Read + horizontally concatenate all leaf pieces of one timestep."""
function _read_frame(pieces::Vector{String})
    pts_all = Matrix{Float32}(undef, 3, 0)
    data_all = Dict{String,Any}()
    names = String[]
    for p in pieces
        pts, nms, data = _read_leaf(p)
        names = nms
        pts_all = hcat(pts_all, pts)
        for nm in nms
            arr = data[nm]
            if ndims(arr) == 2
                data_all[nm] = haskey(data_all, nm) ? hcat(data_all[nm], arr) : arr
            else
                data_all[nm] = haskey(data_all, nm) ? vcat(data_all[nm], arr) : arr
            end
        end
    end
    return pts_all, names, data_all
end

# ---------------------------------------------------------------------------
# Field-name resolution
# ---------------------------------------------------------------------------

function _resolve_field(
    requested::Union{Nothing,AbstractString},
    available::Vector{String};
    role::Symbol,
    required::Bool,
)
    if requested !== nothing
        for a in available
            lowercase(a) == lowercase(requested) && return a
        end
    end
    for alias in get(_VTK_FIELD_ALIASES, role, String[])
        for a in available
            lowercase(a) == alias && return a
        end
    end
    if required
        throw(
            ArgumentError(
                "Could not resolve the $role field" *
                (requested === nothing ? "" : " (requested \"$requested\")") *
                ". Available PointData arrays: $available. " *
                "Pass the correct name via the corresponding keyword argument.",
            ),
        )
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Trajectory assembly
# ---------------------------------------------------------------------------

"""
Assemble one trajectory (one `source`) into aligned 3D arrays. Returns a
NamedTuple with `position/velocity/acceleration` (`dims × N × T`), `types`,
`dt`, `extras`, and derived `bounds`/`est_radius`/`n_types`.
"""
function _build_trajectory(
    source::AbstractString;
    dims::Vector{Int},
    velocity_field,
    type_field,
    id_field,
    acc_field,
    extra_fields::Vector{String},
    recompute_acc::Bool,
    interpolation_scheme::String,
    dt::Union{Nothing,Real},
)
    times, piece_files = _resolve_source_timesteps(source)

    # Contiguous non-empty prefix: stop at the first 0-point frame.
    kept = Int[]
    for (ti, pieces) in enumerate(piece_files)
        np = sum(_leaf_npoints(p) for p in pieces)
        if np == 0
            break
        end
        push!(kept, ti)
    end
    n_dropped = length(piece_files) - length(kept)
    n_dropped > 0 &&
        @info "vtk_to_hdf5: dropped $n_dropped empty/trailing frame(s) from $(basename(source))"
    length(kept) >= 2 ||
        throw(ArgumentError("Need >= 2 non-empty frames in $source, found $(length(kept))"))

    T = length(kept)
    n_dims = length(dims)

    # Resolve field roles from the first kept frame.
    pts0, names0, _ = _read_frame(piece_files[kept[1]])
    N = size(pts0, 2)
    vel_name = _resolve_field(velocity_field, names0; role=:velocity, required=true)
    type_name = _resolve_field(type_field, names0; role=:type, required=true)
    id_name = _resolve_field(id_field, names0; role=:id, required=false)
    acc_name = if recompute_acc
        nothing
    else
        _resolve_field(acc_field, names0; role=:acceleration, required=false)
    end
    extra_names = String[
        _resolve_field(e, names0; role=:extra, required=true) for e in extra_fields
    ]

    position = Array{Float32,3}(undef, n_dims, N, T)
    velocity = Array{Float32,3}(undef, n_dims, N, T)
    acceleration_read =
        acc_name === nothing ? nothing : Array{Float32,3}(undef, n_dims, N, T)
    extras_read = Dict(e => Array{Float32,3}(undef, 1, N, T) for e in extra_names)
    types = Int[]

    for (k, ti) in enumerate(kept)
        pts, _, data = _read_frame(piece_files[ti])
        size(pts, 2) == N || throw(
            ArgumentError(
                "Particle count changes across frames ($(size(pts,2)) vs $N at frame $ti). " *
                "Variable-count trajectories are not yet supported — trim the source to a " *
                "constant-count window.",
            ),
        )
        perm = id_name === nothing ? (1:N) : sortperm(vec(data[id_name]))

        position[:, :, k] = pts[dims, perm]
        velocity[:, :, k] = data[vel_name][dims, perm]
        acc_name === nothing || (acceleration_read[:, :, k] = data[acc_name][dims, perm])
        for e in extra_names
            extras_read[e][1, :, k] = vec(data[e])[perm]
        end
        if k == 1
            types = Int.(vec(data[type_name])[perm])
        end
    end

    # dt: provided wins; else infer from the kept timesteps.
    dt_val = if dt !== nothing
        Float64(dt)
    elseif length(kept) >= 2
        Float64(sum(diff(times[kept])) / (length(kept) - 1))
    else
        1.0
    end

    # Acceleration: read directly, or (re)compute from velocity/position.
    if acc_name !== nothing
        acceleration = acceleration_read
    else
        slice = _get_acceleration_slice(T, interpolation_scheme)
        Tacc = length(slice)
        acceleration = Array{Float32,3}(undef, n_dims, N, Tacc)
        for p in 1:N
            pos_data = [Float64.(position[d, p, :]) for d in 1:n_dims]
            vel_data = [Float64.(velocity[d, p, :]) for d in 1:n_dims]
            acc_p = _calculate_acceleration(
                pos_data, vel_data, dt_val, interpolation_scheme
            )
            for d in 1:n_dims
                acceleration[d, p, :] = Float32.(acc_p[d][1:Tacc])
            end
        end
        position = position[:, :, slice]
        velocity = velocity[:, :, slice]
        for e in extra_names
            extras_read[e] = extras_read[e][:, :, slice]
        end
    end

    bounds = [
        [Float64(minimum(position[d, :, :])), Float64(maximum(position[d, :, :]))] for
        d in 1:n_dims
    ]
    extent = [b[2] - b[1] for b in bounds]
    dx = (prod(extent) / N)^(1 / n_dims)
    est_radius = 3 * dx

    return (
        position=position,
        velocity=velocity,
        acceleration=acceleration,
        types=types,
        dt=dt_val,
        extras=extras_read,
        bounds=bounds,
        est_radius=est_radius,
        n_types=length(unique(types)),
    )
end

# ---------------------------------------------------------------------------
# meta.json
# ---------------------------------------------------------------------------

function _install_provided_meta(meta_path::AbstractString, output::AbstractString)
    isfile(meta_path) ||
        throw(ArgumentError("Provided meta.json does not exist: $meta_path"))
    dest = joinpath(dirname(abspath(output)), "meta.json")
    # Light structural check: every dynamic feature key `X[$t]` must have `X[1]`
    # in the written file, every static feature key must exist.
    m = JSON.parsefile(meta_path)
    h5open(output, "r") do fid
        g = fid[first(keys(fid))]
        for (fn, spec) in get(m, "features", Dict())
            key = get(spec, "key", nothing)
            key === nothing && continue
            probe = occursin("\$t", key) ? replace(key, "\$t" => "1") : key
            haskey(g, probe) ||
                @warn "Provided meta.json references \"$probe\" (feature \"$fn\") which is absent from $output"
        end
    end
    abspath(meta_path) == dest || cp(meta_path, dest; force=true)
    @info "vtk_to_hdf5: installed provided meta.json at $dest"
    return dest
end

function _feature_meanstd(getarr, trajs, n_dims)
    means = zeros(Float64, n_dims)
    stds = ones(Float64, n_dims)
    for d in 1:n_dims
        vals = reduce(vcat, [vec(getarr(t)[d, :, :]) for t in trajs])
        means[d] = Statistics.mean(vals)
        stds[d] = Statistics.std(vals)
    end
    return round.(means; digits=6), round.(stds; digits=6)
end

"""Emit a fully-populated *skeleton* meta.json (Tier B) with computed stats."""
function _write_meta_skeleton(
    output::AbstractString,
    trajs,
    n_dims::Int,
    connectivity_radius::Union{Nothing,Real},
    has_extras::Vector{String},
)
    pos_mean, pos_std = _feature_meanstd(t -> t.position, trajs, n_dims)
    vel_mean, vel_std = _feature_meanstd(t -> t.velocity, trajs, n_dims)
    acc_mean, acc_std = _feature_meanstd(t -> t.acceleration, trajs, n_dims)
    n_types = maximum(t -> t.n_types, trajs)

    bounds = [
        [minimum(t -> t.bounds[d][1], trajs), maximum(t -> t.bounds[d][2], trajs)] for
        d in 1:n_dims
    ]
    lengths = unique(size(t.acceleration, 3) for t in trajs)
    tl = length(lengths) == 1 ? only(lengths) : "trajectory_length"
    rc =
        connectivity_radius === nothing ? trajs[1].est_radius : Float64(connectivity_radius)
    if connectivity_radius === nothing
        @warn "vtk_to_hdf5: default_connectivity_radius was ESTIMATED as $(round(rc; sigdigits=4)) " *
            "from mean particle spacing — set it explicitly for real training."
    end

    feats = Dict{String,Any}(
        "node_type" => Dict(
            "key" => "type",
            "dtype" => "int32",
            "dim" => 1,
            "onehot" => true,
            "type" => "static",
            "data_min" => 1,
            "data_max" => n_types,
        ),
        "position" => Dict(
            "key" => "pos[\$t]",
            "dtype" => "float32",
            "dim" => n_dims,
            "type" => "dynamic",
            "data_mean" => pos_mean,
            "data_std" => pos_std,
        ),
        "velocity" => Dict(
            "key" => "vel[\$t]",
            "dtype" => "float32",
            "dim" => n_dims,
            "type" => "dynamic",
            "data_mean" => vel_mean,
            "data_std" => vel_std,
        ),
        "acceleration" => Dict(
            "key" => "acc[\$t]",
            "dtype" => "float32",
            "dim" => n_dims,
            "type" => "dynamic",
            "data_mean" => acc_mean,
            "data_std" => acc_std,
        ),
    )
    feature_names = ["node_type", "position", "velocity", "acceleration"]
    for e in has_extras
        emean, estd = _feature_meanstd(t -> t.extras[e], trajs, 1)
        feats[e] = Dict(
            "key" => "$e[\$t]",
            "dtype" => "float32",
            "dim" => 1,
            "type" => "dynamic",
            "data_mean" => emean,
            "data_std" => estd,
        )
        push!(feature_names, e)
    end

    meta = Dict{String,Any}(
        "feature_names" => feature_names,
        "features" => feats,
        "input_features" => ["velocity"],
        "output_features" => ["acceleration"],
        "target_features" => ["acceleration", "velocity"],
        "derivative_target_features" => ["acceleration"],
        "solver_target_features" => ["position"],
        "dt" => "dt",
        "n_particles" => "n_particles",
        "trajectory_length" => tl,
        "n_trajectories" => length(trajs),
        "subgroups" => false,
        "dims" => n_dims,
        "bounds" => bounds,
        "default_connectivity_radius" => rc,
    )
    dest = joinpath(dirname(abspath(output)), "meta.json")
    open(dest, "w") do f
        JSON.print(f, meta, 2)
    end
    @info "vtk_to_hdf5: wrote skeleton meta.json at $dest (review feature roles & connectivity radius)"
    return dest
end

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    vtk_to_hdf5(sources, output::String; kwargs...)

Convert VTK particle trajectories into the HDF5 layout GraphNetSim trains on.
Each entry in `sources` becomes one trajectory group (`trajectory_1`,
`trajectory_2`, …) in `output`. Supported inputs:

- ParaView / DualSPHysics XML (`.pvd` collections, `.pvtu` parallel wrappers, `.vtu` pieces), and
- legacy **BINARY** `.vtk` (`POLYDATA`, e.g. DualSPHysics `PartAll_*.vtk`).

The `.pvd`/`.pvtu` wrappers are parsed internally; leaf `.vtu` pieces are decoded
via ReadVTK while legacy `.vtk` files use a built-in reader (ReadVTK handles
neither the wrappers nor legacy `.vtk`). Empty (0-point) frames are skipped and
particles are kept row-aligned across time by sorting on `id_field` when present.

## Arguments
- `sources`: a single `.pvd`, a directory (of `.pvtu`/`.vtu`/`.vtk`), a `.pvtu`/`.vtu`/`.vtk`,
  or a vector of such (one per trajectory). A directory's leaves are ordered lexicographically.
- `output::String`: destination `.h5` file.

## Keyword Arguments
- `dims::Vector{Int}=[1,2,3]`: which of the 3 VTK point components to keep.
- `velocity_field="Vel"`, `type_field="phaseID"`, `id_field="prtlID"`, `acc_field="Acc"`:
  PointData array names for each role; auto-detected from common aliases when absent.
- `extra_fields::Vector{String}=String[]`: extra scalar PointData arrays to copy through.
- `recompute_acc::Bool=false`: derive acceleration from velocity/position instead of reading
  `acc_field`. Recomputation also happens automatically when no acceleration field is present
  (e.g. legacy `.vtk`).
- `interpolation_scheme::String="pchip"`: scheme used when (re)computing acceleration (see [`csv_to_hdf5`](@ref)).
- `dt::Union{Nothing,Real}=nothing`: fixed timestep; inferred from `.pvd` timesteps when `nothing`.
  A directory/glob source carries no timestamps, so pass `dt` explicitly there (it defaults to 1.0).
- `meta::Union{Nothing,String}=nothing`: path to an existing meta.json to validate + copy next to `output` (Tier A).
- `write_meta::Bool=false`: emit a computed skeleton meta.json instead (Tier B).
- `connectivity_radius::Union{Nothing,Real}=nothing`: value for the skeleton meta (estimated + warned when `nothing`).
"""
function vtk_to_hdf5(
    sources,
    output::String;
    dims::Vector{Int}=[1, 2, 3],
    velocity_field::Union{Nothing,AbstractString}="Vel",
    type_field::Union{Nothing,AbstractString}="phaseID",
    id_field::Union{Nothing,AbstractString}="prtlID",
    acc_field::Union{Nothing,AbstractString}="Acc",
    extra_fields::Vector{<:AbstractString}=String[],
    recompute_acc::Bool=false,
    interpolation_scheme::String="pchip",
    dt::Union{Nothing,Real}=nothing,
    meta::Union{Nothing,AbstractString}=nothing,
    write_meta::Bool=false,
    connectivity_radius::Union{Nothing,Real}=nothing,
)
    all(d in 1:3 for d in dims) ||
        throw(ArgumentError("dims must contain values in [1, 2, 3]"))
    isempty(dims) && throw(ArgumentError("dims must not be empty"))

    src_list = sources isa AbstractString ? [sources] : collect(sources)
    isempty(src_list) && throw(ArgumentError("sources is empty"))
    extras = String.(extra_fields)

    mkpath(dirname(abspath(output)))
    trajs = Any[]
    fid = h5open(output, "w")
    try
        for (i, src) in enumerate(src_list)
            traj = _build_trajectory(
                src;
                dims=dims,
                velocity_field=velocity_field,
                type_field=type_field,
                id_field=id_field,
                acc_field=acc_field,
                extra_fields=extras,
                recompute_acc=recompute_acc,
                interpolation_scheme=interpolation_scheme,
                dt=dt,
            )
            _write_trajectory!(
                fid,
                i,
                traj.position,
                traj.velocity,
                traj.acceleration,
                traj.types,
                traj.dt;
                extras=traj.extras,
            )
            push!(trajs, traj)
        end
    finally
        close(fid)
    end

    if meta !== nothing
        _install_provided_meta(meta, output)
    elseif write_meta
        _write_meta_skeleton(output, trajs, length(dims), connectivity_radius, extras)
    end

    return output
end
