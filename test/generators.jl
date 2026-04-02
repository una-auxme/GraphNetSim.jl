#
# Dataset generator modules for ballistic_small and dam_break_small.
# Each generator is wrapped in its own module to avoid const-name collisions.
#
# Provides:
#   _needs_generation(dir)         — true if any of the 4 dataset files are missing
#   _GenBallistic.generate(outdir) — generate ballistic_small into outdir
#   _GenDamBreak.generate(outdir)  — generate dam_break_small into outdir
#
# This file only defines — it does NOT run any generation.
# Callers decide where and when to generate.
#

using HDF5, JSON, Statistics, LinearAlgebra, Random

function _needs_generation(dir)
    for f in ["train.h5", "valid.h5", "test.h5", "meta.json"]
        isfile(joinpath(dir, f)) || return true
    end
    return false
end

# ============================================================================
# Ballistic dataset generator
# ============================================================================
module _GenBallistic

using HDF5, JSON, Statistics, Random

const N_PARTICLES = 10
const T_LENGTH    = 67
const DT          = 0.002f0
const G           = 9.81f0
const GAMMA       = 0.5f0
const CR          = 0.00004f0

const N_TRAIN = 5
const N_VALID = 2
const N_TEST  = 2

function simulate(seed::Int)
    rng = MersenneTwister(seed)

    cx  = randn(rng, Float32) * 0.10f0
    cy  = randn(rng, Float32) * 0.05f0
    cz  = randn(rng, Float32) * 0.05f0
    cvx = 5.0f0 + randn(rng, Float32) * 2.0f0
    cvy = randn(rng, Float32) * 1.0f0
    cvz = randn(rng, Float32) * 1.0f0

    px0 = cx .+ randn(rng, Float32, N_PARTICLES) .* 1.0f-5
    py0 = cy .+ randn(rng, Float32, N_PARTICLES) .* 1.0f-5
    pz0 = cz .+ randn(rng, Float32, N_PARTICLES) .* 1.0f-5

    vx0 = cvx .+ randn(rng, Float32, N_PARTICLES) .* 1.0f-3
    vy0 = cvy .+ randn(rng, Float32, N_PARTICLES) .* 1.0f-3
    vz0 = cvz .+ randn(rng, Float32, N_PARTICLES) .* 1.0f-3

    pos = zeros(Float32, 3, N_PARTICLES, T_LENGTH)
    vel = zeros(Float32, 3, N_PARTICLES, T_LENGTH)
    acc = zeros(Float32, 3, N_PARTICLES, T_LENGTH)

    for t in 1:T_LENGTH
        τ = (t - 1) * DT
        e = exp(-GAMMA * τ)

        pos[1, :, t] = px0 .+ vx0 ./ GAMMA .* (1.0f0 .- e)
        pos[2, :, t] = py0 .+ vy0 ./ GAMMA .* (1.0f0 .- e)
        pos[3, :, t] = pz0 .+ (vz0 .+ G / GAMMA) ./ GAMMA .* (1.0f0 .- e) .- G / GAMMA .* τ

        vel[1, :, t] = vx0 .* e
        vel[2, :, t] = vy0 .* e
        vel[3, :, t] = (vz0 .+ G / GAMMA) .* e .- G / GAMMA

        acc[1, :, t] = -GAMMA .* vel[1, :, t]
        acc[2, :, t] = -GAMMA .* vel[2, :, t]
        acc[3, :, t] = -G .- GAMMA .* vel[3, :, t]
    end

    return pos, vel, acc
end

function write_split(path::String, seeds::Vector{Int})
    h5open(path, "w") do f
        for (i, seed) in enumerate(seeds)
            pos, vel, acc = simulate(seed)
            g = create_group(f, "trajectory_$i")
            g["dt"] = DT
            g["n_particles"] = Int64(N_PARTICLES)
            g["type"] = Int32.(ones(Int32, N_PARTICLES))
            for t in 1:T_LENGTH
                g["pos[$t]"] = pos[:, :, t]
                g["vel[$t]"] = vel[:, :, t]
                g["acc[$t]"] = acc[:, :, t]
            end
        end
    end
end

function compute_stats(seeds::Vector{Int})
    all_pos = zeros(Float32, 3, N_PARTICLES * T_LENGTH * length(seeds))
    all_vel = zeros(Float32, 3, N_PARTICLES * T_LENGTH * length(seeds))
    all_acc = zeros(Float32, 3, N_PARTICLES * T_LENGTH * length(seeds))

    offset = 0
    for seed in seeds
        pos, vel, acc = simulate(seed)
        n = N_PARTICLES * T_LENGTH
        all_pos[:, offset+1:offset+n] = reshape(pos, 3, :)
        all_vel[:, offset+1:offset+n] = reshape(vel, 3, :)
        all_acc[:, offset+1:offset+n] = reshape(acc, 3, :)
        offset += n
    end

    return (
        pos_mean=mean(all_pos; dims=2)[:, 1],
        pos_std=std(all_pos; dims=2)[:, 1],
        vel_mean=mean(all_vel; dims=2)[:, 1],
        vel_std=std(all_vel; dims=2)[:, 1],
        acc_mean=mean(all_acc; dims=2)[:, 1],
        acc_std=std(all_acc; dims=2)[:, 1],
    )
end

function generate(outdir::String)
    mkpath(outdir)
    train_seeds = collect(1:N_TRAIN)
    valid_seeds = collect(100:100:N_VALID * 100)
    test_seeds = collect(200:100:N_TEST * 100)

    write_split(joinpath(outdir, "train.h5"), train_seeds)
    write_split(joinpath(outdir, "valid.h5"), valid_seeds)
    write_split(joinpath(outdir, "test.h5"), test_seeds)

    s = compute_stats(train_seeds)

    meta = Dict(
        "dt" => "dt",
        "n_particles" => "n_particles",
        "bounds" => [[-10, 10], [-10, 10], [-10, 10]],
        "trajectory_length" => T_LENGTH,
        "n_trajectories" => N_TRAIN,
        "default_connectivity_radius" => Float64(CR),
        "subgroups" => false,
        "dims" => 3,
        "feature_names" => ["node_type", "position", "velocity", "acceleration"],
        "target_features" => ["acceleration", "velocity"],
        "derivative_target_features" => ["acceleration"],
        "input_features" => ["velocity"],
        "output_features" => ["acceleration"],
        "solver_target_features" => ["position"],
        "features" => Dict(
            "node_type" => Dict(
                "type" => "static", "key" => "type", "dim" => 1,
                "dtype" => "int32", "onehot" => true,
                "data_min" => 1, "data_max" => 1,
            ),
            "position" => Dict(
                "key" => "pos[\$t]", "dim" => 3, "dtype" => "float32",
                "type" => "dynamic",
                "data_mean" => round.(Float64.(s.pos_mean); digits=6),
                "data_std" => round.(Float64.(s.pos_std); digits=6),
            ),
            "velocity" => Dict(
                "key" => "vel[\$t]", "dim" => 3, "dtype" => "float32",
                "type" => "dynamic",
                "data_mean" => round.(Float64.(s.vel_mean); digits=6),
                "data_std" => round.(Float64.(s.vel_std); digits=6),
            ),
            "acceleration" => Dict(
                "key" => "acc[\$t]", "dim" => 3, "dtype" => "float32",
                "type" => "dynamic",
                "data_mean" => round.(Float64.(s.acc_mean); digits=6),
                "data_std" => round.(Float64.(s.acc_std); digits=6),
            ),
        ),
    )

    open(joinpath(outdir, "meta.json"), "w") do io
        JSON.print(io, meta, 4)
    end
end

end  # module _GenBallistic

# ============================================================================
# Dam break dataset generator
# ============================================================================
module _GenDamBreak

using HDF5, JSON, Statistics, LinearAlgebra, Random

const DP     = 0.03f0
const H_SPH  = 1.2f0 * DP
const CR     = 2.0f0 * H_SPH
const RHO0   = 1000.0f0
const G_GRAV = 9.81f0
const ALPHA  = 0.1f0
const C_SND  = 10.0f0 * sqrt(2.0f0 * G_GRAV * 3.0f0 * DP)
const B_EOS  = RHO0 * C_SND^2 / 7.0f0
const PART_M = RHO0 * DP^2

const DT       = 0.001f0
const T_LENGTH = 150

const N_TRAIN = 4
const N_VALID = 2
const N_TEST  = 2

const NX_FLOOR = 11
const NY_WALL  = 4
const N_LAYERS = 2
const N_FLOOR  = NX_FLOOR * N_LAYERS        # 22
const N_LWALL  = N_LAYERS * NY_WALL          # 8
const N_RWALL  = N_LAYERS * NY_WALL          # 8
const N_BND    = N_FLOOR + N_LWALL + N_RWALL # 38
const N_FLD    = 9
const N_TOT    = N_BND + N_FLD               # 47

function W_kernel(r::Float32, h::Float32)::Float32
    q = r / h
    σ = 10.0f0 / (7.0f0 * Float32(π) * h^2)
    if q < 1.0f0
        return σ * (1.0f0 - 1.5f0 * q^2 + 0.75f0 * q^3)
    elseif q < 2.0f0
        return σ * 0.25f0 * (2.0f0 - q)^3
    else
        return 0.0f0
    end
end

function ∇W_kernel(dr::AbstractVector{Float32}, r::Float32, h::Float32)
    r < 1.0f-12 && return zeros(Float32, 2)
    q = r / h
    σ = 10.0f0 / (7.0f0 * Float32(π) * h^2)
    if q < 1.0f0
        dWdq = σ * (-3.0f0 * q + 2.25f0 * q^2)
    elseif q < 2.0f0
        dWdq = σ * (-0.75f0 * (2.0f0 - q)^2)
    else
        return zeros(Float32, 2)
    end
    return (dWdq / h) .* (dr ./ r)
end

function compute_density(pos::Matrix{Float32}, types::Vector{Int32})
    N = size(pos, 2)
    rho = zeros(Float32, N)
    for i in 1:N
        for j in 1:N
            dr = pos[:, i] .- pos[:, j]
            r = Float32(norm(dr))
            r < CR && (rho[i] += PART_M * W_kernel(r, H_SPH))
        end
        types[i] == 1 && (rho[i] = RHO0)
    end
    return rho
end

function compute_pressure(rho::Vector{Float32})
    return B_EOS .* (max.(rho ./ RHO0, 1.0f0) .^ 7 .- 1.0f0)
end

function compute_acceleration(
    pos::Matrix{Float32},
    vel::Matrix{Float32},
    types::Vector{Int32},
    rho::Vector{Float32},
    p::Vector{Float32},
)
    N = size(pos, 2)
    acc = zeros(Float32, 2, N)

    for i in 1:N
        types[i] == 1 && continue
        acc[2, i] = -G_GRAV

        for j in 1:N
            i == j && continue
            dr = pos[:, i] .- pos[:, j]
            r = Float32(norm(dr))
            r >= CR && continue

            ∇W = ∇W_kernel(dr, r, H_SPH)

            p_term = p[i] / rho[i]^2
            types[j] == 2 && (p_term += p[j] / rho[j]^2)
            acc[:, i] .-= PART_M .* p_term .* ∇W

            if types[j] == 2
                dv = vel[:, i] .- vel[:, j]
                v_dot_r = dot(dv, dr)
                if v_dot_r < 0.0f0
                    mu = H_SPH * v_dot_r / (r^2 + 0.01f0 * H_SPH^2)
                    Π = -ALPHA * C_SND * mu / (0.5f0 * (rho[i] + rho[j]))
                    acc[:, i] .-= PART_M .* Π .* ∇W
                end
            end
        end
    end
    return acc
end

function init_positions(seed::Int)
    rng = MersenneTwister(seed)
    pos = zeros(Float32, 2, N_TOT)
    idx = 1

    # Floor — 2 layers at y = {0, DP}, x = {0, DP, ..., 10*DP}
    for ly in 0:(N_LAYERS - 1), ix in 0:(NX_FLOOR - 1)
        pos[1, idx] = Float32(ix) * DP
        pos[2, idx] = Float32(ly) * DP
        idx += 1
    end

    # Left wall — 2 layers at x = {0, DP}, y = {2*DP, ..., 5*DP}
    for lx in 0:(N_LAYERS - 1), iy in 1:NY_WALL
        pos[1, idx] = Float32(lx) * DP
        pos[2, idx] = Float32(iy + 1) * DP
        idx += 1
    end

    # Right wall — 2 layers at x = {9*DP, 10*DP}, y = {2*DP, ..., 5*DP}
    for lx in 0:(N_LAYERS - 1), iy in 1:NY_WALL
        pos[1, idx] = Float32(NX_FLOOR - 2 + lx) * DP
        pos[2, idx] = Float32(iy + 1) * DP
        idx += 1
    end

    # Fluid 3×3 grid with small per-trajectory horizontal shift
    dx = randn(rng, Float32) * 2.0f-3
    for iy in 1:3, ix in 1:3
        pos[1, idx] = Float32(ix + 1) * DP + dx
        pos[2, idx] = Float32(iy + 1) * DP
        idx += 1
    end
    return pos
end

function simulate(seed::Int)
    types = Int32.(vcat(fill(Int32(1), N_BND), fill(Int32(2), N_FLD)))
    pos = init_positions(seed)
    vel = zeros(Float32, 2, N_TOT)
    bnd_pos = copy(pos[:, 1:N_BND])

    out_pos = zeros(Float32, 2, N_TOT, T_LENGTH)
    out_vel = zeros(Float32, 2, N_TOT, T_LENGTH)
    out_acc = zeros(Float32, 2, N_TOT, T_LENGTH)

    for t in 1:T_LENGTH
        rho = compute_density(pos, types)
        p = compute_pressure(rho)
        acc = compute_acceleration(pos, vel, types, rho, p)

        out_pos[:, :, t] = pos
        out_vel[:, :, t] = vel
        out_acc[:, :, t] = acc

        vel_new = vel .+ DT .* acc
        pos_new = pos .+ DT .* vel_new

        vel_new[:, 1:N_BND] .= 0.0f0
        pos_new[:, 1:N_BND] = bnd_pos

        pos = pos_new
        vel = vel_new
    end

    return out_pos, out_vel, out_acc, types
end

function write_split(path::String, seeds::Vector{Int})
    h5open(path, "w") do f
        for (i, seed) in enumerate(seeds)
            pos, vel, acc, types = simulate(seed)
            g = create_group(f, "trajectory_$i")
            g["dt"] = DT
            g["n_particles"] = Int64(N_TOT)
            g["type"] = types
            for t in 1:T_LENGTH
                g["pos[$t]"] = pos[:, :, t]
                g["vel[$t]"] = vel[:, :, t]
                g["acc[$t]"] = acc[:, :, t]
            end
        end
    end
end

function compute_stats(seeds::Vector{Int})
    n = N_TOT * T_LENGTH * length(seeds)
    all_pos = zeros(Float32, 2, n)
    all_vel = zeros(Float32, 2, n)
    all_acc = zeros(Float32, 2, n)
    offset = 0
    for seed in seeds
        pos, vel, acc, _ = simulate(seed)
        m = N_TOT * T_LENGTH
        all_pos[:, offset+1:offset+m] = reshape(pos, 2, :)
        all_vel[:, offset+1:offset+m] = reshape(vel, 2, :)
        all_acc[:, offset+1:offset+m] = reshape(acc, 2, :)
        offset += m
    end
    return (
        pos_mean=mean(all_pos; dims=2)[:, 1],
        pos_std=std(all_pos; dims=2)[:, 1],
        vel_mean=mean(all_vel; dims=2)[:, 1],
        vel_std=std(all_vel; dims=2)[:, 1],
        acc_mean=mean(all_acc; dims=2)[:, 1],
        acc_std=std(all_acc; dims=2)[:, 1],
    )
end

function generate(outdir::String)
    mkpath(outdir)
    train_seeds = collect(1:N_TRAIN)
    valid_seeds = collect(100:100:(N_VALID * 100))
    test_seeds = collect(200:100:(N_TEST * 100))

    write_split(joinpath(outdir, "train.h5"), train_seeds)
    write_split(joinpath(outdir, "valid.h5"), valid_seeds)
    write_split(joinpath(outdir, "test.h5"), test_seeds)

    s = compute_stats(train_seeds)

    meta = Dict(
        "dt" => "dt",
        "n_particles" => "n_particles",
        "bounds" => [[0, Float64(NX_FLOOR - 1) * Float64(DP)],
                      [0, Float64(NY_WALL + 1) * Float64(DP)]],
        "trajectory_length" => T_LENGTH,
        "n_trajectories" => N_TRAIN,
        "default_connectivity_radius" => Float64(CR),
        "subgroups" => false,
        "dims" => 2,
        "feature_names" => ["node_type", "position", "velocity", "acceleration"],
        "target_features" => ["acceleration", "velocity"],
        "derivative_target_features" => ["acceleration"],
        "input_features" => ["velocity"],
        "output_features" => ["acceleration"],
        "solver_target_features" => ["position"],
        "features" => Dict(
            "node_type" => Dict(
                "type" => "static", "key" => "type", "dim" => 1,
                "dtype" => "int32", "onehot" => true,
                "data_min" => 1, "data_max" => 2,
            ),
            "position" => Dict(
                "key" => "pos[\$t]", "dim" => 2, "dtype" => "float32",
                "type" => "dynamic",
                "data_mean" => round.(Float64.(s.pos_mean); digits=6),
                "data_std" => round.(Float64.(s.pos_std); digits=6),
            ),
            "velocity" => Dict(
                "key" => "vel[\$t]", "dim" => 2, "dtype" => "float32",
                "type" => "dynamic",
                "data_mean" => round.(Float64.(s.vel_mean); digits=6),
                "data_std" => round.(Float64.(s.vel_std); digits=6),
            ),
            "acceleration" => Dict(
                "key" => "acc[\$t]", "dim" => 2, "dtype" => "float32",
                "type" => "dynamic",
                "data_mean" => round.(Float64.(s.acc_mean); digits=6),
                "data_std" => round.(Float64.(s.acc_std); digits=6),
            ),
        ),
    )

    open(joinpath(outdir, "meta.json"), "w") do io
        JSON.print(io, meta, 4)
    end
end

end  # module _GenDamBreak
