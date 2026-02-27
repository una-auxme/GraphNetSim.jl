#
# Copyright (c) 2026 Josef Kircher, Julian Trommer
# Licensed under the MIT license. See LICENSE file in the project root for details.
#   

using HDF5
using CSV
using DataFrames
using DataInterpolations

"""
    csv_to_hdf5(source::String, output::String; 
                 dt::Float64=0.01, 
                 n_trajectories::Int=1,
                 dims::Vector{Int}=[1, 2],
                 groupby_col::Symbol=:Idp,
                 interpolation_scheme::String="pchip",
                 pos_col_prefix::String="Points",
                 vel_col_prefix::String="Vel",
                 type_col::Symbol=:Type,
                 extra_fields::Vector{Symbol}=Symbol[])

Convert particle trajectory data from CSV to HDF5 format with computed accelerations.

## Arguments
- `source::String`: Path to input CSV file
- `output::String`: Path to output HDF5 file

## Keyword Arguments
- `dt::Float64`: Time step (default: 0.01)
- `n_trajectories::Int`: Number of trajectories to process (default: 1)
- `dims::Vector{Int}`: Spatial dimensions to extract, e.g. [1, 2] or [1, 3] (default: [1, 2])
- `groupby_col::Symbol`: Column name for grouping particles (default: :Idp)
- `interpolation_scheme::String`: Acceleration calculation method (default: "pchip")
  - "central_diff": Central difference scheme
  - "forward_diff": Forward difference scheme
  - "backward_diff": Backward difference scheme
  - "from_pos": Acceleration from position (2nd order central difference)
  - "pchip": PCHIP interpolation of velocity derivatives
  - "linear": LinearInterpolation
  - "quadratic": QuadraticInterpolation
  - "cubic_spline": CubicSpline
  - "quadratic_spline": QuadraticSpline
  - "cubic_hermite": CubicHermiteSpline
  - "lagrange": LagrangeInterpolation
  - "akima": AkimaInterpolation
- `pos_col_prefix::String`: Prefix for position columns (default: "Points")
- `vel_col_prefix::String`: Prefix for velocity columns (default: "Vel")
- `type_col::Symbol`: Column name for particle type (default: :Type)
- `extra_fields::Vector{Symbol}`: Additional CSV columns to copy to HDF5 
  (e.g. [:Mass, :Temperature])

## Example
```julia
# 3D simulation with PCHIP interpolation
csv_to_hdf5("data/dam_break.csv", "data/dam_break_first.h5"; 
            dt=0.01, dims=[1, 2, 3], interpolation_scheme="pchip")

# 2D (skip y-dimension), with extra fields
csv_to_hdf5("data/input.csv", "output.h5"; 
            dims=[1, 3], interpolation_scheme="cubic_spline",
            extra_fields=[:Mass, :Pressure])
```
"""
function csv_to_hdf5(source::String, output::String; 
                     dt::Float64=0.01, 
                     n_trajectories::Int=1,
                     dims::Vector{Int}=[1, 2],
                     groupby_col::Symbol=:Idp,
                     interpolation_scheme::String="pchip",
                     pos_col_prefix::String="Points",
                     vel_col_prefix::String="Vel",
                     type_col::Symbol=:Type,
                     extra_fields::Vector{Symbol}=Symbol[])
    
    # Validate inputs
    all(d ∈ [1, 2, 3] for d in dims) || throw(ArgumentError("dims must contain values in [1, 2, 3]"))
    length(dims) > 0 || throw(ArgumentError("dims must not be empty"))
    
    valid_schemes = ["central_diff", "forward_diff", "backward_diff", "from_pos", 
                     "pchip", "linear", "quadratic", "cubic_spline", "quadratic_spline",
                     "cubic_hermite", "lagrange", "akima"]
    interpolation_scheme in valid_schemes ||
        throw(ArgumentError("Invalid interpolation_scheme: $interpolation_scheme. Valid options: $valid_schemes"))
    
    # Read CSV file
    data_df = DataFrame(CSV.File(source))
    
    # Group by particle ID
    grp = groupby(data_df, groupby_col)
    
    # Open HDF5 file for writing
    fid = h5open(output, "w")
    
    try
        for traj_id = 1:n_trajectories
            tra = "trajectory_$traj_id"
            create_group(fid, tra)
            fid["$tra/n_particles"] = length(grp)
            fid["$tra/dt"] = dt
            
            # Store trajectory length and initialize arrays (determined from first particle)
            trajectory_length = 0
            position_arrays = nothing
            velocity_arrays = nothing
            acceleration_arrays = nothing
            particle_types = nothing
            
            # Process each particle
            for (j, prtl) in enumerate(grp)
                # Extract velocity data for selected dimensions
                vel_data = [collect(prtl[!, Symbol("$vel_col_prefix:$(d-1)")]) for d in dims]
                
                # Extract position data for selected dimensions
                pos_data = [collect(prtl[!, Symbol("$pos_col_prefix:$(d-1)")]) for d in dims]
                
                # Calculate acceleration based on selected scheme
                acc_data = _calculate_acceleration(pos_data, vel_data, dt, interpolation_scheme)
                
                # Get the slice indices for acceleration (may be shorter than input)
                acc_slice = _get_acceleration_slice(length(vel_data[1]), interpolation_scheme)
                
                # Apply slice to all data arrays for consistency
                pos_data_sliced = [pos_data[i][acc_slice] for i in 1:length(pos_data)]
                vel_data_sliced = [vel_data[i][acc_slice] for i in 1:length(vel_data)]
                acc_data_final = [acc_data[i][1:length(pos_data_sliced[1])] for i in 1:length(acc_data)]
                
                # Set trajectory length and initialize arrays from first particle
                if j == 1
                    trajectory_length = length(pos_data_sliced[1])
                    n_dims = length(dims)
                    n_particles = length(grp)
                    position_arrays = Array{Float32,3}(undef, n_particles, n_dims, trajectory_length)
                    velocity_arrays = Array{Float32,3}(undef, n_particles, n_dims, trajectory_length)
                    acceleration_arrays = Array{Float32,3}(undef, n_particles, n_dims, trajectory_length)
                    particle_types = Array{Int}(undef, n_particles)
                end
                
                # Populate 3D arrays
                for idx in 1:length(dims)
                    position_arrays[j, idx, :] = pos_data_sliced[idx]
                    velocity_arrays[j, idx, :] = vel_data_sliced[idx]
                    acceleration_arrays[j, idx, :] = acc_data_final[idx]
                end
                particle_types[j] = prtl[!, type_col][1]
            end
            
            # Store trajectory length and metadata
            fid["$tra/trajectory_length"] = trajectory_length
            fid["$tra/type"] = particle_types
            
            # Store data in timestep-based format
            for t in 1:trajectory_length
                fid["$tra/pos[$t]"] = convert(Matrix{Float32}, position_arrays[:, :, t])
                fid["$tra/vel[$t]"] = convert(Matrix{Float32}, velocity_arrays[:, :, t])
                fid["$tra/acc[$t]"] = convert(Matrix{Float32}, acceleration_arrays[:, :, t])
            end
        end
    finally
        close(fid)
    end
end


"""
    _get_acceleration_slice(n_points::Int, scheme::String)::UnitRange

Get the range of indices that should be kept for extra fields based on acceleration shortening.
Different schemes shorten acceleration arrays differently, and extra fields should match.

# Arguments
- `n_points::Int`: Total number of data points
- `scheme::String`: Interpolation scheme name

# Returns
- `UnitRange`: Range of indices to keep (e.g., 2:end-1 for central_diff)
"""
function _get_acceleration_slice(n_points::Int, scheme::String)::UnitRange
    if scheme == "central_diff" || scheme == "from_pos"
        # These remove first and last point
        return 2:n_points-1
    elseif scheme == "forward_diff"
        # Forward difference: removes last point (estimates at times 0:n-2)
        return 1:n_points-1
    elseif scheme == "backward_diff"
        # Backward difference: removes first point (estimates at times 1:n-1)
        return 2:n_points
    else
        # All other schemes keep all points
        return 1:n_points
    end
end


"""
    _calculate_acceleration(pos_data::Vector, vel_data::Vector, dt::Float64, 
                           scheme::String)::Vector

Calculate acceleration from velocity or position data using specified scheme.
Supports both analytical difference schemes and DataInterpolations.jl methods.

# Arguments
- `pos_data::Vector`: Vector of position arrays (one per dimension)
- `vel_data::Vector`: Vector of velocity arrays (one per dimension)
- `dt::Float64`: Time step
- `scheme::String`: Interpolation scheme: "central_diff", "forward_diff", "backward_diff",
                    "from_pos", "pchip", "linear", "quadratic", "cubic_spline",
                    "quadratic_spline", "cubic_hermite", "lagrange", "akima"

# Returns
- `Vector`: Vector of acceleration arrays (one per dimension)
"""
function _calculate_acceleration(pos_data::Vector, vel_data::Vector, dt::Float64, 
                                scheme::String)::Vector
    
    n_dims = length(vel_data)
    acc_data = Vector{Vector{Float64}}(undef, n_dims)
    
    if scheme == "central_diff"
        # Central difference: a[i] = (v[i+1] - v[i-1]) / (2*dt)
        for dim in 1:n_dims
            acc_data[dim] = (vel_data[dim][3:end] .- vel_data[dim][1:end-2]) ./ (2 * dt)
        end
        
    elseif scheme == "forward_diff"
        # Forward difference: a[i] = (v[i+1] - v[i]) / dt
        for dim in 1:n_dims
            acc_data[dim] = (vel_data[dim][2:end] .- vel_data[dim][1:end-1]) ./ dt
        end
        
    elseif scheme == "backward_diff"
        # Backward difference: a[i] = (v[i] - v[i-1]) / dt  
        # Same computation as forward, but represents acceleration at different time points
        for dim in 1:n_dims
            acc_data[dim] = (vel_data[dim][2:end] .- vel_data[dim][1:end-1]) ./ dt
        end
        
    elseif scheme == "from_pos"
        # Calculate from position: a[i] = (x[i+1] - 2*x[i] + x[i-1]) / dt^2
        for dim in 1:n_dims
            acc_data[dim] = (pos_data[dim][3:end] .- 2 .* pos_data[dim][2:end-1] .+ 
                           pos_data[dim][1:end-2]) ./ (dt * dt)
        end
        
    else
        # All other schemes use DataInterpolations
        n_points = length(vel_data[1])
        t = 0.0:dt:((n_points - 1) * dt)
        
        for dim in 1:n_dims
            # Create interpolation object based on scheme
            vel_interp = _create_interpolation(vel_data[dim], collect(t), scheme)
            
            # Calculate velocity derivatives (acceleration)
            acc_temp = Float64[]
            for time in t
                push!(acc_temp, DataInterpolations.derivative(vel_interp, time))
            end
            acc_data[dim] = acc_temp
        end
    end
    
    return acc_data
end


"""
    _create_interpolation(y::Vector, t::Vector, scheme::String)

Create an interpolation object from DataInterpolations.jl based on the scheme name.

# Arguments
- `y::Vector`: Data values
- `t::Vector`: Time points
- `scheme::String`: Interpolation scheme name

# Returns
- Interpolation object with implemented derivative support
"""
function _create_interpolation(y::Vector, t::Vector, scheme::String)
    if scheme == "pchip"
        return PCHIPInterpolation(y, t)
    elseif scheme == "linear"
        return LinearInterpolation(y, t)
    elseif scheme == "quadratic"
        return QuadraticInterpolation(y, t)
    elseif scheme == "cubic_spline"
        return CubicSpline(y, t)
    elseif scheme == "quadratic_spline"
        return QuadraticSpline(y, t)
    elseif scheme == "cubic_hermite"
        # CubicHermiteSpline requires derivative values
        # Approximate derivatives using finite differences
        dy = _estimate_derivatives(y, t)
        return CubicHermiteSpline(y, dy, t)
    elseif scheme == "lagrange"
        return LagrangeInterpolation(y, t)
    elseif scheme == "akima"
        return AkimaInterpolation(y, t)
    else
        throw(ArgumentError("Unknown interpolation scheme: $scheme"))
    end
end


"""
    _estimate_derivatives(y::Vector, t::Vector)::Vector

Estimate derivatives using central differences, with forward/backward at boundaries.

# Arguments
- `y::Vector`: Function values
- `t::Vector`: Time points (assumed uniform)

# Returns
- `Vector`: Estimated derivatives at each point
"""
function _estimate_derivatives(y::Vector, t::Vector)::Vector
    n = length(y)
    dy = zeros(n)
    
    if n < 2
        return dy
    end
    
    # Use average time step (assumed mostly uniform)
    dt_avg = (t[end] - t[1]) / (n - 1)
    
    # Interior points: central difference
    for i in 2:n-1
        dy[i] = (y[i+1] - y[i-1]) / (2 * dt_avg)
    end
    
    # Boundary points: forward/backward differences
    dy[1] = (y[2] - y[1]) / dt_avg
    dy[n] = (y[n] - y[n-1]) / dt_avg
    
    return dy
end


# Example usage:
# 2D simulation with PCHIP
# csv_to_hdf5(pwd() * "/data/dam_break.csv", "data/dam_break_first_new.h5"; 
            # dt=0.01, dims=[1, 3], interpolation_scheme="pchip")
# 
# 3D simulation with cubic spline
# csv_to_hdf5(pwd() * "/data/dam_break.csv", "data/dam_break_second.h5"; 
#             dt=0.01, dims=[1, 2, 3], interpolation_scheme="cubic_spline")
#
# 2D (skip y-dimension) with extra fields
# csv_to_hdf5(pwd() * "/data/dam_break.csv", "output.h5";
#             dims=[1, 3], interpolation_scheme="linear",
#             extra_fields=[:Mass, :Temperature])


function particleToArray(
    inputPath::String,
    outputPath::String;
    timesteps=0, #TODO change this for trajectories with different number of timesteps
)
    inputFile = HDF5.h5open(inputPath, "r")
    trajectorys = keys(inputFile)
    outputFile = h5open(outputPath, "w")
    for trajectory in trajectorys
        position = 0
        velocity = 0
        acceleration = 0
        type = 0
        dt = 0
        n_particles = 0

        top = inputFile[trajectory]
        numberParticles = HDF5.read(top, "n_particles")
        trajectory_length = HDF5.read(top, "trajectory_length")
        
        # Determine dimension by checking available position data
        dimension = 0
        for d in 1:3
            try
                HDF5.read(top, "particle[1].pos[$d]")
                dimension += 1
            catch
                break
            end
        end

        if timesteps == 0
            start_timestep = 1
            position = Array{Float32,3}(
                undef, numberParticles, dimension, trajectory_length
            )
            velocity = Array{Float32,3}(
                undef, numberParticles, dimension, trajectory_length
            )
            acceleration = Array{Float32,3}(
                undef, numberParticles, dimension, trajectory_length
            )
            timesteps = start_timestep:trajectory_length
            println(timesteps)
        else
            position = Array{Float32,3}(
                undef, numberParticles, dimension, maximum(timesteps)
            )
            velocity = Array{Float32,3}(
                undef, numberParticles, dimension, maximum(timesteps)
            )
            acceleration = Array{Float32,3}(
                undef, numberParticles, dimension, maximum(timesteps)
            )
        end
        type = Array{Int}(undef, numberParticles)

        dt = HDF5.read(top, "dt")
        n_particles = HDF5.read(top, "n_particles")

        for p in 1:numberParticles
            pos_1 = HDF5.read(top, "particle[$p].pos[1]")
            pos_2 = HDF5.read(top, "particle[$p].pos[2]")
            # pos_3 = HDF5.read(top, "particle[$p].pos[3]")

            vel_1 = HDF5.read(top, "particle[$p].vel[1]")
            vel_2 = HDF5.read(top, "particle[$p].vel[2]")
            # vel_3 = HDF5.read(top, "particle[$p].vel[3]")

            acc_1 = HDF5.read(top, "particle[$p].acc[1]")
            acc_2 = HDF5.read(top, "particle[$p].acc[2]")
            # acc_3 = HDF5.read(top, "particle[$p].acc[3]")

            pos = hcat(pos_1, pos_2)
            vel = hcat(vel_1, vel_2)
            acc = hcat(acc_1, acc_2)

            type[p] = HDF5.read(top, "particle[$p].type")

            for n in timesteps
                println(size(pos[n,:]))
                println(size(position[p, :, n]))
                position[p, :, n] = pos[n, :]
                velocity[p, :, n] = vel[n, :]
                acceleration[p, :, n] = acc[n, :]
            end
        end
        top_o = create_group(outputFile, trajectory)

        top_o["type"] = type
        top_o["dt"] = dt
        top_o["n_particles"] = n_particles
        top_o["trajectory_length"] = trajectory_length - start_timestep + 1

        offset = timesteps[1]
        # println("Offset: $offset")

        for n in timesteps
            top_o["pos[$(n-offset+1)]"] = convert(Matrix, position[:, :, n]')
            top_o["vel[$(n-offset+1)]"] = convert(Matrix, velocity[:, :, n]')
            top_o["acc[$(n-offset+1)]"] = convert(Matrix, acceleration[:, :, n]')
        end
        timesteps = 0

        @info size(position)
        @info size(velocity)
        @info size(acceleration)
        @info size(type)
    end
    close(outputFile)
    close(inputFile)
end

# particleToArray(
#     "data/dam_break_first.h5",
#     "data/dam_break_second.h5",#;
#     # "data/ball_down_short_interp.h5",
#     # "data/Ball_no_boundary/datasets/interpol_meanstd_2/valid.h5",
#     # 8:74
# )
