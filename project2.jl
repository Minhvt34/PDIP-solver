# Primal-Dual Interior-Point Solver
# By Minh Vu, Vincent Wang
using Printf;
using LinearAlgebra;
using MatrixDepot, SparseArrays

include("problem_def.jl")
include("starting_point.jl")
include("presolve_extended.jl")
include("conversions.jl")
include("solve_dev.jl")

include("presolve_simple.jl")
include("starting_point_simple.jl")
include("conversions_simple.jl")
include("solve_simple.jl")


# --- Simple Path Attempt ---
function _attempt_simple_path(Problem, tol, maxit)
    @info "Attempting Simple Path..."
    try
        solution = @time simple_iplp(Problem, tol; maxit)

        @show solution.flag

        if solution.flag
            @info "Simple Path Successful."
            return solution
        else
             @info "Simple IPM Failed to Converge."
             return nothing # Indicate simple path failure
        end

    catch e
        @error "Error during Simple Path execution." exception=(e, catch_backtrace())
        return nothing # Indicate simple path failure
    end
end

# --- Extended/Robust Path Attempt ---
function _attempt_extended_path(Problem, tol, maxit)
    @info "Attempting Extended/Robust Path..."
    try
        solution = @time extended_iplp(Problem, tol; maxit)

        return solution

    catch e
        @error "Error during Extended/Robust Path execution." exception=(e, catch_backtrace())
        return nothing # Indicate simple path failure
    end
end


# --- Main iplp Function --- 
function iplp(Problem, tol; maxit=100)
    # Try the simple path first
    simple_result = _attempt_simple_path(Problem, tol, maxit)

    if simple_result !== nothing && simple_result.flag == true # Check if simple path succeeded AND converged
        @info "Using result from Simple Path."
        return simple_result
    else
        if simple_result !== nothing && simple_result.flag == false
            @info "Simple path ran but did not converge."
        elseif simple_result === nothing
             @info "Simple path failed during setup or encountered an error."
        end
        @info "Proceeding to Extended/Robust Path."
        # Simple path failed, try the extended path
        extended_result = _attempt_extended_path(Problem, tol, maxit)
        return extended_result
    end
end

