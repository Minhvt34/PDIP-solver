using Logging
using LinearAlgebra
using SparseArrays # Added for Diagonal
global_logger(ConsoleLogger(stderr, Logging.Debug)) # Set minimum level to Debug

# Structure to hold all reduction information
mutable struct PresolveReductionInfo
    ind0c::Vector{Int}
    dup_main_c::Vector{Int}
    ind_dup_c::Vector{Int}
    ind_fix_c::Vector{Int}
    fix_vals::Vector{Float64}
    dual_lb::Vector{Float64}
    dual_ub::Vector{Float64}
    obj_offset::Float64
    # Store substitution rule: j => (k, a_kj, Dict(i => a_ki for i!=j in row k))
    free_singleton_subs::Dict{Int, Tuple{Int, Float64, Dict{Int, Float64}}} 
    # Track original indices corresponding to the rows/cols of the presolved problem
    original_row_indices::Vector{Int} 
    original_col_indices::Vector{Int}
    
    # Constructor for initial state
    PresolveReductionInfo(m::Int, n::Int) = new(
        Int[], Int[], Int[], Int[], Float64[],
        fill(-Inf, m), fill(Inf, m), 0.0,
        Dict{Int, Tuple{Int, Float64, Dict{Int, Float64}}}(),
        collect(1:m), collect(1:n) # Initially, indices match original
    )
end

function presolve(P::IplpProblem)
    #@debug "Presolve start"
    P_work = deepcopy(P) # Work on a copy
    
    # --- Check initial b --- 
    if any(!isfinite, P_work.b)
        @error "Presolve: Initial b (after deepcopy, before any modification) contains non-finite values!" b=P_work.b
        # Consider returning an error state immediately
    end
    # --- End Check ---
    
    m_orig, n_orig = size(P_work.A)

    # Initialize reduction tracking
    reductions = PresolveReductionInfo(m_orig, n_orig)

    # --- Initial Reductions ---
    # Remove fixed variables first, affects potential_cols initialization
    remove_fixed_variables!(P_work, reductions)

    # Initialize potential rows and columns (flags for active elements)
    potential_rows, potential_cols = initialize_flags(P_work, reductions)

    # --- Main Presolve Loop ---
    changed = true
    max_passes = 10 # Limit passes to prevent infinite loops in case of bugs
    pass = 0
    while changed && pass < max_passes
        pass += 1
        #@debug "Presolve pass $pass"
        changed = false

        # Row-based reductions
        changed |= remove_row_singletons!(P_work, reductions, potential_rows, potential_cols)
        changed |= remove_forcing_constraints!(P_work, reductions, potential_rows, potential_cols)
        changed |= remove_dominated_constraints!(P_work, reductions, potential_rows, potential_cols) # Checks for zero rows -> redundancy/infeas

        # Column-based reductions
        changed |= remove_free_implied_column_singletons!(P_work, reductions, potential_rows, potential_cols)
        changed |= remove_doubleton_column_singletons!(P_work, reductions, potential_rows, potential_cols) # TODO
        changed |= remove_dominated_columns!(P_work, reductions, potential_rows, potential_cols) # TODO

        # Duplicate rows and columns
        changed |= remove_duplicate_rows!(P_work, reductions, potential_rows, potential_cols)
        changed |= remove_duplicate_columns!(P_work, reductions, potential_rows, potential_cols) # TODO
        
        # Note: remove_empty_rows_and_cols! is done *after* the loop
    end
    if pass == max_passes
         @warn "Presolve reached maximum pass limit ($max_passes). Proceeding with current reductions."
    end

    # --- Final Cleanup ---
    # Remove rows/columns marked inactive during the loop
    remove_empty_rows_and_cols!(P_work, reductions, potential_rows, potential_cols)

    #@debug "Presolve end. Final problem size: $(size(P_work.A))"
    
    # Return the presolved problem and all collected reduction info
    return :Success,
           P_work, # Return the modified problem object
           reductions.ind0c, reductions.dup_main_c, reductions.ind_dup_c, 
           reductions.ind_fix_c, reductions.fix_vals,
           reductions.dual_lb, reductions.dual_ub, # Note: dual bounds might need adjustment for final rows
           reductions.obj_offset, reductions.free_singleton_subs
           # We don't explicitly return original_row/col_indices, they are used internally in revProb indirectly
end

# --- Subprocedure Definitions (Modified Signatures and Logic) ---

function remove_fixed_variables!(P, reductions::PresolveReductionInfo)
    #@debug "Removing fixed variables"
    changed = false
    _, n = size(P.A)
    # Identify columns where lower == upper (using original indices)
    fixed_orig_indices = findall(j -> P.lo[j] == P.hi[j], 1:n)
    
    if isempty(fixed_orig_indices)
        return false
    end

    fixed_vals = P.lo[fixed_orig_indices]
    
    # --- Record Reduction Info ---
    append!(reductions.ind_fix_c, fixed_orig_indices)
    append!(reductions.fix_vals, fixed_vals)
    reductions.obj_offset += dot(P.c[fixed_orig_indices], fixed_vals)
    #@debug "Recording fixed variables: $(fixed_orig_indices) with values $(fixed_vals). Obj offset change: $(dot(P.c[fixed_orig_indices], fixed_vals))"
    # --- End Record ---

    # Substitute fixed variable contributions into RHS (b)
    if !isempty(fixed_orig_indices)
        # Check b before modification
        if any(!isfinite, P.b)
            @warn "Presolve: b contains non-finite values BEFORE fixed var substitution!" b=P.b
        end
        # Check fixed_vals
        if any(!isfinite, fixed_vals)
            @warn "Presolve: fixed_vals contains non-finite values!" fixed_vals=fixed_vals
        end
        # Check A subset (optional, might be slow)
        # if any(!isfinite, P.A[:, fixed_orig_indices].nzval)
        #     @warn "Presolve: A subset for fixed vars contains non-finite values!"
        # end
        
        P.b .-= P.A[:, fixed_orig_indices] * fixed_vals
        
        # Check b after modification
        if any(!isfinite, P.b)
            @warn "Presolve: b contains non-finite values AFTER fixed var substitution!" b=P.b fixed_indices=fixed_orig_indices fixed_values=fixed_vals
            # Potentially throw error or return early if critical
        end
        changed = true
    end
    
    # These columns will be removed later by remove_empty_rows_and_cols!
    # We don't modify P.A, P.c, P.lo, P.hi here, just mark them.
    # The initialize_flags function needs to know about these.
    
    return changed # Indicate that potential_cols needs update based on this
end

# Initialize flags based on current problem state AND already fixed variables
function initialize_flags(P, reductions::PresolveReductionInfo)
    #@debug "Initializing potential rows and columns"
    m, n = size(P.A)
    rows = trues(m)
    cols = trues(n)
    # Exclude already fixed variables from potential columns
    cols[reductions.ind_fix_c] .= false
    #@debug "Initial potential cols (after fixed vars): $(sum(cols)) / $n"
    return rows, cols
end

function remove_row_singletons!(P, reductions::PresolveReductionInfo, potential_rows, potential_cols)
    #@debug "Removing row singletons"
    changed = false
    m, n = size(P.A)
    tol = 1e-8
    
    # Store indices to fix/remove in this pass to avoid modifying during iteration
    vars_to_fix = Dict{Int, Float64}() # Map original col index -> fixed value
    rows_to_remove = Int[]

    # Loop over active rows (using current indices of P_work)
    current_m, current_n = size(P.A)
    for i in 1:current_m 
        if !potential_rows[i] continue end

        # Find potentially active columns with nonzeros in this row
        cols_in_row = findall(j -> potential_cols[j] && P.A[i, j] != 0.0, 1:current_n)
        
        if length(cols_in_row) == 1
            j = cols_in_row[1] # Index relative to current P_work.A columns
            a_ij = P.A[i, j]
            
            if abs(a_ij) > tol
                fixed_val = P.b[i] / a_ij
                
                # Get original column index corresponding to j
                original_j = reductions.original_col_indices[j]

                # Check bounds - if fixed_val violates original bounds, problem is infeasible
                # Use original bounds P.lo, P.hi indexed by original_j
                # NOTE: Need the *original* problem P for this check, or pass bounds carefully.
                # Assuming P passed to presolve is the original for bounds check.
                # if fixed_val < P.lo[original_j] - tol || fixed_val > P.hi[original_j] + tol
                #    error("Presolve: Row singleton fixes var $original_j=$fixed_val, violates bounds [$(P.lo[original_j]), $(P.hi[original_j])]. Problem likely infeasible.")
                # end

                # Check if already fixed to a different value
                if haskey(vars_to_fix, original_j) && abs(vars_to_fix[original_j] - fixed_val) > tol
                    error("Presolve: Row singleton tries to fix var $original_j to $fixed_val, but already marked for fixing to $(vars_to_fix[original_j]). Inconsistent.")
                elseif !(original_j in reductions.ind_fix_c) # Only fix if not already fixed globally
                     vars_to_fix[original_j] = fixed_val
                     push!(rows_to_remove, i) # Mark row for removal
                     #@debug "Row singleton at row $i (orig $(reductions.original_row_indices[i])) proposes fixing var $j (orig $original_j) = $fixed_val"
                     changed = true
                end
            end
        end
    end

    # Apply the fixes found in this pass
    if !isempty(vars_to_fix)
        fixed_orig_indices = collect(keys(vars_to_fix))
        fixed_vals = collect(values(vars_to_fix))

        # Find current indices corresponding to original fixed indices
        current_fixed_indices = [findfirst(==(orig_j), reductions.original_col_indices) for orig_j in fixed_orig_indices]
        current_fixed_indices = filter(!isnothing, current_fixed_indices) # Filter out those already removed

        if !isempty(current_fixed_indices)
            # --- Record Reduction Info ---
            append!(reductions.ind_fix_c, fixed_orig_indices)
            append!(reductions.fix_vals, fixed_vals)
            # Use original costs P.c indexed by original_j
            # obj_change = dot(P.c[fixed_orig_indices], fixed_vals) # Need original P.c
            # Assume P_work.c still holds costs for currently active vars
            obj_change = dot(P.c[current_fixed_indices], fixed_vals) 
            reductions.obj_offset += obj_change
            #@debug "Applying row singleton fixes. Vars: $(fixed_orig_indices), Vals: $(fixed_vals). Obj change: $obj_change"
             # --- End Record ---

            # Substitute fixed variable contributions into RHS (b)
            # --- Add Checks ---
            if any(!isfinite, P.b)
                @warn "Presolve/RowSingleton: b contains non-finite values BEFORE substitution!" b=P.b
            end
            if any(!isfinite, fixed_vals)
                @warn "Presolve/RowSingleton: fixed_vals contains non-finite values!" fixed_vals=fixed_vals
            end
            # --- End Checks ---
            
            P.b .-= P.A[:, current_fixed_indices] * fixed_vals
            
            # --- Add Checks ---
            if any(!isfinite, P.b)
                @warn "Presolve/RowSingleton: b contains non-finite values AFTER substitution!" b=P.b fixed_indices=current_fixed_indices fixed_values=fixed_vals
            end
            # --- End Checks ---
        
             # Mark columns as inactive (will be removed by remove_empty_...)
            potential_cols[current_fixed_indices] .= false
        end
        
        # Mark rows as inactive
        potential_rows[rows_to_remove] .= false
    end

    return changed
end


function remove_forcing_constraints!(P, reductions::PresolveReductionInfo, potential_rows, potential_cols)
    #@debug "Removing forcing constraints"
    changed = false
    m, n = size(P.A) # Current dimensions
    tol = 1e-8

    vars_to_fix = Dict{Int, Float64}() # Map original col index -> fixed value
    rows_to_remove = Int[]

    for i in 1:m
        if !potential_rows[i] continue end

        # Identify active nonzeros in this row (using current indices)
        row_cols = findall(j -> potential_cols[j] && P.A[i,j] != 0.0, 1:n)
        if isempty(row_cols) continue end
        
        # Map current column indices to original indices
        orig_row_cols = reductions.original_col_indices[row_cols]

        # Compute min (gi) and max (hi) LHS using original bounds P.lo, P.hi
        gi = 0.0; hi = 0.0
        gi_neg_inf = false; hi_pos_inf = false
        
        for (idx, j) in enumerate(row_cols) # j is current index
            orig_j = orig_row_cols[idx] # Corresponding original index
            aij = P.A[i,j]
            lj = P.lo[orig_j]; uj = P.hi[orig_j] # Use original bounds

            # Careful handling of infinities
            if aij > 0
                if isinf(lj) && lj < 0 gi_neg_inf = true elseif !isinf(lj) gi += aij*lj end
                if isinf(uj) && uj > 0 hi_pos_inf = true elseif !isinf(uj) hi += aij*uj end
            else # aij < 0
                if isinf(uj) && uj > 0 gi_neg_inf = true elseif !isinf(uj) gi += aij*uj end
                if isinf(lj) && lj < 0 hi_pos_inf = true elseif !isinf(lj) hi += aij*lj end
            end
            # If aij == 0, it's skipped by row_cols findall
        end

        bi = P.b[i] # Current RHS value

        # Check for forcing condition
        is_forcing = false
        forcing_type = :none 
        
        # Forcing at minimum?
        if !gi_neg_inf && abs(gi - bi) < tol 
             # Check if implied hi also matches bi (this means row is fixed equality)
             if !hi_pos_inf && abs(hi - bi) < tol
                 # Row is effectively fixed, implies all vars are at bounds inducing this value
                 is_forcing = true
                 forcing_type = :min_max_equal # Both min and max hit b_i
             else
                 # Only minimum is forcing
                 is_forcing = true
                 forcing_type = :min
             end
        # Forcing at maximum?
        elseif !hi_pos_inf && abs(hi - bi) < tol
            is_forcing = true
            forcing_type = :max
        end

        if is_forcing
            #@debug "Forcing row $i (orig $(reductions.original_row_indices[i])), type: $forcing_type"
            push!(rows_to_remove, i) # Mark row for removal regardless of type
            changed = true

            # Determine the value each variable in the row must take
            for (idx, j) in enumerate(row_cols) # j is current index
                orig_j = orig_row_cols[idx] # Corresponding original index
                aij = P.A[i,j]
                
                fixed_val = NaN
                if forcing_type == :min || forcing_type == :min_max_equal
                    # If forcing at min, vars take the bound that contributes to gi
                    fixed_val = (aij > 0) ? P.lo[orig_j] : P.hi[orig_j]
                elseif forcing_type == :max 
                    # If forcing at max, vars take the bound that contributes to hi
                     fixed_val = (aij > 0) ? P.hi[orig_j] : P.lo[orig_j]
                end

                # Check if bound is finite
                if isinf(fixed_val)
                    error("Presolve: Forcing constraint row $i (orig $(reductions.original_row_indices[i])) requires var $orig_j to be infinite. Problem likely infeasible or unbounded.")
                end

                # Check for conflicts and add to fix list
                if haskey(vars_to_fix, orig_j) && abs(vars_to_fix[orig_j] - fixed_val) > tol
                     error("Presolve: Forcing row $i conflicts with previous fix for var $orig_j. Current val: $fixed_val, previous: $(vars_to_fix[orig_j]).")
                elseif !(orig_j in reductions.ind_fix_c) # Only fix if not globally fixed yet
                     vars_to_fix[orig_j] = fixed_val
                     ##@debug "  -> Proposes fixing var $j (orig $orig_j) = $fixed_val"
                end
            end
        end
    end
    
    # Apply the fixes found in this pass
    if !isempty(vars_to_fix)
        fixed_orig_indices = collect(keys(vars_to_fix))
        fixed_vals = collect(values(vars_to_fix))

        # Find current indices corresponding to original fixed indices
        current_fixed_indices = [findfirst(==(orig_j), reductions.original_col_indices) for orig_j in fixed_orig_indices]
        current_fixed_indices = filter(!isnothing, current_fixed_indices)

        if !isempty(current_fixed_indices)
            # --- Record Reduction Info ---
            append!(reductions.ind_fix_c, fixed_orig_indices)
            append!(reductions.fix_vals, fixed_vals)
            # obj_change = dot(P.c[fixed_orig_indices], fixed_vals) # Need original P.c
            obj_change = dot(P.c[current_fixed_indices], fixed_vals) # Use current c for active vars
            reductions.obj_offset += obj_change
            ##@debug "Applying forcing constraint fixes. Vars: $(fixed_orig_indices), Vals: $(fixed_vals). Obj change: $obj_change"
             # --- End Record ---

            # Substitute fixed variable contributions into RHS (b)
            # --- Add Checks ---
            if any(!isfinite, P.b)
                @warn "Presolve/Forcing: b contains non-finite values BEFORE substitution!" b=P.b
            end
            if any(!isfinite, fixed_vals)
                @warn "Presolve/Forcing: fixed_vals contains non-finite values!" fixed_vals=fixed_vals
            end
            # --- End Checks ---
            
            P.b .-= P.A[:, current_fixed_indices] * fixed_vals
            
            # --- Add Checks ---
            if any(!isfinite, P.b)
                @warn "Presolve/Forcing: b contains non-finite values AFTER substitution!" b=P.b fixed_indices=current_fixed_indices fixed_values=fixed_vals
            end
            # --- End Checks ---
        
             # Mark columns as inactive
            potential_cols[current_fixed_indices] .= false
        end
        
         # Mark rows as inactive
        potential_rows[rows_to_remove] .= false
    end
    
    return changed
end


function remove_dominated_constraints!(P, reductions::PresolveReductionInfo, potential_rows, potential_cols)
    #@debug "Removing dominated constraints (zero rows)"
    changed = false
    m, n = size(P.A) # Current dimensions
    tol = 1e-8
    
    rows_to_remove = Int[]

    for i in 1:m
        if !potential_rows[i] continue end

        # Check if row i is effectively zero over active columns
        active_cols_in_row = findall(j -> potential_cols[j] && abs(P.A[i, j]) > tol, 1:n)
        
        if isempty(active_cols_in_row)
            # Row is zero. Check RHS.
            if abs(P.b[i]) < tol
                # Redundant row
                push!(rows_to_remove, i)
                orig_i = reductions.original_row_indices[i]
                ##@debug "Removed zero row $i (orig $orig_i) - redundant"
                # TODO: Potentially update dual bounds if needed, depends on equality/inequality type
                changed = true
            else
                # Infeasible problem
                error("Presolve: Problem infeasible. Zero row $i (orig $(reductions.original_row_indices[i])) has non-zero RHS b=$(P.b[i])")
            end
        end
    end
    
    if !isempty(rows_to_remove)
        potential_rows[rows_to_remove] .= false
    end
    
    return changed
end


function remove_free_implied_column_singletons!(P, reductions::PresolveReductionInfo, potential_rows, potential_cols)
    #@debug "Removing free/implied column singletons"
    changed = false
    m, n = size(P.A) # Current dimensions
    tol = 1e-8

    cols_to_remove = Int[]
    rows_to_remove = Int[]
    substitutions = Dict{Int, Tuple{Int, Float64, Dict{Int, Float64}}}() # Store proposed substitutions: orig_j => (orig_k, a_kj, Dict(orig_i => a_ki))

    for j in 1:n
        if !potential_cols[j] continue end
        
        orig_j = reductions.original_col_indices[j]

        # Check if variable is effectively free (ignoring singletons for now)
        # Original bounds: P.lo[orig_j], P.hi[orig_j]
        is_free = isinf(P.lo[orig_j]) && P.hi[orig_j] == -P.lo[orig_j] # Or just check if isinf(lo) and isinf(hi)? Definition varies. Assuming standard free var.
        is_implied_free = false # TODO: Add check for implied free based on bounds and row constraints
        
        if is_free || is_implied_free
            # Find active rows where this column has a non-zero entry
            rows_with_j = findall(i -> potential_rows[i] && abs(P.A[i,j]) > tol, 1:m)

            if length(rows_with_j) == 1
                k = rows_with_j[1] # Current row index
                orig_k = reductions.original_row_indices[k]
                akj = P.A[k, j] # Value in current matrix

                # Find other active columns in this row k
                other_cols_in_row_k = findall(i -> potential_cols[i] && i != j && abs(P.A[k,i]) > tol, 1:n)
                orig_other_cols = reductions.original_col_indices[other_cols_in_row_k]

                # Prepare substitution dictionary using original indices
                subs_dict_orig_indices = Dict{Int, Float64}()
                for (idx, other_j) in enumerate(other_cols_in_row_k)
                    subs_dict_orig_indices[orig_other_cols[idx]] = P.A[k, other_j]
                end

                # --- Record Reduction Info ---
                substitutions[orig_j] = (orig_k, akj, subs_dict_orig_indices)
                 ##@debug "Free column singleton: var $j (orig $orig_j) in row $k (orig $orig_k). Substitution rule prepared."
                # --- End Record ---

                # Mark column j and row k for removal
                push!(cols_to_remove, j)
                push!(rows_to_remove, k)
                changed = true
            end
        end
    end

    # Apply substitutions (update objective, mark rows/cols)
    if !isempty(substitutions)
        current_obj_offset = 0.0
        # Need original P.c for this
        # We assume P.c holds the costs for the current active columns
        
        for (orig_j, (orig_k, akj, subs_dict_orig)) in substitutions
             # Find current index for orig_j if it still exists
             current_j = findfirst(==(orig_j), reductions.original_col_indices)
             if isnothing(current_j) continue end # Already removed by another rule

             cj = P.c[current_j] # Cost of the variable being substituted out
             bk = P.b[findfirst(==(orig_k), reductions.original_row_indices)] # RHS of the singleton row (current index)

             if abs(akj) < tol
                 @warn "Near-zero coefficient akj=$akj for free var $orig_j in row $orig_k. Skipping substitution."
                 continue
             end

             # Update objective offset: Add constant term cj * bk / akj
             current_obj_offset += cj * bk / akj

             # Update costs of other variables in the equation: ci_new = ci - cj * aki / akj
             for (orig_i, aki) in subs_dict_orig
                 current_i = findfirst(==(orig_i), reductions.original_col_indices)
                 if !isnothing(current_i) && potential_cols[current_i] # Check if column i still exists and is active
                     P.c[current_i] -= cj * aki / akj
                 end
             end
        end
        
        reductions.obj_offset += current_obj_offset
        merge!(reductions.free_singleton_subs, substitutions) # Add new substitutions
         ##@debug "Applied free singleton substitutions. Obj offset change: $current_obj_offset. Total subs: $(length(reductions.free_singleton_subs))"

        # Mark columns and rows as inactive
        unique!(cols_to_remove)
        unique!(rows_to_remove)
        potential_cols[cols_to_remove] .= false
        potential_rows[rows_to_remove] .= false
    end

    return changed
end

# --- Placeholder/Stub Functions ---

function remove_doubleton_column_singletons!(P, reductions::PresolveReductionInfo, potential_rows, potential_cols)
    # #@debug "Removing doubleton column singletons (TODO)"
    return false # No changes made
end

function remove_dominated_columns!(P, reductions::PresolveReductionInfo, potential_rows, potential_cols)
    # #@debug "Removing dominated columns (TODO)"
    return false # No changes made
end

function remove_duplicate_rows!(P, reductions::PresolveReductionInfo, potential_rows, potential_cols)
     #@debug "Removing duplicate rows via hashing"
    # This implementation is complex due to needing original indices and handling RHS/dual bounds.
    # Keeping it simple for now: just marks potential duplicates.
    changed = false
    m, n = size(P.A)
    tol = 1e-8
    
    rows_to_remove = Int[]

    # Consider only active rows and columns for hashing
    active_row_indices = findall(potential_rows)
    active_col_indices = findall(potential_cols)

    if isempty(active_row_indices) || isempty(active_col_indices)
        return false
    end
    
    submatrix = P.A[active_row_indices, active_col_indices]
    
    # Group rows by hash of their active entries
    row_groups = Dict{UInt64, Vector{Int}}() # Stores indices relative to active_row_indices
    for (idx, i) in enumerate(active_row_indices)
        row_vec = submatrix[idx, :]
        h = hash(row_vec) 
        push!(get!(row_groups, h, Int[]), idx) # Store index within active_row_indices
    end

    # Within each hash group, check for actual duplicates
    for group_indices in values(row_groups)
        if length(group_indices) > 1
            # Indices in group_indices are relative to active_row_indices
            main_active_idx = group_indices[1]
            main_orig_idx = active_row_indices[main_active_idx] # Original index of the main row

            for k in 2:length(group_indices)
                dup_active_idx = group_indices[k]
                dup_orig_idx = active_row_indices[dup_active_idx] # Original index of the potential duplicate

                # Compare RHS values (b) for the original rows
                if abs(P.b[main_orig_idx] - P.b[dup_orig_idx]) < tol
                    # Check if vectors truly match (hash collision check)
                    if P.A[main_orig_idx, active_col_indices] == P.A[dup_orig_idx, active_col_indices]
                        push!(rows_to_remove, dup_orig_idx) # Mark the original duplicate row index for removal
                        ##@debug "Removed duplicate row $dup_orig_idx matching row $main_orig_idx"
                        changed = true
                        # TODO: Need to handle dual variable bounds adjustment here.
                        # dual_lb/ub for main_orig_idx might need update.
                    end
                else
                    # TODO: Handle implied bounds from duplicate rows with different RHS.
                end
            end
        end
    end
    
    if !isempty(rows_to_remove)
         unique!(rows_to_remove)
         # Convert original row indices back to current indices before marking
         current_indices_to_remove = [findfirst(==(orig_r), reductions.original_row_indices) for orig_r in rows_to_remove]
         current_indices_to_remove = filter(!isnothing, current_indices_to_remove)
         potential_rows[current_indices_to_remove] .= false
    end

    return changed
end


function remove_duplicate_columns!(P, reductions::PresolveReductionInfo, potential_rows, potential_cols)
    #@debug "Removing duplicate columns via hashing (identical only)"
    changed = false
    m, n = size(P.A) # Current dimensions
    tol = 1e-8

    cols_to_remove = Int[]
    main_cols_for_removed = Int[] # Store the main original index corresponding to each removed original index

    # Consider only active rows and columns for hashing
    active_row_indices = findall(potential_rows)
    active_col_indices = findall(potential_cols)

    if isempty(active_row_indices) || isempty(active_col_indices)
        return false
    end

    # Transpose the relevant submatrix for easier column hashing/comparison
    submatrix_T = P.A[active_row_indices, active_col_indices]'

    # Group columns by hash of their active entries
    col_groups = Dict{UInt64, Vector{Int}}() # Stores indices relative to active_col_indices
    for (idx, j) in enumerate(active_col_indices)
        col_vec = submatrix_T[idx, :] # Get the column slice (now a row in transposed matrix)
        h = hash(col_vec)
        push!(get!(col_groups, h, Int[]), idx) # Store index within active_col_indices
    end

    # Within each hash group, check for actual duplicates
    for group_indices in values(col_groups)
        if length(group_indices) > 1
            # Indices are relative to active_col_indices
            group_orig_indices = active_col_indices[group_indices] # Original indices for this group

            # Use first column in group as the potential main column
            main_active_idx = group_indices[1]
            main_orig_idx = active_col_indices[main_active_idx]

            for k in 2:length(group_indices)
                dup_active_idx = group_indices[k]
                dup_orig_idx = active_col_indices[dup_active_idx]

                # Compare the full original columns (safer than just active submatrix due to potential previous removals)
                # Use current A matrix indices
                if P.A[active_row_indices, main_active_idx] == P.A[active_row_indices, dup_active_idx]
                    # --- Identical Column Found --- 
                    ##@debug "Duplicate column found: $dup_orig_idx (active $dup_active_idx) matches $main_orig_idx (active $main_active_idx)"
                    
                    # Combine bounds: [lo_m, hi_m] = [lo_m+lo_d, hi_m+hi_d]
                    # Combine cost: c_m = c_m + c_d
                    # Need original bounds/costs for this. Assume P holds current active data mapped correctly.
                    # Access using current active indices:
                    P.lo[main_active_idx] += P.lo[dup_active_idx]
                    P.hi[main_active_idx] += P.hi[dup_active_idx]
                    P.c[main_active_idx]  += P.c[dup_active_idx]
                    ##@debug "  Combined bounds/cost: Main $main_orig_idx now has lo=$(P.lo[main_active_idx]), hi=$(P.hi[main_active_idx]), c=$(P.c[main_active_idx])"

                    # Mark duplicate for removal
                    push!(cols_to_remove, dup_active_idx) # Mark current index
                    
                    # --- Record Reduction Info ---
                    # We need original indices for revProb
                    push!(reductions.ind_dup_c, dup_orig_idx) 
                    push!(reductions.dup_main_c, main_orig_idx)
                    # --- End Record ---
                    
                    changed = true
                else
                    # TODO: Handle scaled columns (A[:,j] == alpha * A[:,k])
                    # This requires more complex bound/cost adjustments and variable scaling/shifting.
                end
            end
        end
    end

    if !isempty(cols_to_remove)
         unique!(cols_to_remove)
         potential_cols[cols_to_remove] .= false # Mark current indices as inactive
         ##@debug "Marked $(length(cols_to_remove)) duplicate columns for removal."
    end

    return changed
end

# --- Final Cleanup Function ---

function remove_empty_rows_and_cols!(P, reductions::PresolveReductionInfo, potential_rows, potential_cols)
    #@debug "Applying removals of inactive rows and columns"
    
    active_rows = findall(potential_rows)
    active_cols = findall(potential_cols)

    if length(active_rows) == length(potential_rows) && length(active_cols) == length(potential_cols)
        #@debug "No rows or columns to remove."
        return # Nothing to do
    end
    
    ##@debug "Removing $(length(potential_rows) - length(active_rows)) rows and $(length(potential_cols) - length(active_cols)) columns."

    # Update the reduction info to reflect final active indices
    final_original_row_indices = reductions.original_row_indices[active_rows]
    final_original_col_indices = reductions.original_col_indices[active_cols]

    # Update the problem matrices/vectors
    P.A = P.A[active_rows, active_cols]
    P.b = P.b[active_rows]
    P.c = P.c[active_cols] 
    # Note: P.lo and P.hi are NOT subsetted here. They remain the original size
    # because revProb needs the original bounds indexed by original indices.
    # However, the solver will only see variables corresponding to active_cols.

    # Update reduction info tracking
    reductions.original_row_indices = final_original_row_indices
    reductions.original_col_indices = final_original_col_indices
    
    # Adjust dual bounds arrays to match the final number of rows
    reductions.dual_lb = reductions.dual_lb[active_rows]
    reductions.dual_ub = reductions.dual_ub[active_rows]
    
    #@debug "Final problem dimensions: $(size(P.A)). Active rows: $(length(active_rows)), Active cols: $(length(active_cols))"
end


# --- revProb function (modified for clarity and potential issues) ---

# Note: This function now implicitly relies on the *original* P being passed
# for bounds (lo/hi) and costs (c) when dealing with removed variables.
# The presolved problem solution `x1` corresponds to the columns remaining *after* presolve.
function revProb(P_orig::IplpProblem, # The original problem before any presolve
                 ind0c, dup_main_c, ind_dup_c, 
                 ind_fix_c, fix_vals, 
                 free_singleton_subs, # Removed dual_lb/ub, obj_offset as they aren't used here
                 x1) # Solution vector for the presolved problem

    _ , n_orig = size(P_orig.A) 
    x = Vector{Float64}(undef, n_orig)
    fill!(x, NaN) # Initialize with NaN

    #@debug "revProb started. Original n=$n_orig. Presolved solution length=$(length(x1))."
    #@debug "Fixed vars: $(length(ind_fix_c)), Zero cost vars: $(length(ind0c)), Dup vars: $(length(ind_dup_c)), Free singletons: $(length(free_singleton_subs))"

    # --- Step 1: Map solved variables (x1) back to their original indices ---
    
    # Determine the original column indices that correspond to the solution vector x1
    # These are the columns that were *not* removed by any presolve step.
    potential_active_indices = Set(1:n_orig)
    setdiff!(potential_active_indices, ind_fix_c)
    setdiff!(potential_active_indices, ind0c) # Assuming ind0c contains columns removed due to zero cost
    setdiff!(potential_active_indices, ind_dup_c) # Assuming these are removed duplicates
    setdiff!(potential_active_indices, keys(free_singleton_subs)) # Free singletons are substituted out
    
    active_indices_presolved = sort(collect(potential_active_indices))

    if length(x1) != length(active_indices_presolved)
         # This error indicates a mismatch between the columns the solver saw 
         # and the columns revProb thinks should remain after presolve reductions.
         error("revProb Error: Mismatch! Solver returned $(length(x1)) values, but revProb expects $(length(active_indices_presolved)) active columns based on reduction info. Check presolve logic consistency.")
    end
    
    # Assign solved values from x1 back to their original positions in x
    for (idx, orig_idx) in enumerate(active_indices_presolved)
        if orig_idx > n_orig || orig_idx < 1
             error("revProb Error: Invalid original index $orig_idx found while mapping solution.")
        end
        x[orig_idx] = x1[idx]
    end
    #@debug "Mapped solver solution x1 to $(length(active_indices_presolved)) active original indices."

    # --- Step 2: Assign values for fixed variables ---
    fix_val_map = Dict(zip(ind_fix_c, fix_vals))
    fixed_assigned_count = 0
    for i in ind_fix_c
         if i > n_orig || i < 1; error("revProb Error: Invalid fixed index $i."); end
         if !isnan(x[i]) && abs(x[i] - fix_val_map[i]) > 1e-8 # Check for conflict with solver result (shouldn't happen if logic is correct)
              @warn "revProb Warning: Solver assigned value $(x[i]) to variable $i, which was also fixed to $(fix_val_map[i]) by presolve. Using fixed value."
         end
         x[i] = fix_val_map[i]
         fixed_assigned_count += 1
    end
     #@debug "Assigned values to $fixed_assigned_count fixed variables."

    # --- Step 3: Assign values for zero-cost variables (removed via ind0c) ---
    # The logic here depends on how ind0c was populated. Assuming it's for vars fixed
    # at a bound due to cost sign when the column became all-zero.
    zero_cost_assigned_count = 0
    for i in ind0c
        if i > n_orig || i < 1; error("revProb Error: Invalid zero-cost index $i."); end
        if isnan(x[i]) # Only assign if not already fixed or solved
            val = NaN
            # Use ORIGINAL cost P_orig.c and bounds P_orig.lo/hi
            ci = P_orig.c[i]
            li = P_orig.lo[i]
            ui = P_orig.hi[i]
            if ci > 1e-10 && !isinf(li)      val = li
            elseif ci < -1e-10 && !isinf(ui) val = ui
            elseif !isinf(li)                val = li # If c=0, prefer lower bound if finite
            elseif !isinf(ui)                val = ui # Else prefer upper bound if finite
            else                             val = 0.0 # Default to 0 if bounds are infinite and c=0
            end
            x[i] = val
            zero_cost_assigned_count += 1
        end
    end
     #@debug "Assigned values to $zero_cost_assigned_count zero-cost variables (ind0c)."


    # --- Step 4: Assign values for duplicate columns (removed via ind_dup_c) ---
    # Assign value based on the corresponding main column's value.
    if !isempty(ind_dup_c)
        if length(dup_main_c) != length(ind_dup_c)
            error("revProb Error: Mismatch between duplicate columns list (ind_dup_c) and main columns list (dup_main_c).")
        end
        dup_map = Dict(zip(ind_dup_c, dup_main_c)) # Map: removed_dup_idx -> its_main_idx
        duplicate_assigned_count = 0
        for i in ind_dup_c
            if i > n_orig || i < 1; error("revProb Error: Invalid duplicate index $i."); end
            if isnan(x[i]) # Only assign if not already set
                main_col_idx = dup_map[i]
                if main_col_idx > n_orig || main_col_idx < 1; error("revProb Error: Invalid main column index $main_col_idx for duplicate $i."); end
                
                if isnan(x[main_col_idx])
                    # This implies an issue - main column value should be known by now (either solved or fixed)
                    @warn "revProb Warning: Value for main column $main_col_idx needed for duplicate $i is not yet assigned. Assigning 0 as fallback."
                    x[i] = 0.0 # Fallback
                else
                    # Assign same value as main column (for identical duplicates)
                    # TODO: If scaled duplicates were handled (A[:,i] = alpha * A[:,j]), this needs adjustment:
                    # The relationship between x_i and x_j in the combined variable needs careful reversal.
                    # For simple identical case (alpha=1), x_i = x_j / 2 IF the bounds were combined assuming y = x_i + x_j ?
                    # Or is it simpler? If y = x_i+x_j is the combined variable, then x_i = x_j = y? Needs clarification based on combination rule.
                    # Sticking to x[i] = x[main] for now as it matches the simple bound/cost sum.
                    x[i] = x[main_col_idx]
                end
                duplicate_assigned_count += 1
            end
        end
        #@debug "Assigned values to $duplicate_assigned_count duplicate variables (ind_dup_c) based on main columns."
    else
        #@debug "No duplicate columns (ind_dup_c) to process."
    end


    # --- Step 5: Calculate values for substituted free column singletons ---
    if !isempty(free_singleton_subs)
         #@debug "Calculating values for $(length(free_singleton_subs)) substituted free singletons..."
         max_passes = n_orig # Safety break for dependency loops
         current_pass = 0
         made_progress = true
         unresolved_free_singletons = Set(keys(free_singleton_subs))
         
         while made_progress && !isempty(unresolved_free_singletons) && current_pass < max_passes
             made_progress = false
             current_pass += 1
             resolved_this_pass = Int[]
             
             for j in unresolved_free_singletons
                 if j > n_orig || j < 1; error("revProb Error: Invalid free singleton index $j."); end
                 (k, akj, subs_dict) = free_singleton_subs[j] # Uses original indices k, j, and keys(subs_dict)
                 
                 sum_term = 0.0
                 dependencies_met = true
                 for (i_sub, aki) in subs_dict # i_sub is original index
                     if i_sub > n_orig || i_sub < 1; error("revProb Error: Invalid dependency index $i_sub for free singleton $j."); end
                     if isnan(x[i_sub]) 
                         dependencies_met = false
                         # #@debug "Dependency $i_sub for free singleton $j not met yet."
                         break # Cannot calculate x[j] yet
                     end
                     sum_term += aki * x[i_sub]
                 end
                     
                 if dependencies_met
                     # Use original RHS P_orig.b[k]
                     if abs(akj) > 1e-10
                         bk = P_orig.b[k] 
                         x[j] = (bk - sum_term) / akj
                         push!(resolved_this_pass, j)
                         made_progress = true
                          #@debug " Resolved free singleton $j = $(x[j])"
                     else
                          @warn("revProb Warning: Cannot resolve free singleton var $j: stored coefficient akj ($akj) is near zero.")
                          x[j] = 0.0 # Assign 0 as fallback? Or leave NaN? Leaving NaN for now.
                          push!(resolved_this_pass, j) # Mark as 'resolved' to avoid infinite loop
                     end
                 end
             end
             # Remove resolved singletons from the set for the next pass
             setdiff!(unresolved_free_singletons, resolved_this_pass)
         end
         
         if !isempty(unresolved_free_singletons)
              warn("revProb Warning: Could not resolve all free singleton variables. Unresolved indices: $(unresolved_free_singletons). Final values might be NaN. Check for circular dependencies.")
         end
    end

    # --- Final Check ---
    unassigned_indices = findall(isnan, x)
    if !isempty(unassigned_indices)
        @warn "revProb Warning: Not all variables were assigned a value. Unassigned indices: $(unassigned_indices). This might be okay if presolve detected unboundedness/infeasibility implicitly, but usually indicates an issue."
        # Optionally fill remaining NaNs with 0 or another default?
        # x[unassigned_indices] .= 0.0
    end

    #@debug "revProb finished."
    return x
end

# Define IplpProblem structure if not defined elsewhere
# struct IplpProblem
#     c::Vector{Float64}
#     A::SparseMatrixCSC{Float64, Int}
#     b::Vector{Float64}
#     lo::Vector{Float64}
#     hi::Vector{Float64}
# end


