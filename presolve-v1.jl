using Logging
using LinearAlgebra
global_logger(ConsoleLogger(stderr, Logging.Debug)) # Set minimum level to Debug

function presolve(P)
"""
Presolve function

"""
    m, n = size(P.A)
    @debug "Presolve Start: Matrix size $(m)x$(n)"
    try
        initial_cond = cond(Matrix(P.A)) # Calculate initial condition number
        @debug "Initial Condition Number (Approx): $initial_cond"
    catch e
        @warn "Could not calculate initial condition number: $e"
    end

    # Check for infeasible bounds (l_j > u_j)
    for j = 1:n
        if P.lo[j] > P.hi[j]
            warn("Problem is infeasible due to variable bounds: lo[$j] ($(P.lo[j])) > hi[$j] ($(P.hi[j]))")
            return false
        end
    end

    ind0r = zeros(Int64,0)
    ind0c = zeros(Int64,0)
    ind_dup_r = zeros(Int64,0)
    ind_dup_c = zeros(Int64,0)
    ind_fix_c = zeros(Int64,0) # Indices of fixed columns
    fix_vals = zeros(Float64,0) # Values of fixed columns
    dup_main_c = Int64[] # Store indices of main columns in duplicate groups
    dup_del_c = Array[]
    remove_ind_row = zeros(Int64,0)
    remove_ind_col = zeros(Int64,0)

    # zero rows and columns in A
    # zero rows
    for i = 1:m
        j = 1
        while (j <= n) && (P.A[i,j] == 0.0)
            j += 1
        end

        if j == n+1
            if P.b[i] == 0.0
                ind0r = [ind0r; i]
            else
                warn("This problem is infeasible.")
                return false
            end
        end
    end
    # zero columns
    for j =1:n
        i = 1
        while (i <= m) && (P.A[i,j] == 0.0)
            i += 1
        end

        if i == m+1
            ind0c = [ind0c; j]
        end
    end

    # Fixed variables (l_j == u_j)
    for j = 1:n
        if j in ind0c # Skip zero columns
            continue
        end
        if P.lo[j] == P.hi[j]
            fixed_val = P.lo[j]
            # Substitute fixed variable value into constraints
            P.b .-= P.A[:, j] * fixed_val
            ind_fix_c = [ind_fix_c; j]
            fix_vals = [fix_vals; fixed_val]
        end
    end

    # --- Define Potential Rows/Cols (Needed earlier for Implied Free check) ---
    potential_cols = trues(n)
    potential_cols[ind0c] .= false
    potential_cols[ind_fix_c] .= false # Columns fixed by l_j=u_j

    potential_rows = trues(m)
    potential_rows[ind0r] .= false
    # Note: ind_dup_r not yet calculated, might need iterative approach later
    # For now, proceed with rows not identified as zero rows.

    # --- Forcing/Dominated Constraint & Implied Bound Check ---
    tol = 1e-8 # Tolerance for floating point checks
    ind_forcing_r = Int64[] # Rows removed because they are forcing
    potential_col_indices_forcing = findall(potential_cols)
    potential_row_indices_forcing = findall(potential_rows)

    # for i in potential_row_indices_forcing
    #     if !potential_rows[i] continue end # Skip if row removed by previous steps/iterations

    #     gi = 0.0
    #     hi = 0.0
    #     gi_is_neg_inf = false
    #     hi_is_pos_inf = false

    #     row_cols = Int64[] # Store columns involved in this row
    #     for j_idx in 1:length(potential_col_indices_forcing)
    #         j = potential_col_indices_forcing[j_idx]
    #         if !potential_cols[j] continue end # Skip if col removed

    #         aij = P.A[i, j]
    #         if aij == 0.0 continue end

    #         push!(row_cols, j) # Add j to list for this row

    #         lj = P.lo[j]
    #         uj = P.hi[j]

    #         # Calculate contribution to gi (minimum LHS value)
    #         if !gi_is_neg_inf
    #             if aij > 0
    #                 if isinf(lj) && lj < 0 gi_is_neg_inf = true; else gi += aij * lj; end
    #             else # aij < 0
    #                 if isinf(uj) && uj > 0 gi_is_neg_inf = true; else gi += aij * uj; end
    #             end
    #         end

    #         # Calculate contribution to hi (maximum LHS value)
    #         if !hi_is_pos_inf
    #             if aij > 0
    #                 if isinf(uj) && uj > 0 hi_is_pos_inf = true; else hi += aij * uj; end
    #             else # aij < 0
    #                 if isinf(lj) && lj < 0 hi_is_pos_inf = true; else hi += aij * lj; end
    #             end
    #         end
    #         # Early exit if both infinities are determined
    #         if gi_is_neg_inf && hi_is_pos_inf break end
    #     end

    #     # Check infeasibility
    #     bi = P.b[i]
    #     if !hi_is_pos_inf && hi < bi - tol
    #         @debug "Presolve Infeasibility: Row $i. Max LHS ($hi) < RHS ($bi). Tolerance: $tol"
    #         @warn("Problem determined infeasible by presolve (row $i: max LHS < RHS)")
    #         return false
    #     end
    #     if !gi_is_neg_inf && bi < gi - tol # Check if bi is significantly smaller than gi
    #         @debug "Presolve Infeasibility: Row $i. Min LHS ($gi) > RHS ($bi). Tolerance: $tol"
    #         @warn("Problem determined infeasible by presolve (row $i: min LHS > RHS)")
    #         return false
    #     end

    #     # Check for forcing constraint and fix variables
    #     is_forcing = false
    #     if !gi_is_neg_inf && abs(gi - bi) < tol
    #         is_forcing = true
    #         #println("DEBUG: Row $i forcing (min case)")
    #         for j in row_cols # Fix variables involved in this row
    #             aij = P.A[i, j] # Get aij again
    #             if aij == 0.0 continue end # Should not happen if row_cols is built correctly
    #             lj = P.lo[j]; uj = P.hi[j]
    #             fixed_val = (aij > 0) ? lj : uj
                
    #             # Check consistency and fix
    #             k_fixed_idx = findfirst(==(j), ind_fix_c)
    #             if k_fixed_idx !== nothing # Already fixed
    #                 if abs(fix_vals[k_fixed_idx] - fixed_val) > tol
    #                     @debug "Presolve Infeasibility: Forcing row $i, var $j. Conflict: existing=$(fix_vals[k_fixed_idx]), required=$fixed_val, tol=$tol"
    #                     @warn("Problem determined infeasible by presolve (forcing conflict)")
    #                     return false
    #                 end
    #                 # Consistent, do nothing
    #             else # New variable to fix
    #                 if isinf(fixed_val)
    #                     @debug "Presolve Infeasibility: Forcing row $i, var $j. Requires infinite value ($fixed_val)"
    #                     @warn("Problem determined infeasible by presolve (forcing requires infinite)")
    #                     return false
    #                 end
    #                 push!(ind_fix_c, j)
    #                 push!(fix_vals, fixed_val)
    #                 potential_cols[j] = false
    #                 #println("DEBUG: Forcing row $i fixed var $j = $fixed_val")
    #             end
    #         end
    #     elseif !hi_is_pos_inf && abs(hi - bi) < tol
    #         is_forcing = true
    #         #println("DEBUG: Row $i forcing (max case)")
    #         for j in row_cols # Fix variables involved in this row
    #             aij = P.A[i, j]
    #             if aij == 0.0 continue end
    #             lj = P.lo[j]; uj = P.hi[j]
    #             fixed_val = (aij > 0) ? uj : lj
                
    #             # Check consistency and fix
    #             k_fixed_idx = findfirst(==(j), ind_fix_c)
    #             if k_fixed_idx !== nothing # Already fixed
    #                 if abs(fix_vals[k_fixed_idx] - fixed_val) > tol
    #                     @debug "Presolve Infeasibility: Forcing row $i, var $j. Conflict: existing=$(fix_vals[k_fixed_idx]), required=$fixed_val, tol=$tol"
    #                     @warn("Problem determined infeasible by presolve (forcing conflict)")
    #                     return false
    #                 end
    #             else # New variable to fix
    #                 if isinf(fixed_val)
    #                     @debug "Presolve Infeasibility: Forcing row $i, var $j. Requires infinite value ($fixed_val)"
    #                     @warn("Problem determined infeasible by presolve (forcing requires infinite)")
    #                     return false
    #                 end
    #                 push!(ind_fix_c, j)
    #                 push!(fix_vals, fixed_val)
    #                 potential_cols[j] = false
    #                 #println("DEBUG: Forcing row $i fixed var $j = $fixed_val")
    #             end
    #         end
    #     end

    #     if is_forcing
    #         potential_rows[i] = false # Mark row for removal
    #         push!(ind_forcing_r, i)
    #         continue # Don't do implied bound check for this row
    #     end

    #     # Refined Implied Bound Check (if row is not forcing)
    #     if !gi_is_neg_inf || !hi_is_pos_inf # Only proceed if at least one bound is finite
    #         for j in row_cols # Check implied bounds for variables in this row
    #             # Skip if variable is already fixed or free
    #             if !potential_cols[j] || (isinf(P.lo[j]) && isinf(P.hi[j])) continue end
                
    #             aij = P.A[i, j]
    #             lj = P.lo[j]; uj = P.hi[j]
    #             l_ij = -Inf
    #             u_ij = Inf

    #             # Calculate u'_ij using g_i or h_i
    #             if aij > 0 && !gi_is_neg_inf
    #                 u_ij = (bi - gi) / aij + lj
    #             elseif aij < 0 && !hi_is_pos_inf
    #                 u_ij = (bi - hi) / aij + lj
    #             end

    #             # Calculate l'_ij using h_i or g_i
    #             if aij > 0 && !hi_is_pos_inf
    #                 l_ij = (bi - hi) / aij + uj
    #             elseif aij < 0 && !gi_is_neg_inf
    #                 l_ij = (bi - gi) / aij + uj
    #             end

    #             # Check if these bounds imply redundancy
    #             if !isinf(l_ij) && l_ij >= P.lo[j] - tol && !isinf(u_ij) && u_ij <= P.hi[j] + tol
    #                 #println("DEBUG: Row $i implies var $j is free. Old [$(P.lo[j]), $(P.hi[j])]. Implied [$l_ij, $u_ij]")
    #                 P.lo[j] = -Inf
    #                 P.hi[j] = +Inf
    #             end
    #         end
    #     end
    # end
    # --- End Forcing/Dominated Constraint & Implied Bound Check ---

    # # --- Free Column Singleton Substitution ---
    obj_offset = 0.0
    free_singleton_subs = Dict{Int, Tuple{Int, Float64, Dict{Int, Float64}}}()
    ind_free_singleton_r = Int64[] # Rows removed by free singleton substitution
    ind_free_singleton_c = Int64[] # Columns removed by free singleton substitution

    # Need to potentially iterate or re-check as one substitution might enable another.
    # For simplicity, let's do one pass first. A more robust implementation might loop.

    m, n = size(P.A) # Ensure n is accessible here for bounds check

    # Keep track of rows potentially active
    potential_rows = trues(m)
    potential_rows[ind0r] .= false 
    potential_rows[ind_dup_r] .= false # Start with rows not already marked

    # # Keep track of columns potentially active (moved definition just before use)
    potential_cols = trues(n)
    potential_cols[ind0c] .= false
    potential_cols[ind_fix_c] .= false # Columns fixed by l_j=u_j

    # Iterate directly over potential columns
    for j in findall(potential_cols)
        # Skip if already processed in this pass or no longer potential
        if !potential_cols[j]
            continue
        end
        
        # Check if variable j is free
        if isinf(P.lo[j]) && isinf(P.hi[j])
            # Find non-zeros in this column among potential rows
            nnz_count = 0
            singleton_row_k = -1
            rows_to_check = findall(potential_rows)
            for k_check in rows_to_check
                if P.A[k_check, j] != 0.0
                    nnz_count += 1
                    singleton_row_k = k_check
                end
                if nnz_count > 1
                    break
                end
            end

            # If it's a free column singleton
            if nnz_count == 1
                k = singleton_row_k
                # Double check row k is still potential (should be if nnz_count was 1)
                if !potential_rows[k]
                    continue # Should not happen
                end

                akj = P.A[k, j]
                cj = P.c[j]
                bk = P.b[k] # Use the potentially modified b from fixed var substitution
                
                factor = cj / akj
                obj_offset += factor * bk
                
                substitution_details = Dict{Int, Float64}()
                # Update objective coefficients c_i for i != j
                other_potential_cols = findall(potential_cols)
                for i in other_potential_cols
                    if i == j
                        continue
                    end
                    aki = P.A[k, i]
                    if aki != 0.0
                        P.c[i] -= factor * aki # Modify cost vector
                        substitution_details[i] = aki
                    end
                end
                
                # Store substitution info
                free_singleton_subs[j] = (k, akj, substitution_details)
                
                # Mark row k and column j for removal
                potential_rows[k] = false
                potential_cols[j] = false
                push!(ind_free_singleton_r, k)
                push!(ind_free_singleton_c, j)
            end
        end
    end
    # --- End Free Column Singleton ---

    # --- Singleton Row Identification and Processing ---
    ind_singleton_r = Int64[] # Rows removed by singleton processing
    
    # Calculate initial nnz per row, considering only potential columns
    nnz_row = zeros(Int, m)
    active_A = P.A[potential_rows, potential_cols] 
    # Need to map back indices if using active_A, safer to loop original A
    
    # Map original column indices to indices within potential_cols
    potential_col_indices = findall(potential_cols)
    map_potential_col_to_original = Dict(zip(1:length(potential_col_indices), potential_col_indices))

    # Map original row indices to indices within potential_rows
    potential_row_indices = findall(potential_rows)
    map_potential_row_to_original = Dict(zip(1:length(potential_row_indices), potential_row_indices))
    map_original_row_to_potential = Dict(zip(potential_row_indices, 1:length(potential_row_indices)))

    singleton_list = Tuple{Int, Int}[] # (original_row_idx, original_col_idx)

    for orig_row_idx in potential_row_indices
        count = 0
        last_k = -1
        for orig_col_idx in potential_col_indices
            if P.A[orig_row_idx, orig_col_idx] != 0.0
                count += 1
                last_k = orig_col_idx
            end
        end
        nnz_row[orig_row_idx] = count
        if count == 1
            push!(singleton_list, (orig_row_idx, last_k))
        end
    end

    processed_singletons = 0
    while !isempty(singleton_list)
        processed_singletons += 1
        # Safety break for potential infinite loops in complex cases
        if processed_singletons > m + n 
            warn("Singleton processing loop exceeded limit, breaking.")
            break 
        end

        (i, k) = popfirst!(singleton_list)

        # Skip if row i or column k has been processed/fixed in the meantime
        if !potential_rows[i] || !potential_cols[k]
            continue
        end

        aik = P.A[i, k]
        if aik == 0.0 # Should not happen if nnz count was correct, but safety check
             warn("Singleton row $i found zero value at supposed non-zero column $k")
             continue
        end
        
        fixed_val = P.b[i] / aik

        # Check bounds
        if fixed_val < P.lo[k] - 1e-8 || fixed_val > P.hi[k] + 1e-8 # Add tolerance
            warn("Problem is infeasible due to singleton row $i fixing var $k=$fixed_val outside bounds [$(P.lo[k]), $(P.hi[k])]")
            return false
        end

        # Check consistency if k was already fixed by l_j == u_j
        k_fixed_idx = findfirst(==(k), ind_fix_c)
        if k_fixed_idx !== nothing
            if abs(fix_vals[k_fixed_idx] - fixed_val) > 1e-8 # Add tolerance
                 warn("Problem is infeasible due to conflicting fixed values for var $k (singleton row $i vs bounds/previous)")
                 return false
            end
             # Value is consistent, no need to re-fix, but proceed to remove row & update nnz
        else
            # Fix variable k 
            push!(ind_fix_c, k)
            push!(fix_vals, fixed_val)
            potential_cols[k] = false # Mark column k as fixed now
        end

        # Mark row i as processed
        potential_rows[i] = false
        push!(ind_singleton_r, i) 
        
        # Update RHS b and nnz counts for other rows affected by fixing k
        # Iterate over rows that are still potential and not the current row i
        for r in potential_row_indices 
             if r == i || !potential_rows[r] # Skip self or already processed rows
                 continue
             end
            
            ark = P.A[r, k]
            if ark != 0.0
                P.b[r] -= ark * fixed_val # Update RHS
                
                # Decrement nnz count for row r
                nnz_row[r] -= 1
                
                # If row r becomes a new singleton
                if nnz_row[r] == 1
                    # Find the remaining single non-zero column k_new in row r among potential_cols
                    k_new = -1
                    for kk in potential_col_indices # Iterate original indices
                         if potential_cols[kk] && P.A[r, kk] != 0.0 # Check if column is still active
                             if k_new != -1 # Should only find one
                                 warn("Error identifying new singleton: Row $r has >1 potential non-zeros after fixing $k.")
                                 k_new = -2 # Mark as error
                                 break
                             end
                             k_new = kk
                         end
                    end
                    
                    if k_new > 0 # Found a valid new singleton
                         # Check if already in list to avoid duplicates? Unlikely but possible
                         if !any(item -> item == (r, k_new), singleton_list)
                            push!(singleton_list, (r, k_new))
                         end
                    elseif k_new == -1
                         # This means nnz_row[r] became 1, but the last element was just fixed (k)
                         # This can happen if the row *only* had k. It should become a zero row.
                         # Or, if the nnz calculation logic/update has an issue.
                         # Let's recalculate nnz for this row to be sure
                         current_nnz_r = 0
                         for kk_check in potential_col_indices
                             if potential_cols[kk_check] && P.A[r, kk_check] != 0.0
                                 current_nnz_r += 1
                             end
                         end
                         nnz_row[r] = current_nnz_r # Correct the count
                         if current_nnz_r == 0 && abs(P.b[r]) > 1e-8 # Check for infeasibility from zero row
                            warn("Problem is infeasible due to row $r becoming 0 = $(P.b[r]) after singleton fix.")
                            return false
                         elseif current_nnz_r == 0
                             # Row became all zero and RHS is zero, mark for removal
                             potential_rows[r] = false
                             # Should we add to ind0r? Or just let it be handled by potential_rows?
                             # Let potential_rows handle it for now.
                         end
                         # If current_nnz_r is still 1 after recalc, something is wrong.
                    end
                elseif nnz_row[r] == 0 && abs(P.b[r]) > 1e-8 # Check for infeasibility if row becomes zero
                     warn("Problem is infeasible due to row $r becoming 0 = $(P.b[r]) after singleton fix.")
                     return false
                elseif nnz_row[r] == 0 
                     # Row became all zero and RHS is zero, mark for removal implicitly by potential_rows
                     potential_rows[r] = false
                     # No need to add to ind0r explicitly if we filter by potential_rows later
                end 
            end
        end # End loop updating other rows
    end # End while singleton_list not empty
    # --- End Singleton Processing ---

    # duplicated rows - Now needs to consider only potential rows/cols
    for i=1:(m-1)
        # Skip if row i is not potential or already in a duplicate set
        if !potential_rows[i] || (i in ind_dup_r) 
            continue
        end

        for j=(i+1):m
             # Skip if row j is not potential or already in a duplicate set
            if !potential_rows[j] || (j in ind_dup_r)
                continue
            end

            k = 1
            match = true
            # Iterate through potential columns only
            for col_idx in potential_col_indices # Use original indices stored here
                if !potential_cols[col_idx] # Skip if column was fixed by singleton
                    continue
                end
                if P.A[i, col_idx] != P.A[j, col_idx]
                    match = false
                    break
                end
            end


            if match # Rows match over potential columns
                if abs(P.b[i] - P.b[j]) < 1e-8 # Use tolerance for float comparison
                    ind_dup_r = [ind_dup_r; j]
                    potential_rows[j] = false # Mark row j as removed
                else
                    warn("This problem is infeasible due to duplicated rows with different RHS.")
                    return false
                end
            end

        end
    end
    # Combine initial zero rows, duplicates, singletons, free singletons, and forcing rows
    @show remove_ind_row = unique([ind0r; ind_dup_r; ind_singleton_r; ind_free_singleton_r; ind_forcing_r]) 

    # duplicate columns 
    # Needs modification to consider only potential rows/cols
    processed_cols = falses(n) # Track columns already part of a duplicate group
    for i_idx in 1:length(potential_col_indices)
        i = potential_col_indices[i_idx]
        if !potential_cols[i] || processed_cols[i] || i in ind0c # Skip if fixed, processed, or zero col
            continue
        end
        
        dup_item_indices = [i] # Store original indices
        
        for j_idx in (i_idx+1):length(potential_col_indices)
            j = potential_col_indices[j_idx]
            if !potential_cols[j] || processed_cols[j] || j in ind0c # Skip if fixed, processed, or zero col
                 continue
            end

            # Compare columns P.A[:,i] and P.A[:,j] over potential rows
            match = true
            for row_idx in potential_row_indices
                 if !potential_rows[row_idx] # Skip removed rows
                     continue
                 end
                 if P.A[row_idx, i] != P.A[row_idx, j]
                     match = false
                     break
                 end
            end

            if match
                push!(dup_item_indices, j)
            end
        end

        if length(dup_item_indices) > 1
            minv, mini_local = findmin(P.c[dup_item_indices])
            main_col_idx = dup_item_indices[mini_local]
            
            push!(dup_main_c, main_col_idx) # Store the index of the main column kept

            for col_idx in dup_item_indices
                 processed_cols[col_idx] = true # Mark all in group as processed
                 if col_idx != main_col_idx
                     # This column is redundant
                     push!(ind_dup_c, col_idx)
                     potential_cols[col_idx] = false # Mark for removal

                     # Transfer bounds if needed? Original code didn't seem to. Assuming not needed.
                     # The variable corresponding to col_idx will be set to 0 in revProb.
                 end
            end
        else
            processed_cols[i] = true # Mark as processed even if no duplicates found
        end
    end
    # Combine initial zero cols, duplicates, fixed cols (bounds+singletons), and free singletons
    @show remove_ind_col = unique([ind0c; ind_dup_c; ind_fix_c; ind_free_singleton_c])


    # Update flags based on final potential status
    flags_row = potential_rows
    flags_col = potential_cols # These now reflect all removals

    # Ensure removed indices lists match the flags (for consistency, though flags are now primary)
    remove_ind_row = findall(.!flags_row)
    remove_ind_col = findall(.!flags_col)


    # The error occurs because we're trying to index vectors with a BitVector
    # We need to convert the BitVector to indices using findall()
    # This returns the indices where flags_row/flags_col are true
    row_indices = findall(flags_row)
    col_indices = findall(flags_col)

    # --- Column Singleton Dual Bound Derivation ---
    dual_lb = fill(-Inf, m) # Lower bounds on dual variables y
    dual_ub = fill(Inf, m)  # Upper bounds on dual variables y
    col_singleton_indices = Set{Int}() # Store indices of cols identified as singletons

    for j in col_indices # Iterate through columns remaining in the presolved problem
        nnz_count = 0
        singleton_row_k = -1
        
        # Count non-zeros in this column among active rows
        for k in row_indices # Iterate through rows remaining in the presolved problem
            if P.A[k, j] != 0.0
                nnz_count += 1
                singleton_row_k = k
            end
            if nnz_count > 1 # Optimization: stop counting if more than one found
                break
            end
        end

        # If it's a column singleton
        if nnz_count == 1
            push!(col_singleton_indices, j) # Record this column index
            k = singleton_row_k
            akj = P.A[k, j]
            lj = P.lo[j]
            uj = P.hi[j]
            cj = P.c[j]
            
            # Apply table logic (using original bounds/cost for column j)
            if !isinf(lj) && isinf(uj) # l > -inf, u = +inf
                if akj > 0
                    dual_ub[k] = min(dual_ub[k], cj / akj)
                elseif akj < 0
                    dual_lb[k] = max(dual_lb[k], cj / akj)
                end
            elseif isinf(lj) && !isinf(uj) # l = -inf, u < +inf
                if akj > 0
                    dual_lb[k] = max(dual_lb[k], cj / akj)
                elseif akj < 0
                    dual_ub[k] = min(dual_ub[k], cj / akj)
                end
            # Case l=-inf, u=+inf, akj!=0 doesn't add bounds here but confirms y_k*akj=cj
            # Case l>-inf, u<+inf is a bounded variable, doesn't imply simple dual bound from this rule
            end
        end
    end
    # --- End Column Singleton Processing ---

    # --- Dominated Column Check (using Dual Bounds) ---
    potential_col_indices_dom = findall(potential_cols) # Get current potential cols
    potential_row_indices_dom = findall(potential_rows) # Get current potential rows

    for j in potential_col_indices_dom
        if !potential_cols[j] continue end # Re-check if fixed by forcing/etc.

        cj = P.c[j]
        lj = P.lo[j]
        uj = P.hi[j]

        ej = 0.0 # Min value of sum(aij * yi)
        dj = 0.0 # Max value of sum(aij * yi)
        ej_is_neg_inf = false
        dj_is_pos_inf = false

        for i in potential_row_indices_dom
            if !potential_rows[i] continue end # Should not happen if indices are up-to-date

            aij = P.A[i, j]
            if aij == 0.0 continue end

            # Use the calculated dual bounds (potentially Inf)
            ylb = dual_lb[i]
            yub = dual_ub[i]

            # Calculate ej component
            if !ej_is_neg_inf
                term = 0.0
                inf_term = false
                if aij > 0 # Min contribution uses ylb
                    if isinf(ylb) && ylb < 0 inf_term = true; else term = aij * ylb; end
                else # aij < 0, Min contribution uses yub
                    if isinf(yub) && yub > 0 inf_term = true; else term = aij * yub; end
                end
                if inf_term ej_is_neg_inf = true; else ej += term; end
            end

            # Calculate dj component
            if !dj_is_pos_inf
                term = 0.0
                inf_term = false
                if aij > 0 # Max contribution uses yub
                    if isinf(yub) && yub > 0 inf_term = true; else term = aij * yub; end
                else # aij < 0, Max contribution uses ylb
                    if isinf(ylb) && ylb < 0 inf_term = true; else term = aij * ylb; end
                end
                if inf_term dj_is_pos_inf = true; else dj += term; end
            end
            if ej_is_neg_inf && dj_is_pos_inf break end # Optimization
        end

        # Check reduced cost bounds: cj - dj <= z*j <= cj - ej

        # Case 1: Lower bound on z*j is positive? (cj - dj > tol)
        if !dj_is_pos_inf && cj - dj > tol
            # Implies z*j > 0, so x*j must be lj
            if isinf(lj) && lj < 0
                @warn("Problem is likely UNBOUNDED (dominated column $j requires lower bound -Inf, z* > 0)")
                return :Unbounded # Indicate unboundedness
            end
            fixed_val = lj
            # Check consistency and fix
            k_fixed_idx = findfirst(==(j), ind_fix_c)
            if k_fixed_idx !== nothing # Already fixed
                if abs(fix_vals[k_fixed_idx] - fixed_val) > tol
                    @warn("Problem infeasible: Dominated column rule for var $j conflicts with previous fixed value")
                    return false # Indicate infeasibility
                end
                # Consistent, already fixed
            else # New variable to fix
                #println("DEBUG: Fixing var $j = $fixed_val due to dominated column (z* > 0)")
                push!(ind_fix_c, j)
                push!(fix_vals, fixed_val)
                P.b .-= P.A[:, j] * fixed_val # Update RHS
                potential_cols[j] = false # Mark as removed
            end

        # Case 2: Upper bound on z*j is negative? (cj - ej < -tol)
        elseif !ej_is_neg_inf && cj - ej < -tol
            # Implies z*j < 0, so x*j must be uj
            if isinf(uj) && uj > 0
                @warn("Problem is likely UNBOUNDED (dominated column $j requires upper bound +Inf, z* < 0)")
                return :Unbounded # Indicate unboundedness
            end
            fixed_val = uj
            # Check consistency and fix
            k_fixed_idx = findfirst(==(j), ind_fix_c)
            if k_fixed_idx !== nothing # Already fixed
                if abs(fix_vals[k_fixed_idx] - fixed_val) > tol
                    @warn("Problem infeasible: Dominated column rule for var $j conflicts with previous fixed value")
                    return false # Indicate infeasibility
                end
            else # New variable to fix
                #println("DEBUG: Fixing var $j = $fixed_val due to dominated column (z* < 0)")
                push!(ind_fix_c, j)
                push!(fix_vals, fixed_val)
                P.b .-= P.A[:, j] * fixed_val # Update RHS
                potential_cols[j] = false # Mark as removed
            end

        # Case 3 & 4: Weakly Dominated (Reduced cost is zero, check bounds)
        # Apply only if j was NOT a column singleton used for dual bounds
        elseif j ∉ col_singleton_indices 
            # Case 3: Lower bound on z*j is zero? (cj - dj = 0) and finite lower bound
            if !dj_is_pos_inf && abs(cj - dj) < tol && !isinf(lj)
                # Implies z*j >= 0. If x*j > lj, then z*j must be 0. Fix at lj.
                fixed_val = lj
                # Check consistency and fix
                k_fixed_idx = findfirst(==(j), ind_fix_c)
                if k_fixed_idx !== nothing # Already fixed
                    if abs(fix_vals[k_fixed_idx] - fixed_val) > tol
                        @warn("Problem infeasible: Weakly dominated column rule (>=0) for var $j conflicts with previous fixed value")
                        return false # Indicate infeasibility
                    end
                else # New variable to fix
                    #println("DEBUG: Fixing var $j = $fixed_val due to weakly dominated column (z* >= 0)")
                    push!(ind_fix_c, j)
                    push!(fix_vals, fixed_val)
                    P.b .-= P.A[:, j] * fixed_val # Update RHS
                    potential_cols[j] = false # Mark as removed
                end
            
            # Case 4: Upper bound on z*j is zero? (cj - ej = 0) and finite upper bound
            elseif !ej_is_neg_inf && abs(cj - ej) < tol && !isinf(uj)
                 # Implies z*j <= 0. If x*j < uj, then z*j must be 0. Fix at uj.
                fixed_val = uj
                # Check consistency and fix
                k_fixed_idx = findfirst(==(j), ind_fix_c)
                if k_fixed_idx !== nothing # Already fixed
                    if abs(fix_vals[k_fixed_idx] - fixed_val) > tol
                        @warn("Problem infeasible: Weakly dominated column rule (<=0) for var $j conflicts with previous fixed value")
                        return false # Indicate infeasibility
                    end
                else # New variable to fix
                    #println("DEBUG: Fixing var $j = $fixed_val due to weakly dominated column (z* <= 0)")
                    push!(ind_fix_c, j)
                    push!(fix_vals, fixed_val)
                    P.b .-= P.A[:, j] * fixed_val # Update RHS
                    potential_cols[j] = false # Mark as removed
                end
            end
        end
    end
    # --- End Dominated Column Check ---

    # Recalculate final row/col indices based on potential_flags *after* all checks
    flags_row = potential_rows
    flags_col = potential_cols 
    row_indices = findall(flags_row)
    col_indices = findall(flags_col)

    # Ensure final removed indices lists are consistent (for @show)
    remove_ind_row = findall(.!flags_row)
    remove_ind_col = findall(.!flags_col)
    @show remove_ind_row
    @show remove_ind_col

    # The error occurs because we're trying to index vectors with a BitVector
    # We need to convert the BitVector to indices using findall()
    # This returns the indices where flags_row/flags_col are true
    row_indices = findall(flags_row)
    col_indices = findall(flags_col)

    # Construct the final presolved matrix
    final_A = P.A[row_indices, col_indices]
    m_final, n_final = size(final_A)
    @debug "Presolve End: Final Matrix size $(m_final)x$(n_final)"
    try
        final_cond = cond(Matrix(final_A)) # Calculate final condition number
        @debug "Final Condition Number (Approx): $final_cond"
    catch e
         @warn "Could not calculate final condition number (matrix might be empty or singular): $e"
         # If the matrix is singular, cond() might throw an error or return Inf
         if isempty(final_A)
             @debug "Final matrix is empty."
         elseif m_final >= n_final && rank(Matrix(final_A)) < n_final
             @warn "Final matrix appears to be singular (rank < number of columns)."
         elseif m_final < n_final && rank(Matrix(final_A)) < m_final
              @warn "Final matrix appears to be singular (rank < number of rows)."
         end
    end

    # Need to create the new problem using the *final* P.b potentially modified by substitutions
    # Also return a success status indicator
    return :Success, IplpProblem(P.c[col_indices], final_A,
     P.b[row_indices], P.lo[col_indices], P.hi[col_indices]), ind0c, dup_main_c, ind_dup_c, ind_fix_c, fix_vals, dual_lb, dual_ub, obj_offset, free_singleton_subs
end

function revProb(P, ind0c, dup_main_c, ind_dup_c, ind_fix_c, fix_vals, free_singleton_subs, x1)
    m_orig, n_orig = size(P.A) # Use original problem dimensions P passed to revProb
    x = Array{Float64}(undef, n_orig)
    fill!(x, NaN) # Use NaN to indicate unassigned instead of Inf, easier debug

    x1_idx = 1 # Index for the solution vector x1 from the reduced problem
    fix_val_map = Dict(zip(ind_fix_c, fix_vals)) # Map fixed indices to values
    dup_main_map = Dict(zip(dup_main_c, dup_main_c)) # For quick lookup if a column is a main duplicate

    # Create a lookup for which main column corresponds to a removed duplicate column
    dup_del_to_main_map = Dict{Int, Int}()
    # This requires the original structure of dup_main_c and ind_dup_c before unique() was potentially called
    # The presolve function needs modification to return this structure if we want perfect reconstruction for duplicates.
    # For now, the old logic sets duplicates to 0, let's stick to that unless specified otherwise.


    active_indices_presolved = # We need the mapping from x1 indices back to original indices
        let 
            flags_col = trues(n_orig)
            # Combine ALL removed/fixed column indices before solving
            removed_cols = unique([ind0c; ind_dup_c; ind_fix_c; collect(keys(free_singleton_subs))]) 
            flags_col[removed_cols] .= false
            findall(flags_col)
        end
    
    if length(x1) != length(active_indices_presolved)
         error("Mismatch between solved variables ($(length(x1))) and expected active columns ($(length(active_indices_presolved)))")
    end
    
    # Assign solved values
    for (idx, orig_idx) in enumerate(active_indices_presolved)
        x[orig_idx] = x1[idx]
    end

    for i = 1:n_orig
        # Assign fixed, zero, duplicate values ONLY if not already assigned by x1
        if isnan(x[i])
            if haskey(fix_val_map, i)
                x[i] = fix_val_map[i]
            elseif i in ind0c
                # Handle zero columns based on cost
                if P.c[i] > 0
                    x[i] = P.lo[i] # Assumes P here is the *original* problem
                elseif P.c[i] < 0
                    x[i] = P.hi[i] # Assumes P here is the *original* problem
                else
                    x[i] = 0.0 # Or P.lo[i] or P.hi[i] if non-zero? Typically 0.
                end
            elseif i in ind_dup_c 
                # This column was removed as a duplicate. Set to 0?
                x[i] = 0.0 
            # Free singletons are handled next
            # elseif haskey(free_singleton_subs, i) 
            #     # Handled in the loop below
            #     continue
            # else
            #     # This case should ideally not be reached if i wasn't fixed, zero, dup, or solved.
            #     warn("Variable $i was not assigned during initial phase of revProb.")
            end
        end
    end
    
    # Calculate values for substituted free column singletons
    # Loop until no more progress can be made (handles dependencies)
    max_passes = n_orig # Safety break
    current_pass = 0
    made_progress = true
    free_singleton_cols = collect(keys(free_singleton_subs))
    
    while made_progress && current_pass < max_passes
        made_progress = false
        current_pass += 1
        
        for j in free_singleton_cols
            if isnan(x[j]) # Only calculate if not already done
                (k, akj, subs_dict) = free_singleton_subs[j]
                
                sum_term = 0.0
                dependencies_met = true
                for (i, aki) in subs_dict
                    if isnan(x[i])
                        dependencies_met = false
                        break # Cannot calculate yet
                    end
                    sum_term += aki * x[i]
                end
                
                if dependencies_met
                    bk = P.b[k] # Use original b[k]
                    x[j] = (bk - sum_term) / akj
                    made_progress = true
                end
            end
        end
    end
    
    # Check if all free singletons were resolved
    for j in free_singleton_cols
        if isnan(x[j])
             warn("Could not resolve free singleton variable $j in revProb, likely circular dependency or error.")
             # Assign NaN or a default? Let's leave NaN for now.
        end
    end

    # Final check: ensure all original variables are assigned
    unassigned_indices = findall(isnan, x)
    if !isempty(unassigned_indices)
        error("Not all variables were assigned in revProb. Unassigned indices: $(unassigned_indices)")
    end

    return x
end
