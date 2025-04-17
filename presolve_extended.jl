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

    # --- Define Potential Rows/Cols --- 
    potential_cols = trues(n)
    potential_cols[ind0c] .= false
    potential_cols[ind_fix_c] .= false # Columns fixed by l_j=u_j

    potential_rows = trues(m)
    potential_rows[ind0r] .= false

    # --- Forcing/Dominated Constraint & Implied Bound Check ---
    tol = 1e-8 # Tolerance for floating point checks
    ind_forcing_r = Int64[] # Rows removed because they are forcing
    potential_col_indices_forcing = findall(potential_cols)
    potential_row_indices_forcing = findall(potential_rows)

    for i in potential_row_indices_forcing
        if !potential_rows[i] continue end # Skip if row removed

        gi = 0.0
        hi = 0.0
        gi_is_neg_inf = false
        hi_is_pos_inf = false

        row_cols = Int64[] # Store columns involved in this row
        for j_idx in 1:length(potential_col_indices_forcing)
            j = potential_col_indices_forcing[j_idx]
            if !potential_cols[j] continue end # Skip if col removed

            aij = P.A[i, j]
            if aij == 0.0 continue end

            push!(row_cols, j)

            lj = P.lo[j]
            uj = P.hi[j]

            # Calculate contribution to gi (minimum LHS value)
            if !gi_is_neg_inf
                if aij > 0
                    if isinf(lj) && lj < 0 gi_is_neg_inf = true; else gi += aij * lj; end
                else # aij < 0
                    if isinf(uj) && uj > 0 gi_is_neg_inf = true; else gi += aij * uj; end
                end
            end

            # Calculate contribution to hi (maximum LHS value)
            if !hi_is_pos_inf
                if aij > 0
                    if isinf(uj) && uj > 0 hi_is_pos_inf = true; else hi += aij * uj; end
                else # aij < 0
                    if isinf(lj) && lj < 0 hi_is_pos_inf = true; else hi += aij * lj; end
                end
            end
            # Early exit if both infinities are determined
            if gi_is_neg_inf && hi_is_pos_inf break end
        end

        # Check infeasibility
        bi = P.b[i]
        if !hi_is_pos_inf && hi < bi - tol
            @debug "Presolve Infeasibility: Row $i. Max LHS ($hi) < RHS ($bi). Tolerance: $tol"
            @warn("Problem determined infeasible by presolve (row $i: max LHS < RHS)")
            return false
        end
        if !gi_is_neg_inf && bi < gi - tol # Check if bi is significantly smaller than gi
            @debug "Presolve Infeasibility: Row $i. Min LHS ($gi) > RHS ($bi). Tolerance: $tol"
            @warn("Problem determined infeasible by presolve (row $i: min LHS > RHS)")
            return false
        end

        # Check for forcing constraint and fix variables
        is_forcing = false
        # Guard against infinities before checking tolerance
        if !gi_is_neg_inf && !hi_is_pos_inf && abs(gi - bi) < tol 
            is_forcing = true
            #@debug "Row $i forcing (min case)"
            for j in row_cols # Fix variables involved in this row
                aij = P.A[i, j] 
                if aij == 0.0 continue end 
                lj = P.lo[j]; uj = P.hi[j]
                fixed_val = (aij > 0) ? lj : uj
                
                # Check consistency and fix
                k_fixed_idx = findfirst(==(j), ind_fix_c)
                if k_fixed_idx !== nothing # Already fixed
                    if abs(fix_vals[k_fixed_idx] - fixed_val) > tol
                        @debug "Presolve Infeasibility: Forcing row $i, var $j. Conflict: existing=$(fix_vals[k_fixed_idx]), required=$fixed_val, tol=$tol"
                        @warn("Problem determined infeasible by presolve (forcing conflict)")
                        return false
                    end
                    # Consistent, do nothing
                else # New variable to fix
                    if isinf(fixed_val)
                        @debug "Presolve Infeasibility: Forcing row $i, var $j. Requires infinite value ($fixed_val)"
                        @warn("Problem determined infeasible by presolve (forcing requires infinite)")
                        return false
                    end
                    push!(ind_fix_c, j)
                    push!(fix_vals, fixed_val)
                    potential_cols[j] = false
                    #@debug "Forcing row $i fixed var $j = $fixed_val"
                end
            end
        # Guard against infinities before checking tolerance
        elseif !gi_is_neg_inf && !hi_is_pos_inf && abs(hi - bi) < tol 
            is_forcing = true
            #@debug "Row $i forcing (max case)"
            for j in row_cols # Fix variables involved in this row
                aij = P.A[i, j]
                if aij == 0.0 continue end
                lj = P.lo[j]; uj = P.hi[j]
                fixed_val = (aij > 0) ? uj : lj
                
                # Check consistency and fix
                k_fixed_idx = findfirst(==(j), ind_fix_c)
                if k_fixed_idx !== nothing # Already fixed
                    if abs(fix_vals[k_fixed_idx] - fixed_val) > tol
                        @debug "Presolve Infeasibility: Forcing row $i, var $j. Conflict: existing=$(fix_vals[k_fixed_idx]), required=$fixed_val, tol=$tol"
                        @warn("Problem determined infeasible by presolve (forcing conflict)")
                        return false
                    end
                else # New variable to fix
                    if isinf(fixed_val)
                        @debug "Presolve Infeasibility: Forcing row $i, var $j. Requires infinite value ($fixed_val)"
                        @warn("Problem determined infeasible by presolve (forcing requires infinite)")
                        return false
                    end
                    push!(ind_fix_c, j)
                    push!(fix_vals, fixed_val)
                    potential_cols[j] = false
                    #@debug "Forcing row $i fixed var $j = $fixed_val"
                end
            end
        end

        if is_forcing
            # potential_rows[i] = false # Mark row for removal <-- Keep commented out for robustness
            push!(ind_forcing_r, i)
            continue # Don't do implied bound check for this row
        end

        # Refined Implied Bound Check (if row is not forcing)
        if !gi_is_neg_inf || !hi_is_pos_inf # Only proceed if at least one bound is finite
            for j in row_cols # Check implied bounds for variables in this row
                # Skip if variable is already fixed or free
                if !potential_cols[j] || (isinf(P.lo[j]) && isinf(P.hi[j])) continue end
                
                aij = P.A[i, j]
                lj = P.lo[j]; uj = P.hi[j]
                l_ij = -Inf
                u_ij = Inf

                # Guard against division by near-zero
                if abs(aij) > 1e-10 
                    # Calculate u'_ij using g_i or h_i
                    if aij > 0 && !gi_is_neg_inf
                        u_ij = (bi - gi) / aij + lj
                    elseif aij < 0 && !hi_is_pos_inf
                        u_ij = (bi - hi) / aij + lj
                    end

                    # Calculate l'_ij using h_i or g_i
                    if aij > 0 && !hi_is_pos_inf
                        l_ij = (bi - hi) / aij + uj
                    elseif aij < 0 && !gi_is_neg_inf
                        l_ij = (bi - gi) / aij + uj
                    end
                else
                    # @debug "Skipping implied bound calculation for var $j in row $i due to near-zero aij ($aij)"
                end

                # Check if these bounds imply redundancy (variable becomes free)
                if !isinf(l_ij) && l_ij >= P.lo[j] - tol && !isinf(u_ij) && u_ij <= P.hi[j] + tol
                    #@debug "Row $i implies var $j is free. Old [$(P.lo[j]), $(P.hi[j])]. Implied [$l_ij, $u_ij]"
                    P.lo[j] = -Inf
                    P.hi[j] = +Inf
                end
            end
        end
    end
    # --- End Forcing/Dominated Constraint & Implied Bound Check ---

    # --- Free Column Singleton Substitution ---
    obj_offset = 0.0
    free_singleton_subs = Dict{Int, Tuple{Int, Float64, Dict{Int, Float64}}}()
    ind_free_singleton_r = Int64[] # Rows removed by free singleton substitution
    ind_free_singleton_c = Int64[] # Columns removed by free singleton substitution

    m, n = size(P.A) # Re-get potentially changed dimensions?

    # Need potential_rows/cols to be up-to-date here if forcing check modified them.
    # Rebuild active sets or ensure flags are consistent.
    potential_row_indices = findall(potential_rows)
    potential_col_indices = findall(potential_cols)

    # Iterate directly over potential columns
    for j in potential_col_indices
        if !potential_cols[j] continue end # Double check, might have been fixed by forcing
        
        # Check if variable j is free
        if isinf(P.lo[j]) && isinf(P.hi[j])
            # Find non-zeros in this column among potential rows
            nnz_count = 0
            singleton_row_k = -1
            rows_to_check = findall(potential_rows) # Use current potential rows
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
                if !potential_rows[k] continue end # Row k must still be potential

                akj = P.A[k, j]
                cj = P.c[j]
                bk = P.b[k] # Use the potentially modified b
                
                # Guard against division by near-zero
                if abs(akj) > 1e-10
                    factor = cj / akj
                    obj_offset += factor * bk
                    
                    substitution_details = Dict{Int, Float64}()
                    # Update objective coefficients c_i for i != j
                    # Iterate over current potential columns again
                    for i_sub in findall(potential_cols) 
                        if i_sub == j continue end

                        aki = P.A[k, i_sub]
                        if aki != 0.0
                            P.c[i_sub] -= factor * aki # Modify cost vector
                            substitution_details[i_sub] = aki
                        end
                    end
                    
                    # Store substitution info
                    free_singleton_subs[j] = (k, akj, substitution_details)
                    
                    # Mark row k and column j for removal
                    potential_rows[k] = false
                    potential_cols[j] = false
                    push!(ind_free_singleton_r, k)
                    push!(ind_free_singleton_c, j)
                else
                     @debug "Skipping free column singleton substitution for var $j (row $k) due to near-zero akj ($akj)"
                end
            end
        end
    end
    # --- End Free Column Singleton ---

    # --- Singleton Row Identification and Processing ---
    ind_singleton_r = Int64[] # Rows removed by singleton processing
    
    # Need potential_rows/cols to be up-to-date here.
    potential_row_indices = findall(potential_rows)
    potential_col_indices = findall(potential_cols)

    # Calculate initial nnz per row, considering only *currently* potential columns
    nnz_row = zeros(Int, m)
    singleton_list = Tuple{Int, Int}[] # (original_row_idx, original_col_idx)

    for orig_row_idx in potential_row_indices
        count = 0
        last_k = -1
        for orig_col_idx in potential_col_indices
            if P.A[orig_row_idx, orig_col_idx] != 0.0
                count += 1
                last_k = orig_col_idx
            end
            # Optimization: break if count > 1? 
            # if count > 1 break end 
        end
        nnz_row[orig_row_idx] = count
        if count == 1 && last_k != -1 # Ensure last_k was actually assigned
            push!(singleton_list, (orig_row_idx, last_k))
        end
    end

    processed_singletons = 0
    while !isempty(singleton_list)
        processed_singletons += 1
        # Safety break for potential infinite loops
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
        if aik == 0.0 # Should not happen if nnz count was correct
             warn("Singleton row $i found zero value at supposed non-zero column $k")
             continue
        end
        
        # Guard against division by near-zero
        if abs(aik) > 1e-10
            fixed_val = P.b[i] / aik

            # Check bounds
            if fixed_val < P.lo[k] - tol || fixed_val > P.hi[k] + tol # Use tol consistently
                warn("Problem is infeasible due to singleton row $i fixing var $k=$fixed_val outside bounds [$(P.lo[k]), $(P.hi[k])]")
                return false
            end

            # Check consistency if k was already fixed 
            k_fixed_idx = findfirst(==(k), ind_fix_c)
            if k_fixed_idx !== nothing
                if abs(fix_vals[k_fixed_idx] - fixed_val) > tol # Use tol consistently
                    warn("Problem is infeasible due to conflicting fixed values for var $k (singleton row $i vs bounds/previous)")
                    return false
                end
                 # Consistent, proceed to remove row & update nnz
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
            for r in potential_row_indices # Iterate over original potential indices 
                 if r == i || !potential_rows[r] # Skip self or already processed rows
                     continue
                 end
                
                ark = P.A[r, k]
                if ark != 0.0
                    P.b[r] -= ark * fixed_val # Update RHS
                    
                    # Update nnz count for row r and check if it becomes a new singleton
                    nnz_row[r] -= 1 # Assuming nnz_row contains counts based on original potential_cols
                        
                    if nnz_row[r] == 1
                        # Find the remaining single non-zero column k_new in row r among *current* potential_cols
                        k_new = -1
                        # Need to iterate through current potential cols, not original list
                        for kk in findall(potential_cols) 
                             if P.A[r, kk] != 0.0 
                                 if k_new != -1 # Should only find one
                                     warn("Error identifying new singleton: Row $r has >1 potential non-zeros after fixing $k.")
                                     k_new = -2 # Mark as error
                                     break
                                 end
                                 k_new = kk
                             end
                        end
                            
                        if k_new > 0 # Found a valid new singleton
                             if !any(item -> item == (r, k_new), singleton_list)
                                push!(singleton_list, (r, k_new))
                             end
                        elseif k_new == -1 # nnz_row count might be wrong, or row became zero
                             # Recalculate to be sure
                             current_nnz_r = 0
                             for kk_check in findall(potential_cols)
                                 if P.A[r, kk_check] != 0.0
                                     current_nnz_r += 1
                                 end
                             end
                             nnz_row[r] = current_nnz_r # Correct the count
                             if current_nnz_r == 0 && abs(P.b[r]) > tol # Check for infeasibility
                                warn("Problem is infeasible due to row $r becoming 0 = $(P.b[r]) after singleton fix.")
                                return false
                             elseif current_nnz_r == 0
                                 potential_rows[r] = false # Mark zero row for removal
                             end
                        end
                    elseif nnz_row[r] == 0 && abs(P.b[r]) > tol # Check for infeasibility if row becomes zero
                         warn("Problem is infeasible due to row $r becoming 0 = $(P.b[r]) after singleton fix.")
                         return false
                    elseif nnz_row[r] == 0 
                         potential_rows[r] = false # Mark zero row for removal
                    end 
                end
            end # End loop updating other rows
        else
            # If aik is near zero, skip processing this singleton row
            @debug "Skipping singleton row processing for row $i (col $k) due to near-zero aik ($aik)"
        end
    end # End while singleton_list not empty
    # --- End Singleton Processing ---

    # duplicated rows - Needs to consider only *currently* potential rows/cols
    potential_row_indices = findall(potential_rows)
    potential_col_indices = findall(potential_cols)
    for i_idx = 1:(length(potential_row_indices)-1)
        i = potential_row_indices[i_idx]
        if i in ind_dup_r continue end # Skip if already marked as duplicate

        for j_idx = (i_idx+1):length(potential_row_indices)
            j = potential_row_indices[j_idx]
            if j in ind_dup_r continue end # Skip if already marked as duplicate

            match = true
            # Iterate through current potential columns
            for col_idx in potential_col_indices 
                if P.A[i, col_idx] != P.A[j, col_idx]
                    match = false
                    break
                end
            end

            if match # Rows match over potential columns
                if abs(P.b[i] - P.b[j]) < tol # Use tolerance
                    push!(ind_dup_r, j)
                    potential_rows[j] = false # Mark row j as removed
                else
                    warn("This problem is infeasible due to duplicated rows with different RHS.")
                    return false
                end
            end
        end
    end
    # Combine initial zero rows, duplicates, singletons, free singletons, and forcing rows that were marked
    # Note: potential_rows flag is the primary indicator now
    # remove_ind_row = unique([ind0r; ind_dup_r; ind_singleton_r; ind_free_singleton_r; ind_forcing_r]) # Original logic
    # @show remove_ind_row 

    # duplicate columns 
    # Needs modification to consider only potential rows/cols
    potential_row_indices = findall(potential_rows) # Update again
    potential_col_indices = findall(potential_cols) # Update again
    processed_cols = falses(n) # Track columns already part of a duplicate group
    dup_groups = [] # Store groups of duplicate column indices

    for i_idx in 1:length(potential_col_indices)
        i = potential_col_indices[i_idx]
        if processed_cols[i] || !potential_cols[i] continue end # Skip if processed or no longer potential
        
        current_dup_group = [i]
        
        for j_idx in (i_idx+1):length(potential_col_indices)
            j = potential_col_indices[j_idx]
            if processed_cols[j] || !potential_cols[j] continue end # Skip if processed or no longer potential

            # Compare columns P.A[:,i] and P.A[:,j] over potential rows
            match = true
            for row_idx in potential_row_indices
                 if P.A[row_idx, i] != P.A[row_idx, j]
                     match = false
                     break
                 end
            end

            if match
                push!(current_dup_group, j)
            end
        end

        if length(current_dup_group) > 1
             push!(dup_groups, current_dup_group)
             # Mark all in this group as processed to avoid redundant checks
             for col_idx in current_dup_group
                 processed_cols[col_idx] = true
             end
        else
            processed_cols[i] = true # Mark as processed even if no duplicates found
        end
    end
    
    # Process duplicate groups found
    for group in dup_groups
        minv, mini_local = findmin(P.c[group]) # Find best cost in group
        main_col_idx = group[mini_local] # Original index of the main column kept
            
        push!(dup_main_c, main_col_idx) # Store the index of the main column kept

        for col_idx in group
             if col_idx != main_col_idx
                 # This column is redundant
                 push!(ind_dup_c, col_idx)
                 potential_cols[col_idx] = false # Mark for removal
             end
        end
    end
    
    # Combine initial zero cols, duplicates, fixed cols (bounds+singletons), and free singletons
    # Note: potential_cols flag is the primary indicator now
    # remove_ind_col = unique([ind0c; ind_dup_c; ind_fix_c; ind_free_singleton_c]) # Original logic
    # @show remove_ind_col


    # Update flags based on final potential status (redundant if flags used directly)
    # flags_row = potential_rows
    # flags_col = potential_cols

    # Final indices for the presolved problem are those where flags are true
    row_indices = findall(potential_rows)
    col_indices = findall(potential_cols)

    # --- Column Singleton Dual Bound Derivation --- Needs updated row/col indices
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
            
            # Guard against division by near-zero
            if abs(akj) > 1e-10
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
                end
            else
                 @debug "Skipping dual bound derivation for col $j (row $k) due to near-zero akj ($akj)"
            end
        end
    end
    # --- End Column Singleton Dual Bound Derivation ---

    # --- Dominated Column Check (using Dual Bounds) ---
    potential_col_indices_dom = findall(potential_cols) # Get current potential cols
    potential_row_indices_dom = findall(potential_rows) # Get current potential rows

    for j in potential_col_indices_dom
        if !potential_cols[j] continue end # Re-check if fixed 

        cj = P.c[j]
        lj = P.lo[j]
        uj = P.hi[j]

        ej = 0.0 # Min value of sum(aij * yi)
        dj = 0.0 # Max value of sum(aij * yi)
        ej_is_neg_inf = false
        dj_is_pos_inf = false

        for i in potential_row_indices_dom
            aij = P.A[i, j]
            if aij == 0.0 continue end

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
            if isinf(lj) && lj < 0
                @warn("Problem is likely UNBOUNDED (dominated column $j requires lower bound -Inf, z* > 0)")
                return :Unbounded 
            end
            fixed_val = lj
            # Check consistency and fix
            k_fixed_idx = findfirst(==(j), ind_fix_c)
            if k_fixed_idx !== nothing # Already fixed
                if abs(fix_vals[k_fixed_idx] - fixed_val) > tol
                    @warn("Problem infeasible: Dominated column rule for var $j conflicts with previous fixed value")
                    return false 
                end
            else 
                #@debug "Fixing var $j = $fixed_val due to dominated column (z* > 0)"
                push!(ind_fix_c, j)
                push!(fix_vals, fixed_val)
                P.b .-= P.A[:, j] * fixed_val # Update RHS
                potential_cols[j] = false # Mark as removed
            end

        # Case 2: Upper bound on z*j is negative? (cj - ej < -tol)
        elseif !ej_is_neg_inf && cj - ej < -tol
            if isinf(uj) && uj > 0
                @warn("Problem is likely UNBOUNDED (dominated column $j requires upper bound +Inf, z* < 0)")
                return :Unbounded 
            end
            fixed_val = uj
            # Check consistency and fix
            k_fixed_idx = findfirst(==(j), ind_fix_c)
            if k_fixed_idx !== nothing # Already fixed
                if abs(fix_vals[k_fixed_idx] - fixed_val) > tol
                    @warn("Problem infeasible: Dominated column rule for var $j conflicts with previous fixed value")
                    return false 
                end
            else 
                #@debug "Fixing var $j = $fixed_val due to dominated column (z* < 0)"
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
                fixed_val = lj
                k_fixed_idx = findfirst(==(j), ind_fix_c)
                if k_fixed_idx !== nothing # Already fixed
                    if abs(fix_vals[k_fixed_idx] - fixed_val) > tol
                        @warn("Problem infeasible: Weakly dominated column rule (>=0) for var $j conflicts with previous fixed value")
                        return false 
                    end
                else 
                    #@debug "Fixing var $j = $fixed_val due to weakly dominated column (z* >= 0)"
                    push!(ind_fix_c, j)
                    push!(fix_vals, fixed_val)
                    P.b .-= P.A[:, j] * fixed_val # Update RHS
                    potential_cols[j] = false # Mark as removed
                end
            
            # Case 4: Upper bound on z*j is zero? (cj - ej = 0) and finite upper bound
            elseif !ej_is_neg_inf && abs(cj - ej) < tol && !isinf(uj)
                fixed_val = uj
                k_fixed_idx = findfirst(==(j), ind_fix_c)
                if k_fixed_idx !== nothing # Already fixed
                    if abs(fix_vals[k_fixed_idx] - fixed_val) > tol
                        @warn("Problem infeasible: Weakly dominated column rule (<=0) for var $j conflicts with previous fixed value")
                        return false 
                    end
                else 
                    #@debug "Fixing var $j = $fixed_val due to weakly dominated column (z* <= 0)"
                    push!(ind_fix_c, j)
                    push!(fix_vals, fixed_val)
                    P.b .-= P.A[:, j] * fixed_val # Update RHS
                    potential_cols[j] = false # Mark as removed
                end
            end
        end
    end
    #--- End Dominated Column Check ---

    # Recalculate final row/col indices based on potential_flags *after* all checks
    row_indices = findall(potential_rows)
    col_indices = findall(potential_cols)

    # Get final removed index lists for debugging/display
    final_removed_rows = findall(.!potential_rows)
    final_removed_cols = findall(.!potential_cols)
    @debug "Final removed row indices: $(final_removed_rows)"
    @debug "Final removed col indices: $(final_removed_cols)"

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
         elseif m_final > 0 && n_final > 0 # Add check for non-empty before rank
            try
                final_rank = rank(Matrix(final_A))
                if m_final >= n_final && final_rank < n_final
                    @warn "Final matrix appears to be column rank deficient (rank $final_rank < n=$n_final)."
                elseif m_final < n_final && final_rank < m_final
                    @warn "Final matrix appears to be row rank deficient (rank $final_rank < m=$m_final)."
                end
            catch rank_e
                 @warn "Could not calculate rank of final matrix: $rank_e"
            end
         end
    end

    # Return presolved problem and mapping information
    return :Success, IplpProblem(P.c[col_indices], final_A, 
     P.b[row_indices], P.lo[col_indices], P.hi[col_indices]), 
     ind0c, dup_main_c, ind_dup_c, ind_fix_c, fix_vals, 
     dual_lb, dual_ub, obj_offset, free_singleton_subs 
end

function revProb(P, ind0c, dup_main_c, ind_dup_c, ind_fix_c, fix_vals, free_singleton_subs, x1)
    m_orig, n_orig = size(P.A) # Use original problem dimensions P passed to revProb
    x = Array{Float64}(undef, n_orig)
    fill!(x, NaN) # Use NaN to indicate unassigned

    fix_val_map = Dict(zip(ind_fix_c, fix_vals)) 
    # For duplicate columns, the current logic sets them to 0. We need the original mapping if 
    # we wanted to assign them the value of their main column.
    # dup_main_map = Dict(zip(dup_main_c, dup_main_c)) 
    # dup_del_to_main_map = Dict{Int, Int}() # Requires more info from presolve

    # Determine which original columns correspond to x1
    active_indices_presolved =
        let 
            flags_col = trues(n_orig)
            # Combine ALL indices known to be removed or fixed
            # Note: ind_fix_c might contain columns fixed for various reasons
            # Ensure all removal lists (ind0c, ind_dup_c, ind_fix_c, keys(free_singleton_subs)) are accurate
            removed_cols = unique([ind0c; ind_dup_c; ind_fix_c; collect(keys(free_singleton_subs))]) 
            flags_col[removed_cols] .= false
            findall(flags_col)
        end
    
    if length(x1) != length(active_indices_presolved)
         error("Mismatch between solved variables ($(length(x1))) and expected active columns ($(length(active_indices_presolved)))")
    end
    
    # Assign solved values from x1 back to original indices
    for (idx, orig_idx) in enumerate(active_indices_presolved)
        x[orig_idx] = x1[idx]
    end

    # Assign values for fixed, zero, and duplicate columns (if not already assigned)
    for i = 1:n_orig
        if isnan(x[i]) # Only assign if not set by the solver result
            if haskey(fix_val_map, i)
                x[i] = fix_val_map[i]
            elseif i in ind0c
                # Handle zero columns based on cost sign (assuming original P passed)
                if P.c[i] > 0 && !isinf(P.lo[i])
                    x[i] = P.lo[i] 
                elseif P.c[i] < 0 && !isinf(P.hi[i])
                    x[i] = P.hi[i] 
                elseif !isinf(P.lo[i]) # If c=0, prefer finite bound if available
                     x[i] = P.lo[i]
                elseif !isinf(P.hi[i])
                     x[i] = P.hi[i]
                else
                    x[i] = 0.0 # Default if bounds are infinite
                end
            elseif i in ind_dup_c 
                # Assign 0 to removed duplicate columns (consistent with previous logic)
                x[i] = 0.0 
            # Free singletons are handled below
            # elseif haskey(free_singleton_subs, i)
            #     # Handled later
            # else
            #     # Ideally, all columns are either active, fixed, zero, or duplicate
            #     # warn("Variable $i was not assigned during initial phase of revProb.")
            end
        end
    end
    
    # Calculate values for substituted free column singletons
    # Need to iterate because one calculation might enable another
    max_passes = n_orig # Safety break
    current_pass = 0
    made_progress = true
    free_singleton_cols = collect(keys(free_singleton_subs))
    unresolved_free_singletons = Set(free_singleton_cols)
    
    while made_progress && !isempty(unresolved_free_singletons) && current_pass < max_passes
        made_progress = false
        current_pass += 1
        resolved_this_pass = Int[]
        
        for j in unresolved_free_singletons
            (k, akj, subs_dict) = free_singleton_subs[j]
            
            sum_term = 0.0
            dependencies_met = true
            for (i_sub, aki) in subs_dict
                # Dependency i_sub must be an assigned value (not NaN)
                if isnan(x[i_sub]) 
                    dependencies_met = false
                    break # Cannot calculate x[j] yet
                end
                sum_term += aki * x[i_sub]
            end
                
            if dependencies_met
                # Check akj again - should be non-zero if stored, but belt-and-suspenders
                if abs(akj) > 1e-10
                    bk = P.b[k] # Use original b[k]
                    x[j] = (bk - sum_term) / akj
                    push!(resolved_this_pass, j)
                    made_progress = true
                else
                     warn("Cannot resolve free singleton var $j in revProb: stored akj ($akj) is near zero.")
                     # Mark as resolved to prevent infinite loop, but value remains NaN
                     push!(resolved_this_pass, j) 
                end
            end
        end
        # Remove resolved singletons from the set for the next pass
        setdiff!(unresolved_free_singletons, resolved_this_pass)
    end
    
    # Check if any free singletons remain unresolved
    if !isempty(unresolved_free_singletons)
         warn("Could not resolve all free singleton variables in revProb. Unresolved: $(unresolved_free_singletons). Possible circular dependency or error.")
         # Values remain NaN
    end

    # Final check: ensure all original variables are assigned (or explicitly handled)
    unassigned_indices = findall(isnan, x)
    if !isempty(unassigned_indices)
        # It might be acceptable for some vars to be NaN if presolve deemed infeasible/unbounded, 
        # but generally all should be assigned if presolve returned :Success
        error("Not all variables were assigned in revProb. Unassigned indices: $(unassigned_indices)")
    end

    return x
end
