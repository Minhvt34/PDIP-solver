using LinearAlgebra # For I
using SparseArrays

function tostandard(P)
    # --- Check Inputs ---
    if any(!isfinite, P.A.nzval); @error "tostandard: Input A contains non-finite values!"; end
    if any(!isfinite, P.b); @error "tostandard: Input b contains non-finite values!"; end
    # Check lo for NaN specifically as replace only handles Inf
    if any(isnan, P.lo); @error "tostandard: Input lo contains NaN values!"; end 
    if any(isinf, P.lo); @warn "tostandard: Input lo contains Inf values (will be replaced)."; end # Warning instead of error
    if any(isnan, P.hi); @error "tostandard: Input hi contains NaN values!"; end
    if any(isinf, P.hi); @warn "tostandard: Input hi contains Inf values (will be replaced)."; end # Warning instead of error
    # --- End Check ---

    # Convert to standard form
    # Four cases to handle
    # Free x
    # - substitute x = x+ - x-
    # A(x+ - x-) = b ->
    # x bounded finitely below (lo <= x < inf)
    # - substitute x = y + lo
    # x bounded finitely above (-inf <= x < hi) + one of above cases
    # - introduce slack: x + s = hi (combine with above case if needed)

    INF = 1e305
    m,n = size(P.A)

    # Indices of modified variables
    free = []
    bounded_above = []
    bounded_below = []
    bounded = []

    for i = 1:n
        if P.lo[i] < -INF && P.hi[i] > INF
            # Free var; x = x+ - x-
            push!(free, i)
        elseif P.lo[i] < -INF
            # Bounded above, unbounded below
            # Add slack variable: [A I][x s]^T = b
            push!(bounded_above, i)
        elseif P.hi[i] > INF
            # UnBounded above, bounded below
            # Slack variable negative: 
            push!(bounded_below, i)
        else
            # Bounded above and below
            push!(bounded, i)
        end
    end

    cs = [
        # x+, x-
         P.c[free];
        -P.c[free];
        # bounded below: apply an offset
         P.c[bounded_below];

        # bounded above: use -variable, apply offset
        -P.c[bounded_above];

        # Bounded on both sides; apply offset and add slack var
        P.c[bounded];
        zeros(size(bounded))
    ]

    # Get # slacks required, # non-fully bounded vars
    ns = size(bounded)[1]
    nub = size(free)[1] * 2 + size(bounded_above)[1] + size(bounded_below)[1]

    As = [
        # Constraint vars per row:
        # X+ X- (X >= lo) (X <= hi -> -X >= -hi) (lo <= X <= hi -> 0 <= X <= hi - lo) (S slacks)
        P.A[:,free] -P.A[:,free] P.A[:,bounded_below] -P.A[:,bounded_above] P.A[:,bounded] spzeros(m, ns);

        # Constraints on slacks: Ax + Is = hi - lo
        spzeros(ns, nub) Matrix{Float64}(I, ns, ns) Matrix{Float64}(I, ns, ns)
    ]

    # Ax = b
    # y = x + lo
    # z = hi - x
    # w = x + lo
    # Ax+ - Ax- + Ay + Az + Aw = ?
    # Ax+ - Ax- + A(x + lo) + A(hi - x) + A(x + lo) = (Ax+ - Ax-) + (Ax) + Alo + Ahi - (Ax) + (Ax) + Alo = b - Alo - Ahi - Alo

    # --- Calculate terms for bs ---
    # Avoid multiplying by Inf bounds; replace Inf/-Inf in bound slices with 0 before multiplication.
    term_b_below = zeros(m)
    if !isempty(bounded_below)
        A_sub_below = P.A[:,bounded_below]
        lo_sub_below = P.lo[bounded_below]
        # Replace +/- Inf with 0.0 to avoid Inf*0 or Inf*finite -> NaN/Inf issues
        # Also replace near-infinite finite values to prevent overflow
        NEAR_INF_THRESHOLD = 1e300 # Define a threshold for near-infinite
        lo_sub_below_finite = map(x -> (isinf(x) || abs(x) > NEAR_INF_THRESHOLD) ? 0.0 : x, lo_sub_below)
        
        # -- Add more checks before multiplication --
        if any(!isfinite, A_sub_below.nzval); @warn "tostandard: A_sub_below still contains non-finite nzval!"; end
        if any(!isfinite, lo_sub_below_finite); @error "tostandard: lo_sub_below_finite is non-finite AFTER replace! Contains NaN?" lo_slice=lo_sub_below finite_lo=lo_sub_below_finite; end
        # -- End Check --

        # -- Check magnitudes before multiplication --
        max_abs_A = isempty(A_sub_below.nzval) ? 0.0 : maximum(abs.(A_sub_below.nzval))
        max_abs_lo = isempty(lo_sub_below_finite) ? 0.0 : maximum(abs.(lo_sub_below_finite))
        @info "tostandard: Multiplying A_sub_below (max abs: $(max_abs_A)) * lo_sub_below_finite (max abs: $(max_abs_lo))"
        # -- End Check --

        term_b_below = A_sub_below * lo_sub_below_finite
    end
    
    term_b_above = zeros(m)
    if !isempty(bounded_above)
        A_sub_above = P.A[:,bounded_above]
        hi_sub_above = P.hi[bounded_above]
        # Replace +/- Inf and near-infinite finite values with 0.0
        NEAR_INF_THRESHOLD = 1e300
        hi_sub_above_finite = map(x -> (isinf(x) || abs(x) > NEAR_INF_THRESHOLD) ? 0.0 : x, hi_sub_above)
        term_b_above = A_sub_above * hi_sub_above_finite
    end

    term_bounded = zeros(m)
    if !isempty(bounded)
        A_sub_bounded = P.A[:,bounded]
        lo_sub_bounded = P.lo[bounded]
        # Replace +/- Inf and near-infinite finite values with 0.0
        NEAR_INF_THRESHOLD = 1e300
        lo_sub_bounded_finite = map(x -> (isinf(x) || abs(x) > NEAR_INF_THRESHOLD) ? 0.0 : x, lo_sub_bounded)
        term_bounded = A_sub_bounded * lo_sub_bounded_finite
    end
    
    term_slacks = isempty(bounded) ? Float64[] : P.hi[bounded]-P.lo[bounded]
    # Replace Inf/-Inf/NaN in term_slacks
    # Replace near-infinite finite values resulting from subtraction?
    NEAR_INF_THRESHOLD = 1e300
    term_slacks_finite = map(x -> (isnan(x) || isinf(x) || abs(x) > NEAR_INF_THRESHOLD) ? 0.0 : x, term_slacks)

    # --- Check intermediate terms (using original terms before finite replacement for debugging) ---
    if any(!isfinite, P.A[:,bounded_below]*P.lo[bounded_below]); @warn "tostandard: Original term_b_below calculation still has non-finite values!"; end
    if any(!isfinite, term_b_above); @warn "tostandard: term_b_above contains non-finite values!"; end
    if any(!isfinite, term_bounded); @warn "tostandard: term_bounded contains non-finite values!"; end
    if any(!isfinite, term_slacks); @warn "tostandard: term_slacks contains non-finite values!"; end
    # --- End Check ---
    
    # --- Check before subtraction ---
    if any(!isfinite, P.b); @warn "tostandard: P.b has non-finite before subtraction!"; end
    if any(!isfinite, term_b_below); @warn "tostandard: term_b_below has non-finite before subtraction!"; end
    if any(!isfinite, term_b_above); @warn "tostandard: term_b_above has non-finite before subtraction!"; end
    if any(!isfinite, term_bounded); @warn "tostandard: term_bounded has non-finite before subtraction!"; end
    # --- End Check ---
    
    bs_part1 = P.b - term_b_below - term_b_above - term_bounded
    
    # --- Check first part of bs ---
    if any(!isfinite, bs_part1); 
        @warn "tostandard: bs_part1 (after subtractions) contains non-finite values!" Pb=P.b tbb=term_b_below tba=term_b_above tb=term_bounded result=bs_part1
    end
    # --- End Check ---

    bs = [bs_part1; term_slacks_finite]

    # --- Check final bs ---
    if any(!isfinite, bs); @error "tostandard: Final bs contains non-finite values!"; end
    # --- End Check ---

    return As, bs, cs, free, bounded_below, bounded_above, bounded
end

function fromstandard(P, xs, free, bounded_below, bounded_above, bounded)
    n1 = length(free)
    n2 = length(bounded_below)
    n3 = length(bounded_above)
    n4 = length(bounded)

    # Reconstruct dim of original var
    n = n1 + n2 + n3 + n4
    x = zeros(n)

    # --- Check input xs magnitude ---
    if length(xs) > 0
        max_abs_xs = maximum(abs.(xs))
        @info "fromstandard: Input xs max abs = $(max_abs_xs)"
    else
        @info "fromstandard: Input xs is empty"
    end
    # --- End Check ---
    
    # Calculate indices (optional, for clarity/debugging)
    idx_free_plus = 1:n1
    idx_free_minus = (n1+1):(2*n1)
    idx_bounded_below = (2*n1 + 1):(2*n1 + n2)
    idx_bounded_above = (2*n1 + n2 + 1):(2*n1 + n2 + n3)
    idx_bounded = (2*n1 + n2 + n3 + 1):(2*n1 + n2 + n3 + n4)

    if n1 > 0
        # Check operands
        max_abs_xs_free_plus = isempty(idx_free_plus) ? 0.0 : maximum(abs.(xs[idx_free_plus]))
        max_abs_xs_free_minus = isempty(idx_free_minus) ? 0.0 : maximum(abs.(xs[idx_free_minus]))
        @info "fromstandard/free: xs+ max abs=$(max_abs_xs_free_plus), xs- max abs=$(max_abs_xs_free_minus)"
        x[free] = xs[idx_free_plus] .- xs[idx_free_minus]
    end
    if n2 > 0
        # Check operands
        max_abs_xs_bb = isempty(idx_bounded_below) ? 0.0 : maximum(abs.(xs[idx_bounded_below]))
        P_lo_bb_slice = P.lo[bounded_below]
        max_abs_lo_bb = isempty(bounded_below) ? 0.0 : maximum(abs.(P_lo_bb_slice))
        @info "fromstandard/bounded_below: xs max abs=$(max_abs_xs_bb), P.lo max abs=$(max_abs_lo_bb)"
        # Replace near-infinite bounds with 0 before adding
        NEAR_INF_THRESHOLD = 1e300
        P_lo_bb_slice_finite = map(x -> abs(x) > NEAR_INF_THRESHOLD ? 0.0 : x, P_lo_bb_slice)
        x[bounded_below] = xs[idx_bounded_below] .+ P_lo_bb_slice_finite
    end
    if n3 > 0
        # Check operands
        max_abs_xs_ba = isempty(idx_bounded_above) ? 0.0 : maximum(abs.(xs[idx_bounded_above]))
        P_hi_ba_slice = P.hi[bounded_above]
        max_abs_hi_ba = isempty(bounded_above) ? 0.0 : maximum(abs.(P_hi_ba_slice))
        @info "fromstandard/bounded_above: P.hi max abs=$(max_abs_hi_ba), xs max abs=$(max_abs_xs_ba)"
        # Replace near-infinite bounds with 0 before subtracting
        NEAR_INF_THRESHOLD = 1e300
        P_hi_ba_slice_finite = map(x -> abs(x) > NEAR_INF_THRESHOLD ? 0.0 : x, P_hi_ba_slice)
        x[bounded_above] = P_hi_ba_slice_finite .- xs[idx_bounded_above]
    end
    if n4 > 0
        # Check operands
        max_abs_xs_b = isempty(idx_bounded) ? 0.0 : maximum(abs.(xs[idx_bounded]))
        P_lo_b_slice = P.lo[bounded]
        max_abs_lo_b = isempty(bounded) ? 0.0 : maximum(abs.(P_lo_b_slice))
        @info "fromstandard/bounded: xs max abs=$(max_abs_xs_b), P.lo max abs=$(max_abs_lo_b)"
        # Replace near-infinite bounds with 0 before adding
        NEAR_INF_THRESHOLD = 1e300
        P_lo_b_slice_finite = map(x -> abs(x) > NEAR_INF_THRESHOLD ? 0.0 : x, P_lo_b_slice)
        x[bounded] = xs[idx_bounded] .+ P_lo_b_slice_finite
    end

    # --- Check output x magnitude ---
    max_abs_x = maximum(abs.(x))
    @info "fromstandard: Output x max abs = $(max_abs_x)"
    # --- End Check ---

    return x
end