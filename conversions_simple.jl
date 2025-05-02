function toStandardSimple(P)
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

    bs = [
        P.b - P.A[:,bounded_below]*P.lo[bounded_below,1] - P.A[:,bounded_above]*P.hi[bounded_above,1] - P.A[:,bounded]*P.lo[bounded,1];
        # Constraints on slacks: Ax + Is = hi - lo
        P.hi[bounded,1]-P.lo[bounded,1]
    ]

    return As, bs, cs, free, bounded_below, bounded_above, bounded
end

function fromStandardSimple(P, xs, free, bounded_below, bounded_above, bounded)
    n1 = length(free)
    n2 = length(bounded_below)
    n3 = length(bounded_above)
    n4 = length(bounded)

    # Reconstruct dim of original var
    n = n1 + n2 + n3 + n4
    x = zeros(n)

    x[free] = xs[1:n1] - xs[1+n1:2*n1]
    x[bounded_below] = xs[1+2*n1 : n2+2*n1] + P.lo[bounded_below]
    x[bounded_above] = P.hi[bounded_above] - xs[1+2*n1+n2 : n3+2*n1+n2]
    x[bounded] = xs[1+2*n1+n2+n3 : n4+2*n1+n2+n3] + P.lo[bounded]

    return x
end