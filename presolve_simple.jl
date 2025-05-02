# Presolving and un-presolving functions for IPLP.
# Solves problems with non-full-rank matrices,
# decides variables ahead of time etc.

# Returns: (A, b, c, feasible)
function simple_presolve(A, b, c, hi, lo)
    m,n = size(A)

    # Check for zero and duplicate columns
    column_set = Dict()
    to_remove = []
    xpre = []
    for j = 1:n
        # Remove zero column
        # Determine what the value of x should be
        if @views all(A[:, j] .== 0)
            if c[j] < 0
                # Set this x to upper bound
                if hi[j] > 1e308
                    # Unbounded below
                    @warn "Presolve step detected problem objective is unbounded."
                    return A, b, c, [], [], [], false
                end
                push!(xpre, hi[j])
            elseif c[j] > 0
                # Set this x to lower bound
                if lo[j] < -1e308
                    # Unbounded below
                    @warn "Presolve step detected problem objective is unbounded."
                    return A, b, c, [], [], [], false
                end
                push!(xpre, lo[j])
            else
                # x value is completely irrelevant, so we'll set it to 0
                push!(xpre, 0)
            end
            push!(to_remove, j)
            continue
        end

        # TODO: Handle duplicate columns
    end
    cols_removed = to_remove
    remaining_cols = setdiff(1:n, to_remove)
    A = A[:, remaining_cols]
    c = c[remaining_cols]
    m,n = size(A)

    # Check for zero and duplicate rows (constraints can be removed)
    # Scale corresponding constraint to norm(A[i, :]) == 1
    to_remove = []
    row_set = Set()
    for i = 1:m
        # Remove zero row
        if @views all(A[i, :] .== 0)
            push!(to_remove, i)
            continue
        end

        # Rescale row and corresponding b using infinity norm
        scale = norm(A[i, :], Inf)
        A[i, :] = A[i, :] / scale
        b[i] = b[i] / scale

        # Check for duplicate row
        if A[i, :] in row_set
            push!(to_remove, i)
        else
            push!(row_set, A[i, :])
        end
    end
    remaining_rows = setdiff(1:m, to_remove)
    A = A[remaining_rows, :]
    b = b[remaining_rows]
    m,n = size(A)

    return A, b, c, remaining_cols, cols_removed, xpre, true
end

function simple_unpresolve(n, xs, remaining_cols, cols_removed, xpre)
    x = zeros(n, 1)
    x[remaining_cols] = xs
    x[cols_removed] = xpre
    return x
end