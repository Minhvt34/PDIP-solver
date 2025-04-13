# Starting point algorithm, from pg. 410 of Nocedal & Wright.
# Related to pg. 224 of Wright.
# Starting point must satisfy KKT conditions:
# A^T\lambda + s = c
# Ax = b
# x >= 0, s >= 0
# For numerical stability, x and s small is good!
# So:
# Solve problem min ||x||_2^2 s.t. Ax = b
# Solve problem min ||s||_2^2 s.t. A^T\lambda + s = c
# (14.40 Nocedal & Wright):
# x = A^T(AA^T)^-1b
# lambda = (AA^T)-1Ac
# s = c - A^Tlambda

function get_starting_point(A, b, c)
    A = A
    AA = A * A'

    x = A' * ((AA) \ b)
    lambda = (AA) \ (A * c)
    s = c - A' * lambda

    # Adjustment step to ensure x, s have no nonpositive attrs
    dx = max(-3/2 * minimum(x), 0)
    ds = max(-3/2 * minimum(s), 0)
    xhat = x .+ dx
    shat = s .+ ds
    xts = 1/2 * dot(xhat, shat)
    dhatx = xts / (sum(shat))
    dhats = xts / (sum(xhat))

    return xhat .+ dhatx, lambda, shat .+ dhats
end