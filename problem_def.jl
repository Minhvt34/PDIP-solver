using MatrixDepot, SparseArrays

mutable struct IplpSolution
  x::Vector{Float64} # the solution vector
  flag::Bool         # a true/false flag indicating convergence or not
  cs::Vector{Float64} # the objective vector in standard form
  As::SparseMatrixCSC{Float64} # the constraint matrix in standard form
  bs::Vector{Float64} # the right hand side (b) in standard form
  xs::Vector{Float64} # the solution in standard form
  lam::Vector{Float64} # the solution lambda in standard form
  s::Vector{Float64} # the solution s in standard form
end

mutable struct IplpProblem
  c::Vector{Float64}
  A::SparseMatrixCSC{Float64}
  b::Vector{Float64}
  lo::Vector{Float64}
  hi::Vector{Float64}
end

function convert_matrixdepot(P::MatrixDepot.MatrixDescriptor)
  # key_base = sort(collect(keys(mmmeta)))[1]
  return IplpProblem(
    vec(P.c), P.A, vec(P.b), vec(P.lo), vec(P.hi))
end