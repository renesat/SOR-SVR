module LibSORSVR

using KernelFunctions: SqExponentialKernel
using LinearAlgebra: dot, LowerTriangular, Diagonal;
using Statistics: mean
import MLJModelInterface
const MMI = MLJModelInterface
# using IterativeSolvers: sor

export SORSVR, sor

MMI.@mlj_model mutable struct SORSVR <: MMI.Probabilistic
    C::Number = 100.0::(_ > 0)
    kernel = dot
    ω::Number = 1.0::(_ > 0)
    ε::Number = 0.001::(_ > 0)
    maxiter::Int = 1000.0::(_ > 0)
end

function sor(A::AbstractMatrix, b; C::Real = 100.0, ω::Real = 1.0, ϵ::Real = 10^-3, maxiter::Int = 500)
    local N = size(A, 1)
    local T = typeof(one(eltype(b))/one(eltype(A)))
    local α = similar(b, T, size(A, 2))
    fill!(α, zero(T))
    local αprev = similar(α)
    fill!(αprev, Inf)
    local tmp = similar(α)

    local iteration = 0
    while (sum(abs.(α - αprev)) > ϵ) && (iteration < maxiter)
        iteration += 1
        αprev = deepcopy(α)
        for j in 1:N
            @simd for i in 1:j-1
                @inbounds tmp[i] -= A[i, j] * α[j]
            end
            @inbounds tmp[j] = b[j]
        end
        for j in 1:N
            @inbounds α[j] += ω * (tmp[j] / A[j,j] - α[j])
            @inbounds α[j] = min(C, max(α[j], 0))
            @simd for i in j+1:N
                @inbounds tmp[i] -= A[i, j] * α[j]
            end
        end
    end
    α
end

function MMI.fit(model::SORSVR, verbosity::Integer, inX, iny; init = nothing)
    local X = MMI.matrix(inX)
    local y = iny |> collect
    local N = size(X, 1)

    local K = hcat([
        [model.kernel(X[a, :], X[b, :]) for a in 1:N]
        for b in 1:N
    ]...)
    local E = ones(N, N)

    local A = vcat(
        hcat(K+E, -K-E),
        hcat(-K-E, K+E)
    )
    local b = vcat(
        y .- model.ε,
        -y .- model.ε
    )

    α = sor(
        A, b, C = model.C,
        ω = model.ω,
        maxiter = model.maxiter
    )


    isSVPoints = α[1:N] .!= α[N+1:end]
    αDiffSV = (
        α[1:N] .- α[N+1:end]
    )[isSVPoints]
    sv = X[isSVPoints, :]
    b = sum(αDiffSV)

    local fitresult = (
        α = α[1:N],
        αStar = α[N+1:end],
        αDiff = αDiffSV,
        sv = sv,
        svLabel = isSVPoints,
        b = b
    )
    local cache = nothing
    local report = nothing
    fitresult, cache, report
end

function MMI.predict(model::SORSVR, fitresult, Xnew)
    local X = MMI.matrix(Xnew)
    local N = size(X, 1)

    local αDiff = fitresult.αDiff
    local b = fitresult.b
    local sv = fitresult.sv

    local yhat
    if length(αDiff) < 1
        yhat = [ NaN for x in eachrow(X)]
    else
        yhat = [
            sum(αDiff .* [model.kernel(x, xsv) for xsv in eachrow(sv)]) + b
            for x in eachrow(X)
        ]
    end

    yhat
end

end
