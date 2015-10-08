module RegisterWorkerApertures

using Images, AffineTransforms, Interpolations
using BlockRegistration, RegisterCore, RegisterDeformation, RegisterFit, RegisterPenalty, RegisterOptimize
# Note: RegisterMismatch/RegisterMismatchCuda is selected below
using BlockRegistrationScheduler, RegisterWorkerShell

import RegisterWorkerShell: worker, init!, close!

export monitor, monitor!, worker, workerpid
export Apertures

type Apertures{A<:AbstractArray,T,K,N} <: AbstractWorker
    fixed::A
    knots::NTuple{N,K}
    maxshift::NTuple{N,Int}
    affinepenalty::AffinePenalty{T,N}
    λrange::Tuple{T,T}
    thresh::T
    preprocess
    normalization::Symbol
    correctbias::Bool
    workerpid::Int
    dev::Int
    CUDArt_module
end

function init!(algorithm::Apertures)
    if algorithm.dev >= 0
        @eval using RegisterMismatchCuda
        modules = [algorithm.CUDArt_module]
        CUDArt.init!(modules, [algorithm.dev])
        algorithm.CUDArt_module = modules[1]
        RegisterMismatchCuda.init([algorithm.dev])
    else
        @eval using RegisterMismatch
    end
    nothing
end

function close!(algorithm::Apertures)
    if algorithm.dev >= 0
        CUDArt.close!([algorithm.CUDArt_module], [algorithm.dev])
        RegisterMismatchCuda.close()
    end
    nothing
end

function Apertures{K,N}(fixed, knots::NTuple{N,K}, maxshift, λrange, preprocess=identity; normalization=:pixels, thresh_fac=(0.5)^ndims(fixed), thresh=nothing, correctbias::Bool=true, pid=1, dev=-1)
    gridsize = map(length, knots)
    nimages(fixed) == 1 || error("Register to a single image")
    if thresh == nothing
        thresh = (thresh_fac/prod(gridsize)) * (normalization==:pixels ? length(fixed) : sumabs2(fixed))
    end
    # T = eltype(fixed) <: AbstractFloat ? eltype(fixed) : Float32
    T = Float64   # Ipopt requires Float64
    cumodule = dev == -1 ? nothing : CuModule
    Apertures{typeof(fixed),T,K,N}(fixed, knots, maxshift, AffinePenalty{T,N}(knots, λrange[1]), (T(λrange[1]),T(λrange[end])), T(thresh), preprocess, normalization, correctbias, pid, dev, cumodule)
end

function worker(algorithm::Apertures, img, tindex, mon)
    moving0 = timedim(img) == 0 ? img : img["t", tindex]
    moving = algorithm.preprocess(moving0)
    use_cuda = algorithm.dev >= 0
    if use_cuda
        device(algorithm.dev)
    end
    mms = mismatch_apertures(algorithm.fixed, moving, map(length, algorithm.knots), algorithm.maxshift; normalization=algorithm.normalization)
    # displaymismatch(mms, thresh=10)
    if algorithm.correctbias
        correctbias!(mms)
    end
    E0 = zeros(size(mms))
    cs = Array(Any, size(mms))
    Qs = Array(Any, size(mms))
    thresh = algorithm.thresh
    for i = 1:length(mms)
        E0[i], cs[i], Qs[i] = qfit(mms[i], thresh)
    end
    mmis = interpolate_mm!(mms)
    ϕ, mismatch, λ, dp, quality = RegisterOptimize.auto_λ(cs, Qs, algorithm.knots, algorithm.affinepenalty, mmis, algorithm.λrange...)
    monitor!(mon, :mismatch, mismatch)
    monitor!(mon, :λ, λ)
    monitor!(mon, :datapenalty, dp)
    monitor!(mon, :sigmoid_quality, quality)
    monitor!(mon, :u, ϕ.u)
    if haskey(mon, :warped)
        warped = warp(moving, ϕ)
        monitor!(mon, :warped, warped)
    end
    if haskey(mon, :warped0)
        warped = warp(moving0, ϕ)
        monitor!(mon, :warped, warped)
    end
    nothing
end

end # module
