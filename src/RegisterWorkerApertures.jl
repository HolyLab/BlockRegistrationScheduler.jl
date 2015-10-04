module RegisterWorkerApertures

using Images, AffineTransforms, Interpolations
using BlockRegistration, RegisterCore, RegisterDeformation, RegisterFit, RegisterPenalty, RegisterOptimize, RegisterWorkerShell
# Note: RegisterMismatch/RegisterMismatchCuda is selected below

import RegisterWorkerShell: worker, init!, close!

export monitor, monitor!, worker, workerpid
export Apertures

type Apertures{A<:AbstractArray,T,K,N} <: AbstractWorker
    fixed::A
    knots::NTuple{N,K}
    maxshift::NTuple{N,Int}
    affinepenalty::AffinePenalty{T}
    thresh::T
    preprocess
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

function Apertures{K,N}(fixed, knots::NTuple{N,K}, maxshift, λ, preprocess=identity; thresh_fac=(0.5)^ndims(fixed), thresh=nothing, pid=1, dev=-1)
    nimages(fixed) == 1 || error("Register to a single image")
    cthresh = thresh == nothing ? thresh_fac*sumabs2(fixed) : thresh
    T = eltype(fixed) <: AbstractFloat ? eltype(fixed) : Float64
    cumodule = dev == -1 ? nothing : CuModule
    Apertures{typeof(fixed),T,K,N}(fixed, knots, maxshift, AffinePenalty(knots, λ), convert(T, cthresh), preprocess, pid, dev, cumodule)
end

function worker(algorithm::Apertures, img, tindex, mon)
    moving0 = timedim(img) == 0 ? img : img["t", tindex]
    moving = algorithm.preprocess(moving0)
    use_cuda = algorithm.dev >= 0
    if use_cuda
        device(algorithm.dev)
    end
    mms = mismatch_apertures(algorithm.fixed, moving, map(length, algorithm.knots), algorithm.maxshift)
    correctbias!(mms)
    E0 = zeros(size(mms))
    cs = Array(Any, size(mms))
    Qs = Array(Any, size(mms))
    for i = 1:length(mms)
        E0[i], cs[i], Qs[i] = qfit(mms[i], 5000)
    end
    u0 = initial_deformation(algorithm.affinepenalty, cs, Qs)
    if haskey(mon, :u0)
        monitor!(mon, :u0, u0)
    end
    ϕ = GridDeformation(u0, (algorithm.knots...))
    mmis = interpolate_mm!(mms)
    ϕ, mismatch = optimize!(ϕ, identity, algorithm.affinepenalty, mmis)
    monitor!(mon, :u, ϕ.u)
    if haskey(mon, :warped0)
        monitor!(mon, :warped0, warp(moving0, ϕ))
    end
    if haskey(mon, :warped)
        monitor!(mon, :warped, warp(moving, ϕ))
    end
    nothing
end

end # module
