module RegisterWorkerApertures

using Images, AffineTransforms, Interpolations
using BlockRegistration, RegisterCore, RegisterDeformation, RegisterFit, RegisterPenalty, RegisterOptimize
# Note: RegisterMismatch/RegisterMismatchCuda is selected below
using BlockRegistrationScheduler, RegisterWorkerShell

import RegisterWorkerShell: worker, init!, close!

export Apertures, PreprocessSNF, monitor, monitor!, worker, workerpid

type Apertures{A<:AbstractArray,T,K,N} <: AbstractWorker
    fixed::A
    knots::NTuple{N,K}
    maxshift::NTuple{N,Int}
    affinepenalty::AffinePenalty{T,N}
    λrange::Union{T,Tuple{T,T}}
    thresh::T
    preprocess  # likely of type PreprocessSNF, but could be a function
    normalization::Symbol
    correctbias::Bool
    workerpid::Int
    dev::Int
    cuda_objects::Dict{Symbol,Any}
end

function init!(algorithm::Apertures)
    if algorithm.dev >= 0
        eval(:(using CUDArt, RegisterMismatchCuda))
        cuda_init(algorithm)
    else
        eval(:(using RegisterMismatch))
    end
    algorithm
end

function cuda_init(algorithm)
    CUDArt.init(algorithm.dev)
    RegisterMismatchCuda.init([algorithm.dev])
    # Allocate the CUDA objects once at the beginning. Even
    # though all temporary arrays appear to be freed, repeated
    # allocation results in "out of memory" errors. (CUDA bug?)
    device(algorithm.dev)
    d_fixed  = CudaPitchedArray(convert(Array{Float32}, sdata(data(algorithm.fixed))))
    algorithm.cuda_objects[:d_fixed] = d_fixed
    algorithm.cuda_objects[:d_moving] = similar(d_fixed)
    gridsize = map(length, algorithm.knots)
    aperture_width = default_aperture_width(algorithm.fixed, gridsize)
    algorithm.cuda_objects[:cms] = CMStorage(Float32, aperture_width, algorithm.maxshift)
end

function close!(algorithm::Apertures)
    if algorithm.dev >= 0
        for (k,v) in algorithm.cuda_objects
            free(v)
        end
        RegisterMismatchCuda.close()
        CUDArt.close(algorithm.dev)
    end
    nothing
end

function Apertures{K,N}(fixed, knots::NTuple{N,K}, maxshift, λrange, preprocess=identity; normalization=:pixels, thresh_fac=(0.5)^ndims(fixed), thresh=nothing, correctbias::Bool=true, pid=1, dev=-1)
    gridsize = map(length, knots)
    nimages(fixed) == 1 || error("Register to a single image")
    isa(λrange, Number) || isa(λrange, NTuple{2}) || error("λrange must be a number or 2-tuple")
    if thresh == nothing
        thresh = (thresh_fac/prod(gridsize)) * (normalization==:pixels ? length(fixed) : sumabs2(fixed))
    end
    # T = eltype(fixed) <: AbstractFloat ? eltype(fixed) : Float32
    T = Float64   # Ipopt requires Float64
    λrange = isa(λrange, Number) ? T(λrange) : (T(first(λrange)), T(last(λrange)))
    Apertures{typeof(fixed),T,K,N}(fixed, knots, maxshift, AffinePenalty{T,N}(knots, first(λrange)), λrange, T(thresh), preprocess, normalization, correctbias, pid, dev, Dict{Symbol,Any}())
end

function worker(algorithm::Apertures, img, tindex, mon)
    moving0 = timedim(img) == 0 ? img : img["t", tindex]
    moving = algorithm.preprocess(moving0)
    gridsize = map(length, algorithm.knots)
    use_cuda = algorithm.dev >= 0
    if use_cuda
        device(algorithm.dev)
        d_fixed  = algorithm.cuda_objects[:d_fixed]
        d_moving = algorithm.cuda_objects[:d_moving]
        cms      = algorithm.cuda_objects[:cms]
        copy!(d_moving, moving)
        cs = coords_spatial(img)
        aperture_centers = aperture_grid(size(img)[cs], gridsize)
        mms = allocate_mmarrays(eltype(cms), gridsize, algorithm.maxshift)
        mismatch_apertures!(mms, d_fixed, d_moving, aperture_centers, cms; normalization=algorithm.normalization)
    else
        mms = mismatch_apertures(algorithm.fixed, moving, gridsize, algorithm.maxshift; normalization=algorithm.normalization)
    end
    # displaymismatch(mms, thresh=10)
    if algorithm.correctbias
        correctbias!(mms)
    end
    E0 = zeros(size(mms))
    cs = Array(Any, size(mms))
    Qs = Array(Any, size(mms))
    thresh = algorithm.thresh
    for i = 1:length(mms)
        E0[i], cs[i], Qs[i] = qfit(mms[i], thresh; opt=false)
    end
    mmis = interpolate_mm!(mms)
    λrange = algorithm.λrange
    if isa(λrange, Number)
        ϕ, mismatch = RegisterOptimize.fixed_λ(cs, Qs, algorithm.knots, algorithm.affinepenalty, mmis)
    else
        ϕ, mismatch, λ, dp, quality = RegisterOptimize.auto_λ(cs, Qs, algorithm.knots, algorithm.affinepenalty, mmis, λrange...)
        monitor!(mon, :λ, λ)
        monitor!(mon, :datapenalty, dp)
        monitor!(mon, :sigmoid_quality, quality)
    end
    monitor!(mon, :mismatch, mismatch)
    monitor!(mon, :u, ϕ.u)
    if haskey(mon, :warped)
        warped = warp(moving, ϕ)
        monitor!(mon, :warped, warped)
    end
    if haskey(mon, :warped0)
        warped = warp(moving0, ϕ)
        monitor!(mon, :warped0, warped)
    end
    mon
end

"""
`pp = PreprocessSNF(bias, sigmalp, sigmahp)` constructs an object that
can be used to pre-process an image as `pp(img)`. The "SNF" part of
the name means "shot-noise filtered," meaning that this preprocessor
is specifically designed for situations in which you are dominated by
shot noise (i.e., from photon-counting statistics).

The processing is of the form
```
    imgout = bandpass(√max(0,img-bias))
```
i.e., the image is bias-subtracted, square-root transformed (to turn
shot noise into constant variance), and then band-pass filtered using
Gaussian filters of width `sigmalp` (for the low-pass) and `sigmahp`
(for the high-pass).
"""
type PreprocessSNF{T}  # Shot-noise filtered
    bias::T
    sigmalp::Vector{Float64}
    sigmahp::Vector{Float64}
end
PreprocessSNF{T}(bias::T, sigmalp, sigmahp) = PreprocessSNF{T}(bias, Float64[sigmalp...], Float64[sigmahp...])

function Base.call(pp::PreprocessSNF, A::AbstractArray)
    Af = sqrt(max(0, A-pp.bias))
    imfilter_gaussian(highpass(Af, pp.sigmahp), pp.sigmalp)
end

end # module
