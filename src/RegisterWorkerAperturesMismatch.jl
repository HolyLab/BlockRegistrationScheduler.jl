__precompile__()

module RegisterWorkerAperturesMismatch

using Images, AffineTransforms, Interpolations, FixedSizeArrays
using RegisterCore, RegisterDeformation, RegisterFit, RegisterPenalty, RegisterOptimize
# Note: RegisterMismatch/RegisterMismatchCuda is selected below
using RegisterWorkerShell, RegisterDriver

import RegisterWorkerShell: worker, init!, close!

export AperturesMismatch, monitor, monitor!, worker, workerpid

type AperturesMismatch{A<:Union{AbstractArray, Tuple{AbstractArray,AbstractArray}},T,K,N} <: AbstractWorker
    fixed::A #the tuple case is useful when odd and even stacks have different fixed images
    knots::NTuple{N,K}
    maxshift::NTuple{N,Int}
    thresh::T
    preprocess  # likely of type PreprocessSNF, but could be a function
    normalization::Symbol
    correctbias::Bool
    Es
    cs
    Qs
    mmis
    workerpid::Int
    dev::Int
    cuda_objects::Dict{Symbol,Any}
end

function init!(algorithm::AperturesMismatch)
    if algorithm.dev >= 0
        eval(:(using CUDArt, RegisterMismatchCuda))
        cuda_init!(algorithm)
    else
        eval(:(using RegisterMismatch))
    end
    nothing
end

function cuda_init!(algorithm)
    CUDArt.init(algorithm.dev)
    RegisterMismatchCuda.init([algorithm.dev])
    # Allocate the CUDA objects once at the beginning: even
    # though all temporary arrays appear to be freed, repeated
    # allocation results in "out of memory" errors. (CUDA bug?)
    device(algorithm.dev)
    gridsize = map(length, algorithm.knots)
    if isa(algorithm.fixed, Tuple)
	d_fixed = (CudaPitchedArray(convert(Array{Float32}, sdata(data(algorithm.fixed[1])))), CudaPitchedArray(convert(Array{Float32}, sdata(data(algorithm.fixed[2])))))
        aperture_width = default_aperture_width(algorithm.fixed[1], gridsize)
    else
        d_fixed  = (CudaPitchedArray(convert(Array{Float32}, sdata(data(algorithm.fixed)))), CudaPitchedArray(convert(Array{Float32}, sdata(data(algorithm.fixed)))))
        aperture_width = default_aperture_width(algorithm.fixed, gridsize)
    end
    algorithm.cuda_objects[:d_fixed] = d_fixed
    algorithm.cuda_objects[:d_moving] = similar(d_fixed[1])

    algorithm.cuda_objects[:cms] = CMStorage(Float32, aperture_width, algorithm.maxshift)
    nothing
end

function close!(algorithm::AperturesMismatch)
    if algorithm.dev >= 0
        for (k,v) in algorithm.cuda_objects
	    if k == :d_fixed
#	        if isa(algorithm.fixed, Tuple)
		    free(v[1]); free(v[2])
 #           	else free(v)
#		end
	    else
		free(v)
	    end
        end
        RegisterMismatchCuda.close()
        CUDArt.close(algorithm.dev)
    end
    nothing
end

"""

`alg = AperturesMismatch(fixed, knots, maxshift, [preprocess=identity];
kwargs...)` creates a worker-object for performing "apertured"
(blocked) registration.  `fixed` is the reference image, `knots`
specifies the grid of apertures, `maxshift` represents the largest
shift (in pixels) that will be evaluated, and `preprocess` allows you
to apply a transformation (e.g., filtering) to the `moving` images
before registration; `fixed` should already have any such
transformations applied.

Registration is performed by calling `driver`.  You should monitor
`Es`, `cs`, `Qs`, and `mmis`.

## Example

Suppose your images are somewhat noisy, in which case a bit of
smoothing might help considerably.  Here we'll illustrate the use of a
pre-processing function, but see also `PreprocessSNF`.

```
   # Raw images are fixed0 and moving0, both two-dimensional
   pp = img -> imfilter_gaussian(img, [3, 3])
   fixed = pp(fixed0)
   # We'll use a 5x7 grid of apertures
   knots = (linspace(1, size(fixed,1), 5), linspace(1, size(fixed,2), 7))
   # Allow shifts of up to 30 pixels in any direction
   maxshift = (30,30)

   # Create the algorithm-object
   alg = AperturesMismatch(fixed, knots, maxshift, pp)

   mon = monitor(alg, (:Es, :cs, :Qs, :mmis))

   # Run the algorithm
   mon = driver(algorithm, moving0, mon)
```

"""
function AperturesMismatch{K,N}(fixed, knots::NTuple{N,K}, maxshift, preprocess=identity; normalization=:pixels, thresh_fac=NaN, thresh=nothing, correctbias::Bool=true, pid=1, dev=-1)
    gridsize = map(length, knots)
    if isa(fixed, Tuple)
	if isnan(thresh_fac)
	    thresh_fac = (0.5)^ndims(fixed[1])
	end
	size(fixed[1]) == size(fixed[2]) || error("Even and odd fixed images must be the same size")
        nimages(fixed[1]) == 1 && nimages(fixed[2]) ==1 || error("Register to a single image")
	if thresh == nothing
            thresh = (thresh_fac/prod(gridsize)) * (normalization==:pixels ? length(fixed[1]) : sumabs2(fixed[1]))
        end
        T = eltype(fixed[1]) <: AbstractFloat ? eltype(fixed[1]) : Float32
    else
	if isnan(thresh_fac)
	    thresh_fac = (0.5)^ndims(fixed)
	end
	nimages(fixed) == 1 || error("Register to a single image")
	if thresh == nothing
            thresh = (thresh_fac/prod(gridsize)) * (normalization==:pixels ? length(fixed) : sumabs2(fixed))
        end
        T = eltype(fixed) <: AbstractFloat ? eltype(fixed) : Float32
    end
    # T = Float64   # Ipopt requires Float64
    Es = ArrayDecl(Array{T,N}, gridsize)
    cs = ArrayDecl(Array{Vec{N,T},N}, gridsize)
    Qs = ArrayDecl(Array{Mat{N,N,T},N}, gridsize)
    mmsize = map(x->2x+1, maxshift)
    mmis = ArrayDecl(Array{NumDenom{T},2*N}, (mmsize...,gridsize...))
    AperturesMismatch{typeof(fixed),T,K,N}(fixed, knots, maxshift, T(thresh), preprocess, normalization, correctbias, Es, cs, Qs, mmis, pid, dev, Dict{Symbol,Any}())
end

function worker(algorithm::AperturesMismatch, img, tindex, mon)
    moving0 = timedim(img) == 0 ? img : slice(img, "t", tindex)
    moving = algorithm.preprocess(moving0,tindex)
    gridsize = map(length, algorithm.knots)
    use_cuda = algorithm.dev >= 0
    if use_cuda
        device(algorithm.dev)
	if isodd(tindex)
            d_fixed  = algorithm.cuda_objects[:d_fixed][1]
        else
	    d_fixed  = algorithm.cuda_objects[:d_fixed][2]
	end
        d_moving = algorithm.cuda_objects[:d_moving]
        cms      = algorithm.cuda_objects[:cms]
        copy!(d_moving, moving)
        cs = coords_spatial(img)
        aperture_centers = aperture_grid(size(img)[cs], gridsize)
        mms = allocate_mmarrays(eltype(cms), gridsize, algorithm.maxshift)
        mismatch_apertures!(mms, d_fixed, d_moving, aperture_centers, cms; normalization=algorithm.normalization)
    else
	if isodd(tindex)
            mms = mismatch_apertures(algorithm.fixed[1], moving, gridsize, algorithm.maxshift; normalization=algorithm.normalization)
	else
            mms = mismatch_apertures(algorithm.fixed[2], moving, gridsize, algorithm.maxshift; normalization=algorithm.normalization)
	end
    end
    # displaymismatch(mms, thresh=10)
    if algorithm.correctbias
        correctbias!(mms)
    end
    Es = zeros(size(mms))
    cs = Array(Any, size(mms))
    Qs = Array(Any, size(mms))
    thresh = algorithm.thresh
    for i = 1:length(mms)
        Es[i], cs[i], Qs[i] = qfit(mms[i], thresh; opt=false)
    end
    monitor!(mon, :Es, Es)
    monitor!(mon, :cs, cs)
    monitor!(mon, :Qs, Qs)
    if haskey(mon, :mmis)
        mmis = interpolate_mm!(mms)
        R = CartesianRange(size(mon[:mmis])[ndims(mmis)+1:end])
        colons = ntuple(d->Colon(), ndims(mmis))
        _copy_mm!(mon[:mmis], mmis, colons, R)
    end
    mon
end

function _copy_mm!(dest, src, colons, R)
    for (I, mm) in zip(R, src)
        dest[colons..., I] = mm.data.coefs
    end
    dest
end

cudatype{T<:Union{Float32,Float64}}(::Type{T}) = T
cudatype(::Any) = Float32

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
type PreprocessSNF  # Shot-noise filtered
    bias::Float32
    sigmalp::Vector{Float32}
    sigmahp::Vector{Float32}
end
# PreprocessSNF(bias::T, sigmalp, sigmahp) = PreprocessSNF{T}(bias, T[sigmalp...], T[sigmahp...])

function Base.call(pp::PreprocessSNF, A::AbstractArray, tindex=0)
    Af = sqrt_subtract_bias(A, pp.bias)
    imfilter_gaussian(highpass(Af, pp.sigmahp), pp.sigmalp)
end

function sqrt_subtract_bias(A, bias)
#    T = typeof(sqrt(one(promote_type(eltype(A), typeof(bias)))))
    T = Float32
    out = Array(T, size(A))
    for I in eachindex(A)
        @inbounds out[I] = sqrt(max(zero(T), convert(T, A[I]) - bias))
    end
    out
end

end # module
