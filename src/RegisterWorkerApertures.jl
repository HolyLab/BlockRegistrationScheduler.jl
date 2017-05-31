__precompile__()

module RegisterWorkerApertures

using Images, AffineTransforms, Interpolations
using RegisterCore, RegisterDeformation, RegisterFit, RegisterPenalty, RegisterOptimize
# Note: RegisterMismatch/RegisterMismatchCuda is selected below
using RegisterWorkerShell, RegisterDriver

import RegisterWorkerShell: worker, init!, close!

export Apertures, monitor, monitor!, worker, workerpid

type Apertures{A<:AbstractArray,T,K,N} <: AbstractWorker
    fixed::A
    knots::NTuple{N,K}
    maxshift::NTuple{N,Int}
    affinepenalty::AffinePenalty{T,N}
    overlap::NTuple{N,Int}
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
    fixed = algorithm.fixed
    T = cudatype(eltype(fixed))
    d_fixed  = CudaPitchedArray(myconvert(Array{T}, sdata(data(fixed))))
    algorithm.cuda_objects[:d_fixed] = d_fixed
    algorithm.cuda_objects[:d_moving] = similar(d_fixed)
    gridsize = map(length, algorithm.knots)
    aperture_width = default_aperture_width(algorithm.fixed, gridsize)
    algorithm.cuda_objects[:cms] = CMStorage(T, aperture_width, algorithm.maxshift)
    nothing
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

"""
`alg = Apertures(fixed, knots, maxshift, λ, [preprocess=identity]; kwargs...)`
creates a worker-object for performing "apertured" (blocked)
registration.  `fixed` is the reference image, `knots` specifies the
grid of apertures, `maxshift` represents the largest shift (in pixels)
that will be evaluated, and `λ` is the coefficient for the deformation
penalty (higher values enforce a more affine-like
deformation). `preprocess` allows you to apply a transformation (e.g.,
filtering) to the `moving` images before registration; `fixed` should
already have any such transformations applied.

Alternatively, `λ` may be specified as a `(λmin, λmax)` tuple, in
which case the "best" `λ` is chosen for you automatically via the
algorithm described in `auto_λ`.  If you `monitor` the variable
`datapenalty`, you can inspect the quality of the sigmoid used to
choose `λ`.

Registration is performed by calling `driver`.

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
   # Try a range of λ values
   λrange = (1e-6, 100)

   # Create the algorithm-object
   alg = Apertures(fixed, knots, maxshift, λrange, pp)

   # Monitor the datapenalty, the chosen value of λ, the deformation
   # u, and also collect the corrected (warped) image. By asking for
   # :warped0, we apply the warping to the unfiltered moving image
   # (:warped would refer to the filtered moving image).
   # We pre-allocate space for :warped0 to illustrate a trick for
   # reducing the overhead of communication between worker and driver
   # processes, even though this example uses just a single process
   # (see `monitor` for further detail).  The other arrays are small,
   # so we don't worry about overhead for them.
   mon = monitor(alg, (), Dict(:λs=>0, :datapenalty=>0, :λ=>0, :u=>0, :warped0 => Array(Float64, size(fixed))))

   # Run the algorithm
   mon = driver(algorithm, moving0, mon)

   # Plot the datapenalty and see how sigmoidal it is. Assumes you're
   # `using Immerse`.
   λs = mon[:λs]
   datapenalty = mon[:datapenalty]
   plot(x=λs, y=datapenalty, xintercept=[mon[:λ]], Geom.point, Geom.vline, Guide.xlabel("λ"), Guide.ylabel("Data penalty"), Scale.x_log10)
```

"""
function Apertures{K,N}(fixed, knots::NTuple{N,K}, maxshift, λrange, preprocess=identity; overlap=zeros(Int, N), normalization=:pixels, thresh_fac=(0.5)^ndims(fixed), thresh=nothing, correctbias::Bool=true, pid=1, dev=-1)
    gridsize = map(length, knots)
    overlap_t = (overlap...) #Make tuple
    length(overlap) == N || throw(DimensionMismatch("overlap must have $N entries"))
    nimages(fixed) == 1 || error("Register to a single image")
    isa(λrange, Number) || isa(λrange, Tuple{Number,Number}) || error("λrange must be a number or 2-tuple")
    if thresh == nothing
        thresh = (thresh_fac/prod(gridsize)) * (normalization==:pixels ? length(fixed) : sumabs2(fixed))
    end
    # T = eltype(fixed) <: AbstractFloat ? eltype(fixed) : Float32
    T = Float64   # Ipopt requires Float64
    λrange = isa(λrange, Number) ? T(λrange) : (T(first(λrange)), T(last(λrange)))
    Apertures{typeof(fixed),T,K,N}(fixed, knots, maxshift, AffinePenalty{T,N}(knots, first(λrange)), overlap_t, λrange, T(thresh), preprocess, normalization, correctbias, pid, dev, Dict{Symbol,Any}())
end

function worker(algorithm::Apertures, img, tindex, mon)
    moving0 = getindex_t(img, tindex)
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
        aperture_centers = aperture_grid(size(img, cs...), gridsize)
        mms = allocate_mmarrays(eltype(cms), gridsize, algorithm.maxshift)
        mismatch_apertures!(mms, d_fixed, d_moving, aperture_centers, cms; normalization=algorithm.normalization)
    else
        #mms = mismatch_apertures(algorithm.fixed, moving, gridsize, algorithm.maxshift; normalization=algorithm.normalization)
        cs = coords_spatial(img) #
        aperture_centers = aperture_grid(size(img, cs...), gridsize)  #
        aperture_width = default_aperture_width(algorithm.fixed, gridsize, algorithm.overlap)  #
        mms = mismatch_apertures(algorithm.fixed, moving, aperture_centers, aperture_width, algorithm.maxshift; normalization=algorithm.normalization)  #
    end
    # displaymismatch(mms, thresh=10)
    if algorithm.correctbias
        correctbias!(mms)
    end
    E0 = zeros(size(mms))
    cs = Array{Any}(size(mms))
    Qs = Array{Any}(size(mms))
    thresh = algorithm.thresh
    for i = 1:length(mms)
        E0[i], cs[i], Qs[i] = qfit(mms[i], thresh; opt=false)
    end
    mmis = interpolate_mm!(mms)
    λrange = algorithm.λrange
    if isa(λrange, Number)
        ϕ, mismatch = RegisterOptimize.fixed_λ(cs, Qs, algorithm.knots, algorithm.affinepenalty, mmis)
    else
        ϕ, mismatch, λ, λs, dp, quality = RegisterOptimize.auto_λ(cs, Qs, algorithm.knots, algorithm.affinepenalty, mmis, λrange)
        monitor!(mon, :λ, λ)
        monitor!(mon, :λs, λs)
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

cudatype{T<:Union{Float32,Float64}}(::Type{T}) = T
cudatype(::Any) = Float32

myconvert{T}(::Type{Array{T}}, A::Array{T}) = A
myconvert{T}(::Type{Array{T}}, A::AbstractArray) = copy!(Array{T}(size(A)), A)

end # module
