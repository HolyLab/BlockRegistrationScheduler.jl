module RegisterWorkerRigid

using Images, AffineTransforms, Interpolations
using BlockRegistration, RegisterCore, RegisterMismatch, RegisterOptimize
using BlockRegistrationScheduler, RegisterWorkerShell

import RegisterWorkerShell: worker

export Rigid

type Rigid{A<:AbstractArray,T} <: AbstractWorker
    fixed::A
    thresh::T
    fixedpa::Tuple{Vector{T},Matrix{T}}
    SD::Matrix{T}
    pat::Bool
    params
    workerpid::Int
end

function Rigid(fixed; thresh_fac=(0.5)^ndims(fixed), thresh=nothing, SD = eye(ndims(fixed)), pat::Bool=true, pid=1, kwargs...)
    nimages(fixed) == 1 || error("Register to a single image")
    if pat
        fixedpa = principalaxes(fixed)
        T = eltype(fixedpa[1])
        SDm = convert(Matrix{T}, SD)
    else
        SDm = convert(Matrix, SD)
        T = eltype(SDm)
        T = T <: AbstractFloat ? T : Float64
        na = convert(T,NaN)
        fixedpa = (fill(na, ndims(fixed)), fill(na, ndims(fixed), ndims(fixed)))
    end
    cthresh = thresh == nothing ? thresh_fac*sumabs2(fixed) : thresh
    params = Dict{Symbol,Any}(kwargs)
    Rigid{typeof(fixed),T}(fixed, convert(T, cthresh), fixedpa, SDm, pat, params, pid)
end

function worker(algorithm::Rigid, img, tindex, mon)
    moving = timedim(img) == 0 ? img : img["t", tindex]
    if algorithm.pat
        tfms = pat_rotation(algorithm.fixedpa, moving, algorithm.SD)
        mov_etp = extrapolate(interpolate(moving, BSpline{Quadratic{Flat}}, OnCell), NaN)
        penalty = tform -> (mm = mismatch0(algorithm.fixed, transform(mov_etp, tform)); ratio(mm, algorithm.thresh, convert(eltype(mm), Inf)))
        mm = map(penalty, tfms)
        tfm = tfms[indmin(mm)]
    else
        tfm = tformeye(ndims(moving))
    end
    tfm, mismatch = optimize_rigid(data(algorithm.fixed), data(moving), tfm, [size(algorithm.fixed)...]/2, algorithm.SD, thresh=algorithm.thresh; print_level=get(algorithm.params, :print_level, 0))

    # There are no Rigid parameters that are expected as outputs,
    # so no need to call monitor!(mon, algorithm)
    monitor!(mon, :tform, tfm)
    monitor!(mon, :mismatch, mismatch)
    if haskey(mon, :warped)
        monitor!(mon, :warped, transform(moving, tfm))
    end
end

end # module
