__precompile__()

module RegisterWorkerRigid

using Images, AffineTransforms, Interpolations
using RegisterCore, RegisterOptimize, RegisterFit, RegisterMismatch
using RegisterWorkerShell
using ProgressMeter, Optim, RegisterPenalty
using SIUnits
using OCPI

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
    preprocess  # likely of type PreprocessSNF, but could be a function
end

function Rigid(fixed; thresh_fac=(0.5)^ndims(fixed), thresh=nothing, SD = eye(ndims(fixed)), pat::Bool=true, pid=1, preprocess=identity, kwargs...)
    nimages(fixed) == 1 || error("Register to a single image")
    if issubtype(eltype(SD), SIUnits.SIQuantity)
        SD = map(float,SD) * 1e6 #to micrometers
    end
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
    Rigid{typeof(fixed),T}(fixed, convert(T, cthresh), fixedpa, SDm, pat, params, pid, preprocess)
end

function init!(algorithm::Rigid)
    eval(:(using RegisterMismatch))
end

function worker(algorithm::Rigid, img, tindex, mon)
#     print("in worker\n")
#     @show img[1]
#     @show img["t", tindex]
    moving = timedim(img) == 0 ? img : img["t", tindex]
    moving = algorithm.preprocess(deepcopy(moving))
#     print("here2\n")
    if algorithm.pat
        tfms = pat_rotation(algorithm.fixedpa, moving, algorithm.SD)
        mov_etp = extrapolate(interpolate(moving, BSpline(Quadratic(Flat())), OnCell()), NaN)
        penalty = tform -> (mm = mismatch0(algorithm.fixed, transform(mov_etp, tform)); ratio(mm, algorithm.thresh, convert(eltype(mm), Inf)))
        mm = map(penalty, tfms)
        tfm = tfms[indmin(mm)]
    else
        tfm = ndims(moving) == 3 ? tformrotate(1e-5*randn(3)) : tformeye(2)
    end

    mm_norm = get(algorithm.params, :normalization, :intensity)
    num_restrictions = map(x->convert(Int64, x), [get(algorithm.params, :num_restrictions, zeros(ndims(moving)))...])
#     print("here3\n")
    f = deepcopy(data(algorithm.fixed))
    m = data(moving)
#     print("here4\n")
    if haskey(algorithm.params, "max_radians") || haskey(algorithm.params, "max_shift")
        gsearch = true
    else
        gsearch = false
    end
    
    max_radians = [get(algorithm.params, :max_radians, fill(pi, ndims(moving) == 2 ? 1 : ndims(moving)))...]
    max_shift0 = [get(algorithm.params, :max_shift, [floor(Int, size(moving,x)/2) for x=1:ndims(moving)])...]
    SD = deepcopy(algorithm.SD)
    if issubtype(eltype(max_shift0), SIUnits.SIQuantity) #convert to pixel units
        max_shift0 = ceil(Int, map(float, max_shift0)./(diag(SD))) #assumes max_shift and SD are in the same units
    end #otherwise assume it's in pixel units already
    max_shift = deepcopy(max_shift0)

    restr_remaining = deepcopy(num_restrictions)
    while any(restr_remaining.>0)
        curr_restr = find(x->x>0, restr_remaining)
        m = Images.restrict(m, curr_restr)
        f = Images.restrict(f, curr_restr)
        #after restriction, the border of the image is unreliable
        rngs = convert(Array{UnitRange}, [2:(size(m,i)-1) for i=1:ndims(m)])
        m = m[rngs...]
        f = f[rngs...]
        for i in curr_restr
            SD[i,i]*=2
            max_shift[i] = ceil(Int, max_shift[i]/2)
        end
        restr_remaining-=1
    end
  
#for looking at just a subregion  
#     if !any(x->isinf(x), max_shift)
#         dim_ctrs = [round(Int, size(f,i)/2) for i=1:ndims(f)]
#         fac = 10
#         frameshift = ceil(Int, max_shift./(diag(SD)*1e6)) #rescale since SD is in meters 
#         idxs = convert(Array{UnitRange}, [(dim_ctrs[i]-fac*frameshift[i]):(dim_ctrs[i]+fac*frameshift[i]) for i=1:ndims(f)])
#         f = f[idxs...]
#         m = m[idxs...]
#     end
    if gsearch
        rotp, dx, best_mm = shift_rot_gridsearch(f,m,max_radians, max_shift, SD; mm_norm=mm_norm, thresh=algorithm.thresh)
        rotm = length(rotp) == 1 ? rotation2(rotp) : rotation3(rotp)        
        tfm = AffineTransform(rotm, dx)
    end
    restr_factors = convert(Array{Float64}, [2^x for x in num_restrictions])
    
#     f2 = deepcopy(data(algorithm.fixed))
#     m2 = deepcopy(data(moving))
    #for looking at just a subregion
#     dim_ctrs = [round(Int, size(f2,i)/2) for i=1:ndims(f2)]
#     fac = 10
#     frameshift0 = ceil(Int, max_shift0./(diag(algorithm.SD)*1e6))
#     rngs = [dim_ctrs[i]-fac*frameshift0[i]:dim_ctrs[i]+fac*frameshift0[i] for i=1:ndims(f2)]
#     f2 = f2[rngs...]
#     m2 = m2[rngs...]
    
    #     tfm, mismatch = optimize_rigid(Images.restrict(data(algorithm.fixed)), Images.restrict(data(moving)), tfm, [max_shiftxy;max_shiftxy;max_shiftz], algorithm.SD, thresh=algorithm.thresh; print_level=get(algorithm.params, :print_level, 0))
    #tfm, mismatch = optimize_rigid(f2, m2, tfm, max_shift0, algorithm.SD, thresh=algorithm.thresh; print_level=get(algorithm.params, :print_level, 0))
#     mm_temp= mismatch0(algorithm.fixed, transform(data(moving), AffineTransform(algorithm.SD\(rotm*algorithm.SD), dx.*restr_factors)); normalization=mm_norm)
#     @show mismatch = ratio(mm_temp, 0, convert(eltype(mm_temp), Inf))

    rotp, dx, mismatch = optimize_rigid(f, m, tfm, max_shift, max_radians, SD, thresh=algorithm.thresh; print_level=get(algorithm.params, :print_level, 0))
    dx = dx.*restr_factors

#on full image
# tfm = AffineTransform(rotm, dx.*restr_factors)
# rotp, dx, mismatch = optimize_rigid(data(algorithm.fixed), data(moving), tfm, ceil(Int, max_shift0), algorithm.SD, thresh=algorithm.thresh; print_level=get(algorithm.params, :print_level, 0))
    rotm = length(rotp) == 1 ? rotation2(rotp) : rotation3(rotp)
    tfm = AffineTransform(algorithm.SD\(rotm*algorithm.SD), dx)
#     @show rotp
#     @show dx
    @show mismatch
    # There are no Rigid parameters that are expected as outputs,
    # so no need to call monitor!(mon, algorithm)
    monitor!(mon, :tform, tfm)
    monitor!(mon, :mismatch, mismatch)
    if haskey(mon, :warped)
        monitor!(mon, :warped, transform(moving, tfm))
    end
    mon
end

#Is there an easy way to write this so that it works in both 2D and 3D?
function shift_rot_gridsearch{T1,T2}(f::AbstractArray{T1,3}, m::AbstractArray{T2,3}, max_radians::Vector, max_shift::Vector, SD::Matrix ; mm_norm=:intensity, thresh = 0)
    if all(x->isfinite(x), max_radians)
        incr = pi/128  #TODO: make this user-specified
        r_rngs = [-max_radians[i]:incr:max_radians[i] for i=1:ndims(f)]
        rotp = zeros(length(max_radians))
        dx = zeros(length(max_shift))
        rotm = length(rotp)==2 ? rotation2(rotp) : rotation3(rotp)
        mm = mismatch0(f, transform(m, AffineTransform(SD\(rotm*SD), dx)); normalization=mm_norm)
        best_mm = ratio(mm, 0, convert(eltype(mm), Inf))
        best_mm_array = []
        idx = []
        search_size = prod([length(i) for i in r_rngs])
        p = Progress(search_size, 2, "Searching a grid of possible rotations and shifts")
        print("Number of rotations to evaluate: $search_size\n")

        for i in r_rngs[1]
            for j in r_rngs[2]
                for k in r_rngs[3]
                    m_warped = transform(m, AffineTransform(SD\(rotation3([i;j;k])*SD), [0;0;0]))
                    mm_array = mismatch(f, m_warped, max_shift; normalization=mm_norm)
                    idx = indmin_mismatch(mm_array, 0)
                    curr_mm = ratio(mm_array[idx], thresh, convert(eltype(mm_array[idx]), Inf))
                    #interpolate and optimize to find optimal sub-pixel shift
                    mm_i = interpolate_mm!(mm_array)
                    nd2float = nd->ratio(nd, thresh, convert(eltype(mm_i[0,0,0]), Inf))
                    grd_nd2float = nd->ratio(nd, -Inf, convert(eltype(mm_i[0,0,0]), Inf))
                    fnc = xyz -> nd2float(mm_i[xyz...])
                    ga = Array(NumDenom, 3)
                    function g!(xyz,storage)
                        nd_g = Interpolations.gradient!(ga, mm_i, xyz...) #this gives us dmm/dnum and dmm/ddenom
                        nd_v = mm_i[xyz...]
                        #find derivative using quotient rule
                        q_rule = (nd_v, nd_g) -> convert(Array{Float64}, [(nd_v.denom * nd_g[ii].num - nd_v.num * nd_g[ii].denom)/(nd_v.denom)^2 for ii=1:length(xyz)])
                        dxyz = q_rule(nd_v, nd_g)
                        copy!(storage, dxyz)
                    end
                    df = DifferentiableFunction(fnc, g!)
                    initial_x = float([idx.I...])
                    constraints = Optim.ConstraintsBox(initial_x-1,initial_x+1) #constrain within +=1 grid point
                    rslt = Optim.cg(df, initial_x, constraints=constraints)
                    #TODO: also interpolate and optimize rotation? seems less important
                    if rslt.f_minimum < best_mm                        
                        best_mm = rslt.f_minimum
                        rotp = [i;j;k]
                        dx = rslt.minimum
                    end
                    next!(p)
                end
            end
        end
         @show best_mm
    else
        error("invalid max_radians")
    end
    return rotp, dx, best_mm
end

#2D
function shift_rot_gridsearch{T1,T2}(f::AbstractArray{T1,2}, m::AbstractArray{T2,2}, max_radians::Vector, max_shift::Vector, SD::Matrix ; mm_norm=:intensity, thresh = 0)
    if all(x->isfinite(x), max_radians)
        incr = pi/128  #TODO: make this user-specified
        r_rngs = [-max_radians[i]:incr:max_radians[i] for i=1:length(max_radians)]
        rotp = zeros(length(max_radians))
        dx = zeros(length(max_shift))
        rotm = length(rotp)==1 ? rotation2(rotp) : rotation3(rotp)
        mm = mismatch0(f, transform(m, AffineTransform(SD\(rotm*SD), dx)); normalization=mm_norm)
        best_mm = ratio(mm, 0, convert(eltype(mm), Inf))
        best_mm_array = []
        idx = []
        search_size = prod([length(i) for i in r_rngs])
        p = Progress(search_size, 2, "Searching a grid of possible rotations and shifts")
        print("Number of rotations to evaluate: $search_size\n")

        for i in r_rngs[1]
            m_warped = transform(m, AffineTransform(SD\(rotation2(i)*SD), [0;0]))
            mm_array = mismatch(f, m_warped, max_shift; normalization=mm_norm)
            idx = indmin_mismatch(mm_array, 0)
            curr_mm = ratio(mm_array[idx], thresh, convert(eltype(mm_array[idx]), Inf))
            #interpolate and optimize to find optimal sub-pixel shift
            mm_i = interpolate_mm!(mm_array)
            nd2float = nd->ratio(nd, thresh, convert(eltype(mm_i[0,0]), Inf))
            grd_nd2float = nd->ratio(nd, -Inf, convert(eltype(mm_i[0,0]), Inf))
            fnc = xy -> nd2float(mm_i[xy...])
            ga = Array(NumDenom, 2)
            function g!(xy,storage)
                nd_g = Interpolations.gradient!(ga, mm_i, xy...) #this gives us dmm/dnum and dmm/ddenom
                nd_v = mm_i[xy...]
                #find derivative using quotient rule
                q_rule = (nd_v, nd_g) -> convert(Array{Float64}, [(nd_v.denom * nd_g[ii].num - nd_v.num * nd_g[ii].denom)/(nd_v.denom)^2 for ii=1:length(xy)])
                dxy = q_rule(nd_v, nd_g)
                copy!(storage, dxy)
            end
            df = DifferentiableFunction(fnc, g!)
            initial_x = float([idx.I...])
            constraints = Optim.ConstraintsBox(initial_x-1,initial_x+1) #constrain within +=1 grid point
            rslt = Optim.cg(df, initial_x, constraints=constraints)
            #TODO: also interpolate and optimize rotation? seems less important
            if rslt.f_minimum < best_mm                        
                best_mm = rslt.f_minimum
                rotp = [i]
                dx = rslt.minimum
            end
            next!(p)
        end
         @show best_mm
    else
        error("invalid max_radians")
    end
    return rotp, dx, best_mm
end

end # module

#returns the best offset found, in physical space units
# function translation_gridsearch(f, m, max_shift, SD, rotp; mm_norm=:intensity)
#     #     xctr = round(Int, size(f, 1)/2)
# #     yctr = round(Int, size(f, 2)/2)
# #     zctr = round(Int, size(f,3)/2)
#     if !any(x->isinf(x), max_shift)
#     #     ftemp = ftemp[xctr-max_shiftxy-1:xctr+max_shiftxy+1, yctr-max_shiftxy-1:yctr+max_shiftxy+1, zctr-2*max_shiftz-1:zctr+max_shiftz+1]
#     #     mtemp = mtemp[xctr-max_shiftxy-1:xctr+max_shiftxy+1, yctr-max_shiftxy-1:yctr+max_shiftxy+1, zctr-2*max_shiftz-1:zctr+max_shiftz+1]
#     #     ftemp = imfilter_gaussian(ftemp, [1.0,1.0,1.0])
#     #     mtemp = imfilter_gaussian(mtemp, [1.0,1.0,1.0])
#         t_rot = length(rotp)==2 ? rotation2(rotp) : rotation3(rotp)
#         t_rngs = [-max_shift[i]:max_shift[i] for i=1:ndims(f)]
#         if ndims(f)==2
#             tfms = [AffineTransform(SD\(t_rot*SD), [i;j]) for i in t_rngs[1], j in t_rngs[2]]
#         else
#             tfms = [AffineTransform(SD\(t_rot*SD), [i;j;k]) for i in t_rngs[1], j in t_rngs[2], k in t_rngs[3]]
#         end
#     #     tfm = AffineTransform(tfm.scalefwd, [tfm.offset[1]*2, tfm.offset[2]*2, tfm.offset[3]])
#     #idea: initialize tfm with brute force gridsearch
#         penalty = tform -> (mm = mismatch0(f, transform(m, tform); normalization=mm_norm); ratio(mm, 0, convert(eltype(mm), Inf)))
#     # penalty = tform -> (mm = mismatch0(ftemp, transform(mtemp, tform)); ratio(mm, 0, convert(eltype(mm), Inf)))
#         print("brute forcing shifts...\n")
#         mm = map(penalty, tfms)
#         dx = tfms[indmin(mm)].offset
#         @show minimum(mm)
#     else
#         error("invalid max_shift")
#     end
#     return dx, minimum(mm)
# end

#returns parameters describing the best rotation found in physical space
#(to apply the rotation to an image with non-uniform voxelspacing, incorporate SD)
# function rotation_gridsearch(f::AbstractArray, m::AbstractArray, max_radians::Vector, SD::Matrix, dx::Vector; mm_norm=:intensity)
#     #now do the same for rotations
#     if !any(x->isinf(x), max_radians)
#         incr = pi/64
#         r_rngs = [-max_radians[i]:incr:max_radians[i] for i=1:ndims(f)]
# #         f_mm = (i,j,k) -> (mm= mismatch0(f, transform(m, tfm_translate*AffineTransform(SD\(rotation3([i;j;k])*SD), [0;0;0])); normalization=:pixels); ratio(mm, 0, convert(eltype(mm), Inf)))
# #         best_mm = f_mm(0,0,0)
#         rotp = zeros(length(dx))
#         rotm = length(rotp)==2 ? rotation2(rotp) : rotation3(rotp)
#         mm_temp= mismatch0(f, transform(m, AffineTransform(SD\(rotm*SD), dx)); normalization=mm_norm)
#         best_mm = ratio(mm_temp, 0, convert(eltype(mm_temp), Inf))
#         
#         print("beginning rotation gridsearch...\n")
#         print("total: $(prod([length(x) for x in r_rngs]))\n")
#         for i in r_rngs[1]
#             for j in r_rngs[2]
#                 for k in r_rngs[3]
#                     mm_temp= mismatch0(f, transform(m, AffineTransform(SD\(rotation3([i;j;k])*SD), dx)); normalization=mm_norm)
#                     curr_mm = ratio(mm_temp, 0, convert(eltype(mm_temp), Inf))
#                     if curr_mm < best_mm                        
#                         best_mm = curr_mm
#                         rotp = [i;j;k]
#                     end
#                 end
#             end
#         end
#         @show best_mm
#     else
#         error("invalid max_radians")
#     end
#     return rotp, best_mm
# end
