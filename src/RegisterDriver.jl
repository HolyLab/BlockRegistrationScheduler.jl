__precompile__()

module RegisterDriver

using Images, JLD, HDF5, FixedSizeArrays, Formatting
using RegisterCore
using RegisterWorkerShell

export driver
export PreprocessSNF

"""
`driver(outfile, algorithm, img, mon)` performs registration of the
image(s) in `img` according to the algorithm selected by
`algorithm`. `algorithm` is either a single instance, or for parallel
computation a vector of instances, of `AbstractWorker` types.  See the
`RegisterWorkerShell` module for more details.

Results are saved in `outfile` according to the information in `mon`.
`mon` is a `Dict`, or for parallel computation a vector of `Dicts` of
the same length as `algorithm`.  The data saved correspond to the keys
(always `Symbol`s) in `mon`, and the values are used for communication
between the worker(s) and the driver.  The usual way to set up `mon`
is like this:

```
    algorithm = RegisterRigid(fixed, params...)   # An AbstractWorker algorithm
    mon = monitor(algorithm, (:tform,:mismatch))  # List of variables to record
```

The list of symbols, taken from the field names of `RegisterRigid`,
specifies the pieces of information to be communicated back to the
driver process for saving and/or display to the user.  It's also
possible to request local variables in the worker, as long as the
worker has been written to look for such settings:

```
    # <in the worker algorithm>
    monitor_copy!(mon, :extra, extra)
```

which will save `extra` only if `:extra` is a key in `mon`.
"""
function driver(outfile::AbstractString, algorithm::Vector, img, mon::Vector)
    nworkers = length(algorithm)
    length(mon) == nworkers || error("Number of monitors must equal number of workers")
    use_workerprocs = nworkers > 1 || workerpid(algorithm[1]) != myid()
    rralgorithm = Array{RemoteChannel}(nworkers)
    if use_workerprocs
        # Push the algorithm objects to the worker processes. This elminates
        # per-iteration serialization penalties, and ensures that any
        # initalization state is retained.
        for i = 1:nworkers
            alg = algorithm[i]
            rralgorithm[i] = put!(RemoteChannel(workerpid(alg)), alg)
        end
        # Perform any needed worker initialization
        @sync for i = 1:nworkers
            p = workerpid(algorithm[i])
            @async remotecall_fetch(init!, p, rralgorithm[i])
        end
    else
        init!(algorithm[1])
    end
    try
        n = nimages(img)
        fs = FormatSpec("0$(ndigits(n))d")  # group names of unpackable objects
        jldopen(outfile, "w") do file
            dsets = Dict{Symbol,Any}()
            firstsave = SharedArray(Bool, 1)
            firstsave[1] = true
            have_unpackable = SharedArray(Bool, 1)
            have_unpackable[1] = false
            # Run the jobs
            nextidx = 0
            getnextidx() = nextidx += 1
            writing_mutex = RemoteChannel()
            @sync begin
                for i = 1:nworkers
                    alg = algorithm[i]
                    @async begin
                        while (idx = getnextidx()) <= n
                            if use_workerprocs
                                remotecall_fetch(println, workerpid(alg), "Worker ", workerpid(alg), " is working on ", idx)
                                # See https://github.com/JuliaLang/julia/issues/22139
                                tmp = remotecall_fetch(worker, workerpid(alg), rralgorithm[i], img, idx, mon[i])
                                copy_all_but_shared!(mon[i], tmp)
                            else
                                println("Working on ", idx)
                                mon[1] = worker(algorithm[1], img, idx, mon[1])
                            end
                            # Save the results
                            put!(writing_mutex, true)  # grab the lock
                            try
                                local g
                                if firstsave[]
                                    firstsave[] = false
                                    have_unpackable[] = initialize_jld!(dsets, file, mon[i], fs, n)
                                end
                                if fetch(have_unpackable[])
                                    g = file[string("stack", fmt(fs, idx))]
                                end
                                for (k,v) in mon[i]
                                    if isa(v, Number)
                                        dsets[k][idx] = v
                                        continue
                                    elseif isa(v, Array) || isa(v, SharedArray)
                                        v = nicehdf5(v)
                                        if eltype(v) <: HDF5.HDF5BitsKind
                                            colons = [Colon() for i = 1:ndims(v)]
                                            dsets[k][colons..., idx] = v
                                            continue
                                        end
                                    end
                                    g[string(k)] = v
                                end
                            finally
                                take!(writing_mutex)   # release the lock
                            end
                        end
                    end
                end
            end
        end
    finally
        # Perform any needed worker cleanup
        if use_workerprocs
            @sync for i = 1:nworkers
                p = workerpid(algorithm[i])
                @async remotecall_fetch(close!, p, rralgorithm[i])
            end
        else
            close!(algorithm[1])
        end
    end
end

driver(outfile::AbstractString, algorithm::AbstractWorker, img, mon::Dict) = driver(outfile, [algorithm], img, [mon])

"""
`mon = driver(algorithm, img, mon)` performs registration on a single
image, returning the results in `mon`.
"""
function driver(algorithm::AbstractWorker, img, mon::Dict)
    nimages(img) == 1 || error("With multiple images, you must store results to a file")
    init!(algorithm)
    worker(algorithm, img, 1, mon)
    close!(algorithm)
    mon
end

# Initialize the datasets in the output JLD file.
# We wait to do this until we get back one valid `mon` object,
# to get the sizes of any returned arrays.
function initialize_jld!(dsets, file, mon, fs, n)
    have_unpackable = false
    for (k,v) in mon
        kstr = string(k)
        if isa(v, Number)
            write(file, kstr, Array(typeof(v), n))
            dsets[k] = file[kstr]
        elseif isa(v, Array) || isa(v, SharedArray)
            v = nicehdf5(v)
            if eltype(v) <: HDF5.HDF5BitsKind
                fullsz = (size(v)..., n)
                dsets[k] = d_create(file.plain, kstr, datatype(eltype(v)), dataspace(fullsz))
            else
                write(file, kstr, Array(eltype(v), size(v)..., n))  # might fail if it's too big, but we tried
            end
            dsets[k] = file[kstr]
        elseif isa(v, ArrayDecl)  # maybe this never happens?
            fullsz = (v.arraysize..., n)
            dsets[k] = d_create(file.plain, kstr, datatype(eltype(v)), dataspace(fullsz))
        else
            have_unpackable = true
        end
    end
    if have_unpackable
        for i = 1:n
            g_create(file, string("stack", fmt(fs, i)))
        end
    end
    have_unpackable
end

function nicehdf5{T<:FixedArray}(v::Union{Array{T},SharedArray{T}})
    nicehdf5(reinterpret(eltype(T), sdata(v), (size(eltype(v))..., size(v)...)))
end

function nicehdf5{T<:NumDenom}(v::Union{Array{T},SharedArray{T}})
    nicehdf5(reinterpret(eltype(T), sdata(v), (2, size(v)...)))
end

nicehdf5(v::SharedArray) = sdata(v)
nicehdf5(v) = v

function copy_all_but_shared!(dest, src)
    for (k, v) in src
        if !isa(v, SharedArray)
            dest[k] = v
        end
    end
    dest
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
(for the high-pass).  You can pass `sigmalp=zeros(n)` to skip low-pass
filtering, and `sigmahp=fill(Inf, n)` to skip high-pass filtering.
"""
type PreprocessSNF  # Shot-noise filtered
    bias::Float32
    sigmalp::Vector{Float32}
    sigmahp::Vector{Float32}
end
# PreprocessSNF(bias::T, sigmalp, sigmahp) = PreprocessSNF{T}(bias, T[sigmalp...], T[sigmahp...])

function preprocess(pp::PreprocessSNF, A::AbstractArray)
    Af = sqrt_subtract_bias(A, pp.bias)
    imfilter(highpass(Af, pp.sigmahp), KernelFactors.IIRGaussian((pp.sigmalp...)))
end
(pp::PreprocessSNF)(A::AbstractArray) = preprocess(pp, A)
(pp::PreprocessSNF)(A::ImageMeta) = shareproperties(A, pp(data(A)))
# For SubArrays, extend to the parent along any non-sliced
# dimension. That way, we keep any information from padding.
function (pp::PreprocessSNF)(A::SubArray)
    Bpad = preprocess(pp, paddedview(A))
    trimmedview(Bpad, A)
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
