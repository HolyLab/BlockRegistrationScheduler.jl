module RegisterDriver

using Images, JLD, HDF5, Formatting, RegisterWorkerShell

export driver

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
    length(algorithm) == length(mon) || error("Number of monitors must equal number of workers")
    # Perform any needed worker initialization
    for alg in algorithm
        init!(alg)
    end
    try
        n = nimages(img)
        # Initialize the variables in the output JLD file
        jldopen(outfile, "w") do file
            dsets = Dict{Symbol,Any}()
            have_unpackable = false
            for (k,v) in mon[1]
                kstr = string(k)
                if isa(v, Number)
                    write(file, kstr, Array(typeof(v), n))
                    dsets[k] = file[kstr]
                elseif isa(v, Array) || isa(v, SharedArray)
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
            fs = FormatSpec("0$(ndigits(n))d")
            if have_unpackable
                for i = 1:n
                    g_create(file, string("stack", fmt(fs, i)))
                end
            end
            # Run the jobs
            nextidx = 0
            getnextidx() = (nextidx += 1)
            writing_mutex = RemoteRef()
            @sync begin
                for (i,alg) in enumerate(algorithm)
                    @async begin
                        while (idx = getnextidx()) <= n
                            remotecall_fetch(workerpid(alg), worker, alg, img, idx, mon[i])
                            # Save the results
                            put!(writing_mutex, true)  # grab the lock
                            try
                                local g
                                if have_unpackable
                                    g = file[string("stack", fmt(fs, idx))]
                                end
                                for (k,v) in mon[i]
                                    if isa(v, Number)
                                        dsets[k][idx] = v
                                    elseif isa(v, Array) || isa(v, SharedArray)
                                        colons = [Colon() for i = 1:ndims(v)]
                                        dsets[k][colons..., idx] = sdata(v)
                                    else
                                        g[string(k)] = v
                                    end
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
        for alg in algorithm
            close!(alg)
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

end # module
