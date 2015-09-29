using Images, JLD, HDF5, Formatting

"""
`driver(outfile, info, img, mon)` performs registration of the
image(s) in `img` according to the algorithm selected by
`info`. `info` is either a single instance, or for parallel
computation a vector of instances, of `RegisterWorker` types.  See
that module for more details.

Results are saved in `outfile` according to the information in `mon`.
`mon` is a `Dict`, or for parallel computation a vector of `Dicts` of
the same length as `info`.  The data saved correspond to the keys
(always `Symbol`s) in `mon`, and the values are used for communication
between the worker(s) and the driver.  The usual way to set up `mon`
is like this:

```
    info = RegisterRigid(fixed, params...)   # A RegisterWorker algorithm
    mon = monitor(info, (:tform,:mismatch))  # List of variables to record
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
function driver(outfile, info::Vector, img, mon::Vector)
    length(info) == length(mon) || error("Number of monitors must equal number of workers")
    n = size(img, "t")
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
                write(file, kstr, Array(eltype(v), size(v)..., n))  # FIXME: need d_create for JLD
                dsets[k] = file[kstr]
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
            for (i,alg) in enumerate(info)
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
end

driver(outfile, info, img, mon) = driver(outfile, [info], img, [mon])
