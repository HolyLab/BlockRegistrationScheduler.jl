module RegisterWorkerShell

export AbstractWorker, ArrayDecl, close!, init!, maybe_sharedarray, monitor, monitor!, worker, workerpid

"""
An `AbstractWorker` type performs registration on a single
image. Aside from the "moving" image (see `worker` for how this is
specified), all inputs/parameters to the algorithm should be supplied
via fields of an object `algorithm` which is a subtype of
`AbstractWorker`.

See `RegisterWorkerShell` for an overview of the API supported by
`AbstractWorker` types.
"""
abstract AbstractWorker

# Not sure about this next type...
immutable ArrayDecl{A<:AbstractArray,N}
    arraysize::NTuple{N,Int}
end
ArrayDecl{A<:AbstractArray}(::Type{A}, sz) = ArrayDecl{A,ndims(A)}(sz)

Base.eltype{A}(::ArrayDecl{A}) = eltype(A)

"""
# RegisterWorkerShell

This module defines the core operations for all `AbstractWorker`
subtypes.  The exported operations are:

  - `monitor` and `monitor!`: passing results from worker(s) to the driver
  - `init!` and `close!`: functions you may specialize if your algorithm
    needs to initialize or clean up resources
  - `worker`: perform registration on an image
  - `workerpid`: extract the process-id for a given worker
"""
RegisterWorkerShell

"""
`mon = monitor(algorithm, (:var1, :var2, ...))` turns on "monitoring"
(reporting) for fields named `:var`, `:var2`, ... in `algorithm`. This
causes results to be passed back to the driver algorithm, which will
then save the values to disk.

The worker should call `monitor!(mon, algorithm)` to copy the values
into `mon`.

One can monitor additional internal variables in the worker algorithm
by manually adding elements to `mon`. For example,

```
    mon[:warped] = SharedArray(T, sz, pids=[1,2])
```

would set up a SharedArray for communicating back the warped image
from worker-process 2 to the driver-process 1, assuming that you
didn't already define a `warped` field of `algorithm`. A
properly-prepared worker algorithm would store this result by calling
`monitor!(mon, :warped, warped_image)` or, if its computation is
expensive, could first check whether it's being requested:

```
    if haskey(mon, :warped)
        monitor!(mon, :warped, warp(moving, ϕ)) # user wants this, so compute it
    end
```

An important detail is that if `workerpid(algorithm) ≠ 1`, then any
requested `AbstractArray` fields in `algorithm` will be turned into
`SharedArray`s for `mon`.

"""
function monitor{N}(algorithm::AbstractWorker, fields::Union{NTuple{N,Symbol},Vector{Symbol}})
    mon = Dict{Symbol,Any}()
    for f in fields
        isdefined(algorithm, f) || continue
        mon[f] = monitor_field(getfield(algorithm, f))
    end
    mon
end

"""
`monitor!(mon, algorithm)` updates `mon` with the current values of
the fields in `algorithm`.  Workers should call this after all
computations have finished.  See `monitor` for more information.

`monitor!(mon, :parameter, algorithm)` copies just the value of
`algorithm.parameter`, after first checking `haskey(mon, :parameter)`.
"""
function monitor!(mon::Dict{Symbol,Any}, algorithm::AbstractWorker)
    for f in fieldnames(algorithm)
        monitor!(mon, f, getfield(algorithm, f))
    end
    mon
end

function monitor!(mon, fn::Symbol, v::AbstractArray)
    if haskey(mon, fn)
        copy!(mon[fn], v)
    end
    mon
end
function monitor!(mon, fn::Symbol, v)
    if haskey(mon, fn)
        mon[fn] = v
    end
    mon
end

"""
`init!(algorithm)` performs any necessary initialization prior to
beginning a registration sequence using algorithm `algorithm`. The
default action is to return `nothing`. If you require initialization,
specialize this function for your `AbstractWorker` subtype.
"""
init!(algorithm::AbstractWorker, args...) = nothing

"""
`close!(algorithm)` performs any necessary cleanup after a
registration sequence using algorithm `algorithm`. The
default action is to return `nothing`. If you require cleanup,
specialize this function for your `AbstractWorker` subtype.
"""
close!(algorithm::AbstractWorker, args...) = nothing

"""
`worker(algorithm, img, tindex, mon)` causes registration to be performed
using the algorithm and parameters defined by `algorithm`, a subtype of
`AbstractWorker`.  Registration is performed on `img["t", tindex]`.
`mon` should be a `Dict(sym=>value)` that chooses the
outputs/variables to be monitored; see `monitor` for details.

You must define this function for your `AbstractWorker` subtype.
"""
worker(args...) = error("Worker modules must define `worker`")

"""
`workerpid(algorithm)` extracts the `pid` associated with the worker
that will be assigned tasks for `algorithm`.  All `AbstractWorker`
subtypes should include a `workerpid` field, or overload this function
to return 1.
"""
workerpid(w::AbstractWorker) = w.workerpid


## Utility functions
function maybe_sharedarray(A::AbstractArray, pid::Int=1)
    if pid != 1
        S = SharedArray(eltype(A), size(A), pids=union(1, pid))
        copy!(S, A)
    else
        S = A
    end
    S
end

function maybe_sharedarray{T}(::Type{T}, sz, pid::Int=1)
    if pid != 1
        S = SharedArray(T, sz, pids=union(1, pid))
    else
        S = Array(T, sz)
    end
    S
end

monitor_field(v::SharedArray) = SharedArray(eltype(v), size(v), pids=procs(v))
monitor_field(v::AbstractArray) = similar(v)
monitor_field{A}(d::ArrayDecl{A}) = A(d.arraysize)
monitor_field(v) = v

end
