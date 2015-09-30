module RegisterWorkerShell

export AbstractWorker, maybe_sharedarray, monitor, monitor!, worker, workerpid

abstract AbstractWorker

# Monitoring is how you pass data back to the driver
# A Dict specifies which fields of the AbstractWorker type you're monitoring.
# You can also monitor additional variables in the worker,
# as long as the worker is set up to look for them in the Dict.
function monitor{N}(alg::AbstractWorker, fields::Union{NTuple{N,Symbol},Vector{Symbol}})
    mon = Dict{Symbol,Any}()
    for f in fields
        isdefined(alg, f) || continue
        mon[f] = monitor_field(getfield(alg, f))
    end
    mon
end

function monitor!(mon::Dict{Symbol,Any}, info::AbstractWorker)
    for f in fieldnames(info)
        monitor!(mon, f, getfield(info, f))
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

worker(args...) = error("Worker modules must define `worker`")
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
monitor_field(v) = v

end
