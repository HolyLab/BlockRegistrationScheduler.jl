### Dummy algorithms to test features of `driver`
module WorkerDummy

using RegisterWorkerShell
import RegisterWorkerShell: worker

export Alg1, Alg2, Alg3, monitor, worker, workerpid

# Dispatch on the algorithm used to perform registration
# Each algorithm has a container it uses for storage and communication
# with the driver process
abstract Alg <: AbstractWorker

type Alg1{A<:AbstractArray} <: Alg
    fixed::A
    λ::Float64
    workerpid::Int
end
function Alg1(fixed, λ; pid=1)
    Alg1(maybe_shared(fixed, pid), λ, pid)
end

type Alg2{A<:AbstractArray,V<:AbstractVector,M<:AbstractMatrix} <: Alg
    fixed::A
    tform::V
    u0::M
    workerpid::Int
end
function Alg2{T}(fixed, ::Type{T}, sz; pid=1)
    Alg2(maybe_shared(fixed, pid), warray(T, 12, pid), warray(T, sz, pid), pid)
end

type Alg3 <: Alg
    string::ASCIIString
    workerpid::Int
end
function Alg3(s::ASCIIString; pid=1)
    Alg3(s, pid)
end

# Here are the "registration algorithms"
function worker(info::Alg1, moving, tindex, mon)
    info.λ = tindex
    monitor!(mon, info)   # just dump output
end

function worker(info::Alg2, moving, tindex, mon)
    # Do stuff to set tform
    tform = linspace(1,12,12)+tindex
    monitor_copy!(mon, :tform, tform)
    # Do more computations...
    monitor_copy!(mon, :u0, zeros(size(info.u0))-tindex)
end

function worker(info::Alg3, moving, tindex, mon)
    monitor!(mon, info)
    if haskey(mon, :extra)
        mon[:extra] = "world"
    end
end


## Utility functions
function maybe_shared(A, pid=1)
    if pid != 1
        S = SharedArray(eltype(A), size(A), pids=union(1, pid))
        copy!(S, A)
    else
        S = A
    end
    S
end

function warray{T}(::Type{T}, sz, pid=1)
    if pid != 1
        S = SharedArray(T, sz, pids=union(1, pid))
    else
        S = Array(T, sz)
    end
    S
end

# Monitoring is how you pass data back to the driver
# A Dict specifies which fields of the Alg type you're monitoring.
# You can also monitor additional variables in the worker,
# as long as the worker is set up to look for them in the Dict.
monitor_field(v::SharedArray) = SharedArray(eltype(v), size(v), pids=procs(v))
monitor_field(v::AbstractArray) = similar(v)
monitor_field(v) = v

function monitor{N}(alg::Alg, fields::Union{NTuple{N,Symbol},Vector{Symbol}})
    mon = Dict{Symbol,Any}()
    for f in fields
        isdefined(alg, f) || continue
        mon[f] = monitor_field(getfield(alg, f))
    end
    mon
end

function monitor_copy!(mon, fn, v::AbstractArray)
    if haskey(mon, fn)
        copy!(mon[fn], v)
    end
    mon
end
function monitor_copy!(mon, fn, v)
    if haskey(mon, fn)
        mon[fn] = v
    end
    mon
end

function monitor!(mon::Dict{Symbol,Any}, info::Alg)
    for f in fieldnames(info)
        monitor_copy!(mon, f, getfield(info, f))
    end
    mon
end

workerpid(alg::Alg) = alg.workerpid

end  # module
