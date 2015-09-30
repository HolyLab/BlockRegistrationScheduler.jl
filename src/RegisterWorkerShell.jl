module RegisterWorkerShell

export AbstractWorker, worker, workerpid

abstract AbstractWorker

worker(args...) = error("Worker modules must define `worker`")
workerpid(w::AbstractWorker) = w.workerpid

end
