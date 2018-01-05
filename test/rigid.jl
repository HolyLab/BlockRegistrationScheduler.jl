using Images, AffineTransforms, TestImages
using BlockRegistration
using BlockRegistrationScheduler, RegisterDriver, RegisterWorkerShell, RegisterWorkerRigid
using Base.Test

#Rigid
fixed = testimage("cameraman")
tfm = tformrotate(pi/12)
moving = transform(fixed, tfm)

alg = Rigid(fixed, pat=false, print_level=5)
mon = monitor(alg, ())
mon[:tform] = nothing
mon[:mismatch] = 0.0
mon = driver(alg, moving, mon)
ptfm = mon[:tform]*tfm
@test ≈(ptfm.scalefwd, eye(2), atol=0.01)
@test all(abs.(ptfm.offset) .< 0.2)
@test mon[:mismatch] < 0.001

#RigidGridStart
fixed = testimage("cameraman")
tfm = tformrotate(pi/10)
moving = transform(fixed, tfm)
moving[find(x->isnan(x), moving)] = zero(eltype(moving))

maxradians = pi/10
rgridsz = 3
mxshift = (3,3)
alg = RigidGridStart(fixed, maxradians, rgridsz, mxshift; print_level=5)
#alg = RigidGridStart(fixed, maxradians, 1, mxshift; print_level=5)
mon = monitor(alg, ())
mon[:tform] = nothing
mon[:mismatch] = 0.0
mon = driver(alg, moving, mon)
ptfm = mon[:tform]*tfm
@test ≈(ptfm.scalefwd, eye(2), atol=0.01)
@test all(abs.(ptfm.offset) .< 0.2)
@test mon[:mismatch] < 0.001
