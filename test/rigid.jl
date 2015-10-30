using Images, AffineTransforms, TestImages
using BlockRegistration
using BlockRegistrationScheduler, RegisterDriver, RegisterWorkerShell, RegisterWorkerRigid
using Base.Test

#2d
fixed = testimage("cameraman")
tfm = tformrotate(pi/12)
moving = transform(fixed, tfm)

alg = Rigid(fixed, pat=false, print_level=5)
mon = monitor(alg, ())
mon[:tform] = nothing
mon[:mismatch] = 0.0
mon = driver(alg, moving, mon)
ptfm = mon[:tform]*tfm
@test_approx_eq_eps ptfm.scalefwd eye(2) 0.01
@test all(abs(ptfm.offset) .< 0.2)
@test mon[:mismatch] < 0.001


#3d
fixed = testimage("cameraman")
fixed0 = deepcopy(fixed)
fixed = zeros(size(fixed)...,20)
for i = 1:20
    fixed[:,:,i] = data(fixed0)
end
fixed = copyproperties(fixed0, fixed)
fixed["spatialorder"] = ["x"; "l"; "z"]
tfm = tformrotate([0;0;1;], pi/12)
moving = transform(fixed, tfm)

alg = Rigid(fixed, pat=false, print_level=5)
mon = monitor(alg, ())
mon[:tform] = nothing
mon[:mismatch] = 0.0
mon = driver(alg, moving, mon)
ptfm = mon[:tform]*tfm
@test_approx_eq_eps ptfm.scalefwd eye(3) 0.01
@test all(abs(ptfm.offset) .< 0.2)
@test mon[:mismatch] < 0.001
