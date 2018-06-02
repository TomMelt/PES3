import xyzpy
import scatter as sc

r = xyzpy.Runner(
        sc.runTrajectory,
        var_names=None,
        )

h = xyzpy.Harvester(r, data_name='data.h5')

N = 10
for i in range(N):
    c = h.Crop(name='run'+str(i), batchsize=20)
    c.reap()
