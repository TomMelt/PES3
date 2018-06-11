import xyzpy
import scatter as sc

N = 28
for i in range(N):
    c = xyzpy.Crop(name='run'+str(i), batchsize=40)
    c.reap()
