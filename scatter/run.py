import xyzpy
import scatter as sc
import numpy as np

r = xyzpy.Runner(
    sc.runTrajectory,
    var_names=None,
)

h = xyzpy.Harvester(r, data_name='data.h5')

N = 28
epsilon = np.round(np.linspace(0.1, 2.7, num=N), 2)
bmax = np.round(np.linspace(10, 3, num=N), 1)

combo = []

for i in range(N):
    combos = [
        ('seed', [0]),
        ('J', [0]),
        ('v', [0]),
        ('epsilon', [epsilon[i]]),
        ('trajID', range(4000)),
        ('bmax', [bmax[i]]),
    ]

    c = h.Crop(name='run'+str(i), batchsize=40)

    c.sow_combos(combos)

    c.qsub_grow(
            minutes=15,
            gigabytes=1,
            launcher="/home/ucaptme/miniconda3/bin/python3"
            )
