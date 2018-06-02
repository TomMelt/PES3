import xyzpy
import scatter as sc
import numpy as np

r = xyzpy.Runner(
    sc.runTrajectory,
    var_names=None,
)

h = xyzpy.Harvester(r, data_name='data.h5')

N=10
epsilon = np.round(np.linspace(0.1, 2.0, num=N), 2)
bmax = np.linspace(10, 3, num=N)

combo = []

for i in range(N):
    combos = [
        ('seed', [0]),
        ('J', [0]),
        ('v', [0]),
        ('epsilon', [epsilon[i]]),
        ('trajID', range(1000)),
        ('bmax', [bmax[i]]),
    ]

    combo.append(combos)

    c = h.Crop(name='run'+str(i), batchsize=20)

    c.sow_combos(combo[i])

    c.qsub_grow(minutes=5, gigabytes=0.25, launcher="/home/ucaptme/miniconda3/bin/python3")
