import pytest # noqa
from scatter import runTrajectory
import xarray as xr


def test_runTrajectory():
    ds = runTrajectory(
            seed=0,
            trajID=0,
            v=0,
            J=0,
            epsilon=1.0,
            bmax=5.,
            returnTraj=False
            )
    cmp2 = ['x', 'y', 'z']
    cmp1 = ['i', 'f']

    ds_record = xr.Dataset(
        data_vars={
            'r': (
                ['cmp1', 'cmp2'],
                [[-1.65346532, -0.11732974, 0.44433845],
                    [-1.31575804, 0.61672778, -1.45370297]]
                ),
            'p': (
                ['cmp1', 'cmp2'],
                [[0., 0., 0.],
                    [1.41892108, -0.07366292, 0.14437914]]
                ),
            'R': (
                ['cmp1', 'cmp2'],
                [[4.59462145, 0., 68.55903718],
                    [-15.68597681, 0.26758156, -78.45345705]]
                ),
            'P': (
                ['cmp1', 'cmp2'],
                [[0., 0., -49.47932462],
                    [-12.50342207, 0.16368249, -47.92339089]]
                ),
            'tf': 3635.07594726,
            'maxErr': 4.39622677e-06,
            'countstep': 321,
            'maxstep': 42.79023149,
            'converge': True,
            'fail': False,
            'l': 2,
            'n': 67,
            'case': 3,
        },
        coords={
            'cmp2': cmp2,
            'cmp1': cmp1,
        }
    )

    xr.testing.assert_allclose(ds, ds_record, rtol=1e-3)
