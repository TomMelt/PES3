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
                [[-1.65346532, -0.11732974,  0.44433845],
                    [-1.30722921,  0.61629506, -1.45248087]]
                ),
            'p': (
                ['cmp1', 'cmp2'],
                [[0., 0., 0.],
                    [1.43799995, -0.08244679,  0.16490921]]
                ),
            'R': (
                ['cmp1', 'cmp2'],
                [[4.59462145, 0., 68.55903718],
                    [-15.74318462, 0.26834937, -78.67333335]]
                ),
            'P': (
                ['cmp1', 'cmp2'],
                [[0., 0., -49.49561484],
                    [-12.50732309,   0.16374558, -47.93859119]]
                ),
            'tf': 3641.880656,
            'maxErr': 0.611788,
            'countstep': 326,
            'maxstep': 40.018579,
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

    xr.testing.assert_allclose(ds, ds_record, rtol=1e-5)
