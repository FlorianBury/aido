import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import Callable

import b2luigi

CUDA_FATAL_ERRORS = (
    "Cannot re-initialize CUDA",
    "CUDA error: initialization error",
    "driver shutting down",
)


class AIDOTask(b2luigi.Task):
    """ Shallow wrapper around b2luigi.Task
    """

    @property
    def htcondor_settings(self):
        return {
            "request_cpus": "1",
            "getenv": "true",
        }


def torch_safe_wrapper(
    func: Callable,
    *args,
    **kwargs,
):
    """
    b2luigi safe wrapper for calls to a torch function. Otherwise torch will raise
    'RuntimeError: Cannot re-initialize CUDA in forked subprocess.
    To use CUDA with multiprocessing, you must use the 'spawn' start method'

    We avoid this by first calling the function 'func' and excepting any errors. If
    that Error is raised by CUDA, we catch it and call that function again but inside
    a subprocess. If any further errors are raised afterwards, they are return. In
    case of no errors, we return the result of the function.
    """
    CUDA_FATAL_ERRORS = (
        "Cannot re-initialize CUDA",
        "CUDA error: initialization error",
        "driver shutting down",
    )
    try:
        return func(*args, **kwargs)
    except RuntimeError as e:
        msg = str(e)

        # Not a fatal CUDA error → propagate
        if not any(err in msg for err in CUDA_FATAL_ERRORS):
            raise

        # Fatal CUDA state → must run in fresh process
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            mp_context=ctx,
            max_workers=1,
        ) as executor:
            future = executor.submit(func, *args, **kwargs)
            return future.result()
