from __future__ import annotations

import asyncio


def run_async(coro):
    """Run *coro* whether or not an event loop is already running.

    - Outside a loop (normal CLI / sync usage): ``asyncio.run()``.
    - Inside a running loop (pytest-asyncio, Jupyter): run in a new thread so
      the coroutine gets its own fresh event loop without deadlocking the outer one.

    Returns the coroutine's return value in both cases.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()  # propagate exceptions + return value
    else:
        return asyncio.run(coro)
