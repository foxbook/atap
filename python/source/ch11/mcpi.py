
import time
import random
import multiprocessing as mp

from functools import wraps


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        return result, time.time() - start
    return wrapper


def mcpi_samples(n):
    """
    Compute the number of points in the unit circle out of n points.
    """
    count = 0
    for i in range(n):
        x, y = random.random(), random.random()
        if x*x + y*y <= 1:
            count += 1
    return count


@timeit
def mcpi_sequential(N):
    count = mcpi_samples(N)
    return count / N * 4


@timeit
def mcpi_parallel(N):
    procs = mp.cpu_count()
    pool  = mp.Pool(processes=procs)

    parts = [int(N/procs)] * procs
    count = sum(pool.map(mcpi_samples, parts))
    return count / N * 4


if __name__ == '__main__':
    N = 10000000
    pi, delta = mcpi_sequential(N)
    print("sequential pi: {} in {:0.2f} seconds".format(pi, delta))

    pi, delta = mcpi_parallel(N)
    print("parallel pi: {} in {:0.2f} seconds".format(pi, delta))
