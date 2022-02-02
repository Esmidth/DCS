from multiprocessing import shared_memory
from multiprocessing.managers import SharedMemoryManager
import pickle
import tracemalloc
import numpy as np
import pandas as pd

import multiprocessing as mp
from multiprocessing import Manager, Pool

import time


def compute():
    pass



if __name__ == "__main__":
    manager = mp.Manager()
    aggreateData = manager.dict()

    start = time.time()

    p = Pool()

    p.apply_async(func=compute,args=())
