import pickle
import numpy as np
import pandas as pd

import multiprocessing as mp
from multiprocessing import Manager, Pool

import time

time_max = 16354233
time_min = 0

total_id_uniq = [  3,   5,   7,   8,   9,  10,  11,  12,  13,  14,  15,  17,  18,
        19,  21,  23,  24,  25,  26,  27,  30,  31,  32,  33,  34,  35,
        36,  37,  39,  40,  41,  42,  43,  44,  45,  46,  47,  49,  50,
        51,  53,  54,  55,  56,  57,  59,  60,  61,  62,  63,  64,  65,
        66,  68,  69,  70,  71,  72,  73,  75,  76,  79,  80,  81,  82,
        84,  85,  87,  88,  89,  92,  93,  94,  95,  96,  97,  98,  99,
       100, 103, 104, 105, 106, 107, 109, 111, 121, 122]

def compute_test(interval):
    with open('time_indices_map.pkl','rb') as f:
        time_indices_map = pickle.load(f)
    
    time_indices_new = np.zeros(int(time_max/interval)+1)

    time_indices_id_map = {}


    
    for id in total_id_uniq:

        for i,item in enumerate(time_indices_new):
            if 1 in time_indices_map[id][i*interval:i*interval+interval]:
                time_indices_new[i] = 1
        
        # for i,item in enumerate(time_indices_map[id]):

        time_indices_id_map[id] = time_indices_new

    with open('time_indices_id_map_interval({}).pkl'.format(interval),'wb') as f:
        pickle.dump(time_indices_id_map,f)
    
    print("done, {}".format(interval))


if __name__ == "__main__":
    manager = mp.Manager()
    aggreateData = manager.dict()

    start = time.time()

    p = Pool(2)

    time_interval_list = np.arange(12,45)
    print(time_interval_list)


    results = []


    for interval in time_interval_list:
        result = p.apply_async(func=compute_test,args=(interval,))
        # results.append(result)
    
    p.close()
    p.join()

    # output_list = []
    
    # for result in results:
        # output_list.append(result.get())
    
    print('compute_time: ',time.time()-start)
