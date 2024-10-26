import numpy as np 
import pickle as pkl
import os
import concurrent
from data.utils import pad_axis

from data.constants import N_MAX_ATOMS
from data.constants import moledit_data_column_dtypes, mask_types, feature_all_type
from data.utils import pad_axis, one_hot

def read_file(file_path):
    with open(file_path, 'rb') as f:
        return pkl.load(f)
    
def read_files_in_parallel(file_paths, num_parallel_worker=32):
    # time0 = datetime.datetime.now()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_parallel_worker) as executor:
        # Map the read_file function to each file path
        results = list(executor.map(read_file, file_paths))
    return results