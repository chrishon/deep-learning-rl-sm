import os
import urllib.request
from pathlib import Path


mujoco_v2 = "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/"
datapath = "deep_learning_rl_sm\\benchmarks\\data\\"
def download_dataset_from_url(env_bm):
    dataset_url = mujoco_v2+env_bm+".hdf5"
    dataset_filepath = datapath+env_bm+".hdf5"
    if not os.path.exists(dataset_filepath):
        print('Downloading dataset:', dataset_url, 'to', dataset_filepath)
        urllib.request.urlretrieve(dataset_url, dataset_filepath)
    if not os.path.exists(dataset_filepath):
        raise IOError("Failed to download dataset from %s" % dataset_url)
    return dataset_filepath

ENV_BENCHMARKS = [
    'halfcheetah_medium-v2',
    'halfcheetah_medium_replay-v2',
    'halfcheetah_medium_expert-v2',
    'walker2d_medium-v2',
    'walker2d_medium_replay-v2',
    'walker2d_medium_expert-v2',
    'hopper_medium-v2',
    'hopper_medium_replay-v2',
    'hopper_medium_expert-v2' 
]
REF_MIN_SCORE = {
    'halfcheetah_medium-v2' : -280.178953 ,
    'halfcheetah_medium_replay-v2' : -280.178953 ,
    'halfcheetah_medium_expert-v2' : -280.178953 ,
    'walker2d_medium-v2' : 1.629008 ,
    'walker2d_medium_replay-v2' : 1.629008 ,
    'walker2d_medium_expert-v2' : 1.629008 ,
    'hopper_medium-v2' : -20.272305 ,
    'hopper_medium_replay-v2' : -20.272305 ,
    'hopper_medium_expert-v2' : -20.272305 
}
REF_MAX_SCORE = {
    'halfcheetah_medium-v2' : 12135.0 ,
    'halfcheetah_medium_replay-v2' : 12135.0 ,
    'halfcheetah_medium_expert-v2' : 12135.0 ,
    'walker2d_medium-v2' : 4592.3 ,
    'walker2d_medium_replay-v2' : 4592.3 ,
    'walker2d_medium_expert-v2' : 4592.3 ,
    'hopper_medium-v2' : 3234.3 ,
    'hopper_medium_replay-v2' : 3234.3 ,
    'hopper_medium_expert-v2' : 3234.3 ,
}

if __name__ == "__main__":
    
    if not os.path.exists(datapath):
        os.makedirs(datapath)
    for env_bm in ENV_BENCHMARKS:
        download_dataset_from_url(env_bm)
