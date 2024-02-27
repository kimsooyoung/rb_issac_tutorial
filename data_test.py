import h5py
import numpy as np
import pylab as plt

file_name = "/home/kimsooyoung/Documents/cam_test.hdf5"

with h5py.File(file_name, 'r') as f:
    # 데이터셋 로드
    # data = f['my_dataset'][:]
    print(f.keys())
    print(f['isaac_save_data'].keys())
    print(f['isaac_save_data']['image'].shape)
    print(f['isaac_save_data']['sim_time'].shape)

    print(type(f['isaac_save_data']['image'][0]))
    print(f['isaac_save_data']['sim_time'][0])

    plt.imshow(f['isaac_save_data']['image'][0])
    plt.show()