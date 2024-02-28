import h5py
import numpy as np
import pylab as plt

file_name = "./ur_bin_filling.hdf5"
# file_name = "./ur_bin_palleting.hdf5"

with h5py.File(file_name, 'r') as f:

    print(f.keys())
    print(f"keys: {f['isaac_dataset'].keys()}")

    print(f"sim_time: {f['isaac_dataset']['sim_time'].shape}")
    print(f"joint_positions: {f['isaac_dataset']['joint_positions'].shape}")
    print(f"joint_velocities: {f['isaac_dataset']['joint_velocities'].shape}")
    print(f"camera_images: {f['isaac_dataset']['camera_images'].keys()}")

    sim_time = f['isaac_dataset']['sim_time'][:]
    joint_positions = f['isaac_dataset']['joint_positions'][:]
    joint_velocities = f['isaac_dataset']['joint_velocities'][:]

    if file_name == "./ur_bin_filling.hdf5":

        ee_camera = f['isaac_dataset']['camera_images']['ee_camera'][:]
        side_camera = f['isaac_dataset']['camera_images']['side_camera'][:]
        front_camera = f['isaac_dataset']['camera_images']['front_camera'][:]

        plt.figure(1)
        plt.imshow(ee_camera[7])

        plt.figure(2)
        plt.imshow(side_camera[7])

        plt.figure(3)
        plt.imshow(front_camera[7])

    elif file_name == "./ur_bin_palleting.hdf5":

        ee_camera = f['isaac_dataset']['camera_images']['ee_camera'][:]
        left_camera = f['isaac_dataset']['camera_images']['left_camera'][:]
        right_camera = f['isaac_dataset']['camera_images']['right_camera'][:]
        front_camera = f['isaac_dataset']['camera_images']['front_camera'][:]
        back_camera = f['isaac_dataset']['camera_images']['back_camera'][:]

        plt.figure(1)
        plt.imshow(ee_camera[1])

        plt.figure(2)
        plt.imshow(left_camera[1])

        plt.figure(3)
        plt.imshow(right_camera[1])

        plt.figure(4)
        plt.imshow(front_camera[1])

        plt.figure(5)
        plt.imshow(back_camera[1])


    print(f"sim_time: {sim_time}")
    print(f"joint_positions[0]: {joint_positions[0]}")
    print(f"joint_velocities[0]: {joint_velocities[0]}")

    plt.show()