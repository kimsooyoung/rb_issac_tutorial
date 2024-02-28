import h5py
import numpy as np
import pylab as plt

file_name = "./franka_nuts_basic.hdf5"
# file_name = "./franka_bolts_nuts_table.hdf5"

with h5py.File(file_name, 'r') as f:

    print(f.keys())
    print(f"keys: {f['isaac_dataset'].keys()}")

    print(f"sim_time: {f['isaac_dataset']['sim_time'].shape}")
    print(f"joint_positions: {f['isaac_dataset']['joint_positions'].shape}")
    print(f"joint_velocities: {f['isaac_dataset']['joint_velocities'].shape}")
    print(f"camera_images: {f['isaac_dataset']['camera_images'].keys()}")

    print(f"camera_images: {f['isaac_dataset']['camera_images']['hand_camera'].shape}")
    print(f"camera_images: {f['isaac_dataset']['camera_images']['top_camera'].shape}")
    print(f"camera_images: {f['isaac_dataset']['camera_images']['front_camera'].shape}")

    sim_time = f['isaac_dataset']['sim_time'][:]
    joint_positions = f['isaac_dataset']['joint_positions'][:]
    joint_velocities = f['isaac_dataset']['joint_velocities'][:]

    hand_camera = f['isaac_dataset']['camera_images']['hand_camera'][:]
    top_camera = f['isaac_dataset']['camera_images']['top_camera'][:]
    front_camera = f['isaac_dataset']['camera_images']['front_camera'][:]

    print(f"sim_time: {sim_time}")
    print(f"joint_positions[0]: {joint_positions[0]}")
    print(f"joint_velocities[0]: {joint_velocities[0]}")

    plt.figure(1)
    plt.imshow(hand_camera[7])

    plt.figure(2)
    plt.imshow(top_camera[7])

    plt.figure(3)
    plt.imshow(front_camera[7])

    plt.show()