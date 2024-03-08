# Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from omni.isaac.examples.base_sample import BaseSample
from datetime import datetime
import numpy as np

# Note: checkout the required tutorials at https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.sensor import Camera
from omni.isaac.core import World

import omni.isaac.core.utils.numpy.rotations as rot_utils
from omni.isaac.core import SimulationContext

from PIL import Image
import carb
import h5py
import omni
import cv2

class HelloCamera(BaseSample):
    def __init__(self) -> None:
        super().__init__()

        self._save_count = 0

        self._f = None
        self._time_list = []
        self._img_list = []

        return

    def setup_camera(self):

        self._camera = Camera(
            prim_path="/World/camera",
            position=np.array([0.0, 0.0, 25.0]),
            frequency=30,
            resolution=(640, 480),
            orientation=rot_utils.euler_angles_to_quats(np.array([0, 90, 0]), degrees=True),
        )
        self._camera.initialize()
        self._camera.add_motion_vectors_to_frame()

    def setup_cube(self, prim_path, name, position, scale, color):
            
        self._fancy_cube = self._world.scene.add(
            DynamicCuboid(
                prim_path=prim_path,
                name=name,
                position=position,
                scale=scale,
                color=color,
            ))

    def setup_dataset(self):

        now = datetime.now() # current date and time
        date_time_str = now.strftime("%m_%d_%Y_%H_%M_%S")

        file_name = f'hello_cam_{date_time_str}.hdf5'
        print(file_name)

        self._f = h5py.File(file_name,'w')
        self._group = self._f.create_group("isaac_save_data")

    def setup_scene(self):
        
        print("setup_scene")

        world = self.get_world()
        world.scene.add_default_ground_plane()
        
        self.setup_camera()
        self.setup_cube(
            prim_path="/World/random_cube",
            name="fancy_cube",
            position=np.array([0, 0, 1.0]),
            scale=np.array([0.5015, 0.5015, 0.5015]),
            color=np.array([0, 0, 1.0]),
        )

        self.setup_dataset()
        
        self.simulation_context = SimulationContext()

    async def setup_post_load(self):
        
        print("setup_post_load")

        world = self.get_world()
        self._camera.add_motion_vectors_to_frame()
        self._world.add_physics_callback("sim_step", callback_fn=self.physics_callback) #callback names have to be unique

        return

    def physics_callback(self, step_size):

        self._camera.get_current_frame()

        if self._save_count % 100 == 0:

            current_time = self.simulation_context.current_time

            self._time_list.append(current_time)
            self._img_list.append(self._camera.get_rgba()[:, :, :3])

            print("Data Collected")

        self._save_count += 1

        if self._save_count == 500:
            self.world_cleanup()

    async def setup_pre_reset(self):

        if self._f is not None:
            self._f.close()
            self._f = None
        elif self._f is None:
            print("Create new file for new data collection...")
            self.setup_dataset()

        self._save_count = 0

        return

    # async def setup_post_reset(self):
    #     return

    def world_cleanup(self):

        if self._f is not None:
            self._group.create_dataset(f"sim_time", data=self._time_list, compression='gzip', compression_opts=9)
            self._group.create_dataset(f"image", data=self._img_list, compression='gzip', compression_opts=9)
            self._f.close()
            print("Data saved")
        elif self._f is None:
            print("Invalid Operation Data not saved")

        self._f = None
        self._save_count = 0

        world = self.get_world()
        world.pause()

        return
