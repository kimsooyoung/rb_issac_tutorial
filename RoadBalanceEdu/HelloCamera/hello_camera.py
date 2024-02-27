# Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from omni.isaac.examples.base_sample import BaseSample
import numpy as np

# Note: checkout the required tutorials at https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.sensor import Camera
from omni.isaac.core import World

import omni.isaac.core.utils.numpy.rotations as rot_utils

from PIL import Image
import cv2

class HelloCamera(BaseSample):
    def __init__(self) -> None:
        super().__init__()

        self.view_count = 0

        return

    def setup_scene(self):

        world = self.get_world()
        world.scene.add_default_ground_plane()
        
        self._camera = Camera(
            prim_path="/World/camera",
            position=np.array([0.0, 0.0, 25.0]),
            frequency=30,
            resolution=(640, 480),
            orientation=rot_utils.euler_angles_to_quats(np.array([0, 90, 0]), degrees=True),
        )
        self._camera.initialize()
        self._camera.add_motion_vectors_to_frame()

        fancy_cube = world.scene.add(
            DynamicCuboid(
                prim_path="/World/random_cube", # The prim path of the cube in the USD stage
                name="fancy_cube", # The unique name used to retrieve the object from the scene later on
                position=np.array([0, 0, 1.0]), # Using the current stage units which is in meters by default.
                scale=np.array([0.5015, 0.5015, 0.5015]), # most arguments accept mainly numpy arrays.
                color=np.array([0, 0, 1.0]), # RGB channels, going from 0-1
            ))
        return

    async def setup_post_load(self):

        world = self.get_world()
        self._camera.add_motion_vectors_to_frame()
        self._world.add_physics_callback("sim_step", callback_fn=self.physics_callback) #callback names have to be unique
        cv2.namedWindow('image')

        return

    def physics_callback(self, step_size):

        if self.view_count == 100:

            self._camera.get_current_frame()
            img = Image.fromarray(self._camera.get_rgba()[:, :, :3])
            img.save("./camera_opencv.png")
            
            # PIL image to cv2
            cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            # image show
            cv2.imshow("image", cv_img)
            if cv2.waitKey(0) == 27:
                pass

            self.view_count = 0

        self.view_count += 1

    # async def setup_pre_reset(self):
    #     return

    # async def setup_post_reset(self):
    #     return

    def world_cleanup(self):
        cv2.destroyAllWindows()
        return
