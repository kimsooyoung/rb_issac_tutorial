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
from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units
from omni.isaac.core.utils.nucleus import get_assets_root_path, get_url_root
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.sensor import Camera
from omni.isaac.core import World

import omni.isaac.core.utils.numpy.rotations as rot_utils
from omni.isaac.core import SimulationContext
from omni.physx.scripts import deformableUtils, physicsUtils  
from pxr import UsdGeom, Gf, UsdPhysics

from PIL import Image
import carb
import h5py
import omni
import cv2

class HelloCamera(BaseSample):
    def __init__(self) -> None:
        super().__init__()

        self._save_count = 0

        self._time_list = []
        self._img_list = []

        return

    def setup_scene(self):

        world = self.get_world()
        stage = omni.usd.get_context().get_stage()
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

        self.simulation_context = SimulationContext()

        # self._f = h5py.File('hello_cam.hdf5','w')
        # self._group = self._f.create_group("isaac_save_data")

        carb.log_info("Check /persistent/isaac/asset_root/default setting")
        default_asset_root = carb.settings.get_settings().get("/persistent/isaac/asset_root/default")
        server_root = get_url_root(default_asset_root)
        print(f"server_root1: {server_root}")

        foot_usd_path = server_root + "/Projects/DInsight/Female_Foot.usd"
        # TODO : check foot availability
        add_reference_to_stage(
            usd_path=foot_usd_path, 
            prim_path=f"/World/foot",
        )

        foot_mesh = UsdGeom.Mesh.Get(stage, "/World/foot")
        physicsUtils.set_or_add_translate_op(foot_mesh, translate=Gf.Vec3f(0.0, 0.0, 0.5))
        physicsUtils.set_or_add_orient_op(foot_mesh, orient=Gf.Quatf(-0.5, -0.5, -0.5, -0.5))
        physicsUtils.set_or_add_scale_op(foot_mesh, scale=Gf.Vec3f(0.001, 0.001, 0.001))

        return

    async def setup_post_load(self):

        world = self.get_world()
        self._camera.add_motion_vectors_to_frame()
        self._world.add_physics_callback("sim_step", callback_fn=self.physics_callback) #callback names have to be unique
        cv2.namedWindow('image')

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

    # async def setup_pre_reset(self):
    #     return

    # async def setup_post_reset(self):
    #     return

    def world_cleanup(self):
        cv2.destroyAllWindows()

        # self._group.create_dataset(f"sim_time", data=self._time_list, compression='gzip', compression_opts=9)
        # self._group.create_dataset(f"image", data=self._img_list, compression='gzip', compression_opts=9)

        # self._f.close()
        print("Data saved")

        self._save_count = 0
        world = self.get_world()
        world.pause()

        return
