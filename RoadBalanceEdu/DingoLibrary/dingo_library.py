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
import omni.graph.core as og

import omni.isaac.core.utils.numpy.rotations as rot_utils
from omni.isaac.core import SimulationContext
from omni.physx.scripts import deformableUtils, physicsUtils  
from pxr import UsdGeom, Gf, UsdPhysics, Sdf, Gf, Tf, UsdLux

from PIL import Image
import carb
import h5py
import omni
import cv2

class DingoLibrary(BaseSample):
    def __init__(self) -> None:
        super().__init__()

        carb.log_info("Check /persistent/isaac/asset_root/default setting")
        default_asset_root = carb.settings.get_settings().get("/persistent/isaac/asset_root/default")
        self._server_root = get_url_root(default_asset_root)

        return

    def og_setup(self):
        camprim1 = "/World/Orbbec_Gemini2/Orbbec_Gemini2/camera_ir_left/camera_left/Stream_depth"
        camprim2 = "/World/Orbbec_Gemini2/Orbbec_Gemini2/camera_rgb/camera_rgb/Stream_rgb"
        
        try:
            og.Controller.edit(
                {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("RenderProduct1", "omni.isaac.core_nodes.IsaacCreateRenderProduct"),
                        ("RenderProduct2", "omni.isaac.core_nodes.IsaacCreateRenderProduct"),
                        ("RGBPublish", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
                        ("DepthPublish", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
                        ("CameraInfoPublish", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
                    ],
                    og.Controller.Keys.SET_VALUES: [
                        ("RenderProduct1.inputs:cameraPrim", camprim1),
                        ("RenderProduct2.inputs:cameraPrim", camprim2),

                        ("RGBPublish.inputs:topicName", "rgb"),
                        ("RGBPublish.inputs:type", "rgb"),
                        ("RGBPublish.inputs:resetSimulationTimeOnStop", True),

                        ("DepthPublish.inputs:topicName", "depth"),
                        ("DepthPublish.inputs:type", "depth"),
                        ("DepthPublish.inputs:resetSimulationTimeOnStop", True),
                        
                        ("CameraInfoPublish.inputs:topicName", "depth_camera_info"),
                        ("CameraInfoPublish.inputs:type", "camera_info"),
                        ("CameraInfoPublish.inputs:resetSimulationTimeOnStop", True),
                    ],
                    og.Controller.Keys.CONNECT: [
                        ("OnPlaybackTick.outputs:tick", "RenderProduct1.inputs:execIn"),
                        ("OnPlaybackTick.outputs:tick", "RenderProduct2.inputs:execIn"),

                        ("RenderProduct1.outputs:execOut", "DepthPublish.inputs:execIn"),
                        ("RenderProduct1.outputs:execOut", "CameraInfoPublish.inputs:execIn"),
                        ("RenderProduct2.outputs:execOut", "RGBPublish.inputs:execIn"),                    
                    
                        ("RenderProduct1.outputs:renderProductPath", "DepthPublish.inputs:renderProductPath"),
                        ("RenderProduct1.outputs:renderProductPath", "CameraInfoPublish.inputs:renderProductPath"),
                        ("RenderProduct2.outputs:renderProductPath", "RGBPublish.inputs:renderProductPath"),
                    ],
                },
            )
        except Exception as e:
            print(e)

    def camera_setup(self):
        
        gemini_usd_path = self._server_root + "/NVIDIA/Assets/Isaac/2023.1.1/Isaac/Sensors/Orbbec/Gemini 2/orbbec_gemini2_V1.0.usd"

        # TODO : check foot availability
        add_reference_to_stage(
            usd_path=gemini_usd_path, 
            prim_path=f"/World/Orbbec_Gemini2",
        )

        self._gemini_mesh = UsdGeom.Mesh.Get(self._stage, "/World/Orbbec_Gemini2")
        physicsUtils.set_or_add_translate_op(self._gemini_mesh, translate=Gf.Vec3f(0.05, 0.5, 0.5))
        # x: -3.1415927, y: 0, z: -1.5707963 
        rot = rot_utils.euler_angles_to_quats(np.array([
                90, 0, 0
            ]), degrees=True)
        physicsUtils.set_or_add_orient_op(self._gemini_mesh, orient=Gf.Quatf(*rot))
        # physicsUtils.set_or_add_scale_op(gemini_mesh, scale=Gf.Vec3f(0.001, 0.001, 0.001))
        ldm_light = self._stage.GetPrimAtPath("/World/Orbbec_Gemini2/Orbbec_Gemini2/camera_ldm/camera_ldm/RectLight")
        ldm_light_intensity = ldm_light.GetAttribute("intensity")
        ldm_light_intensity.Set(0)

    def add_background(self):

        bg_path = self._server_root + "/Projects/RBROS2/LibraryNoRoof/Library_No_Roof_Collide_Light.usd"
        
        add_reference_to_stage(
            usd_path=bg_path, 
            prim_path=f"/World/Library_No_Roof",
        )

        bg_mesh = UsdGeom.Mesh.Get(self._stage, "/World/Library_No_Roof")
        # physicsUtils.set_or_add_translate_op(bg_mesh, translate=Gf.Vec3f(0.0, 0.0, 0.0))
        # physicsUtils.set_or_add_orient_op(bg_mesh, orient=Gf.Quatf(-0.5, -0.5, -0.5, -0.5))
        physicsUtils.set_or_add_scale_op(bg_mesh, scale=Gf.Vec3f(0.01, 0.01, 0.01))


    def add_dingo(self):
        
        dingo_usd_path = self._server_root + "/NVIDIA/Assets/Isaac/2023.1.1/Isaac/Robots/Clearpath/Dingo/dingo.usd"
        
        add_reference_to_stage(
            usd_path=dingo_usd_path, 
            prim_path=f"/World/dingo",
        )

        dingo_mesh = UsdGeom.Mesh.Get(self._stage, "/World/dingo")
        physicsUtils.set_or_add_translate_op(dingo_mesh, translate=Gf.Vec3f(0.0, 0.0, 0.02))
        # physicsUtils.set_or_add_orient_op(dingo_mesh, orient=Gf.Quatf(-0.5, -0.5, -0.5, -0.5))
        # physicsUtils.set_or_add_scale_op(dingo_mesh, scale=Gf.Vec3f(0.001, 0.001, 0.001))

    def add_light(self):
        sphereLight1 = UsdLux.SphereLight.Define(self._stage, Sdf.Path("/World/SphereLight1"))
        sphereLight1.CreateIntensityAttr(100000)
        sphereLight1.CreateRadiusAttr(100.0)
        sphereLight1.AddTranslateOp().Set(Gf.Vec3f(885.0, 657.0, 226.0))

    def setup_scene(self):

        self._world = self.get_world()
        self._stage = omni.usd.get_context().get_stage()
        self.simulation_context = SimulationContext()
        
        self.add_background()
        self.add_dingo()
        
        return

    async def setup_post_load(self):

        self._world.add_physics_callback("sim_step", callback_fn=self.physics_callback) #callback names have to be unique

        return

    def physics_callback(self, step_size):

        # self._camera.get_current_frame()

        # if self._save_count % 11 == 0:

        #     current_time = self.simulation_context.current_time

        #     omega = 2 * np.pi * self._rotate_count / 100
        #     x_offset, y_offset = 0.5 * np.cos(omega), 0.5 * np.sin(omega)

        #     physicsUtils.set_or_add_translate_op(self._gemini_mesh, translate=Gf.Vec3f(
        #             0.05, x_offset, 0.5 + y_offset))

        #     rot = rot_utils.euler_angles_to_quats(
        #             np.array([
        #                 90 + 360 / 100 * self._rotate_count, 0, 0
        #             ]), degrees=True)
        #     print(f"rot: {rot}")

        #     physicsUtils.set_or_add_orient_op(self._gemini_mesh, 
        #         orient=Gf.Quatf(*rot))

        #     self._time_list.append(current_time)
        #     # self._img_list.append(self._camera.get_rgba()[:, :, :3])

        #     print("Data Collected")
     
        #     self._rotate_count += 1

        # self._save_count += 1

        # if self._save_count > 1100:
        #     self.world_cleanup()
        
        return 

    # async def setup_pre_reset(self):
    #     return

    # async def setup_post_reset(self):
    #     return

    def world_cleanup(self):

        self._world.pause()

        return
