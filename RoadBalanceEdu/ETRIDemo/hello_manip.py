# Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.franka import Franka
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.franka.controllers import PickPlaceController
import numpy as np

from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units
from omni.isaac.core.utils.prims import define_prim, get_prim_at_path
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.universal_robots import UR10

import carb

class HelloManip(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        return

    def setup_scene(self):
        world = self.get_world()
        world.scene.add_default_ground_plane()
        prim = get_prim_at_path("/World/defaultGroundPlane")

        # franka = world.scene.add(Franka(prim_path="/World/Fancy_Franka", name="fancy_franka"))
        # ur10 = world.scene.add(UR10(prim_path="/World/UR10", name="UR10"))

        asset_path = get_assets_root_path() + "/Isaac/Props/Blocks/nvidia_cube.usd"
        if asset_path is None:
            carb.log_error("Could not find Isaac Sim assets server")
        else:
            print(asset_path)
            prim.GetReferences().AddReference(asset_path)

        # world.scene.add(
        #     DynamicCuboid(
        #         prim_path="/World/random_cube",
        #         name="fancy_cube",
        #         position=np.array([0.3, 0.3, 0.3]),
        #         scale=np.array([0.0515, 0.0515, 0.0515]),
        #         color=np.array([0, 0, 1.0]),
        #     )
        # )

        # self._bin_initial_position = np.array([0.35, 0.15, -0.40]) / get_stage_units()

        # self._packing_bin = world.scene.add(
        #     RigidPrim(
        #         prim_path="/World/bin",
        #         name="packing_bin",
        #         position=self._bin_initial_position,
        #         orientation=euler_angles_to_quat(np.array([0, 0, np.pi / 2])),
        #     )
        # )

        return

    # async def setup_post_load(self):
    #     self._world = self.get_world()
    #     self._franka = self._world.scene.get_object("fancy_franka")
    #     self._fancy_cube = self._world.scene.get_object("fancy_cube")
    #     # Initialize a pick and place controller
    #     self._controller = PickPlaceController(
    #         name="pick_place_controller",
    #         gripper=self._franka.gripper,
    #         robot_articulation=self._franka,
    #     )
    #     self._world.add_physics_callback("sim_step", callback_fn=self.physics_step)
    #     # World has pause, stop, play..etc
    #     # Note: if async version exists, use it in any async function is this workflow
    #     self._franka.gripper.set_joint_positions(self._franka.gripper.joint_opened_positions)
    #     await self._world.play_async()
    #     return

    # # This function is called after Reset button is pressed
    # # Resetting anything in the world should happen here
    # async def setup_post_reset(self):
    #     self._controller.reset()
    #     self._franka.gripper.set_joint_positions(self._franka.gripper.joint_opened_positions)
    #     await self._world.play_async()
    #     return

    # def physics_step(self, step_size):
    #     # cube_position, _ = self._fancy_cube.get_world_pose()
    #     cube_position = np.array([0.3, -0.3, 0.1])
    #     # goal_position = np.array([-0.3, -0.3, 0.0515 / 2.0])
    #     goal_position = np.array([-0.3, -0.3, 0.1])
    #     current_joint_positions = self._franka.get_joint_positions()
    #     actions = self._controller.forward(
    #         picking_position=cube_position,
    #         placing_position=goal_position,
    #         current_joint_positions=current_joint_positions,
    #     )
    #     # actions2 = self._controller.forward(
    #     #     picking_position=cube_position,
    #     #     placing_position=goal_position,
    #     #     current_joint_positions=current_joint_positions,
    #     # )
    #     self._franka.apply_action(actions)
    #     # Only for the pick and place controller, indicating if the state
    #     # machine reached the final state.
    #     if self._controller.is_done():
    #         self._world.pause()
    #     return