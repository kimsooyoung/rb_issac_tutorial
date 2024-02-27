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


class HelloManip(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        return

    def setup_scene(self):
        world = self.get_world()
        world.scene.add_default_ground_plane()
        franka = world.scene.add(Franka(prim_path="/World/Fancy_Franka", name="fancy_franka"))
        world.scene.add(
            DynamicCuboid(
                prim_path="/World/random_cube1",
                name="fancy_cube1",
                position=np.array([0.3, 0.3, 0.3]),
                scale=np.array([0.0515, 0.0515, 0.0515]),
                color=np.array([0, 0, 1.0]),
            )
        )
        world.scene.add(
            DynamicCuboid(
                prim_path="/World/random_cube2",
                name="fancy_cube2",
                position=np.array([0.5, 0.0, 0.3]),
                scale=np.array([0.0515, 0.0515, 0.0515]),
                color=np.array([0, 0, 1.0]),
            )
        )

        self._event = 0

        return

    async def setup_post_load(self):
        self._world = self.get_world()
        self._franka = self._world.scene.get_object("fancy_franka")
        self._fancy_cube1 = self._world.scene.get_object("fancy_cube1")
        self._fancy_cube2 = self._world.scene.get_object("fancy_cube2")
        # Initialize a pick and place controller
        self._controller = PickPlaceController(
            name="pick_place_controller",
            gripper=self._franka.gripper,
            robot_articulation=self._franka,
        )

        self._world.add_physics_callback("sim_step", callback_fn=self.physics_step)
        # World has pause, stop, play..etc
        # Note: if async version exists, use it in any async function is this workflow
        self._franka.gripper.set_joint_positions(self._franka.gripper.joint_opened_positions)
        await self._world.play_async()
        return

    # This function is called after Reset button is pressed
    # Resetting anything in the world should happen here
    async def setup_post_reset(self):
        self._controller.reset()
        self._franka.gripper.set_joint_positions(self._franka.gripper.joint_opened_positions)
        await self._world.play_async()
        return

    def physics_step(self, step_size):
        cube_position1, _ = self._fancy_cube1.get_world_pose()
        cube_position2, _ = self._fancy_cube2.get_world_pose()
        goal_position1 = np.array([-0.3, -0.3, 0.0515 / 2.0])
        goal_position2 = np.array([-0.2, -0.3, 0.0515 / 2.0])

        current_joint_positions = self._franka.get_joint_positions()

        if self._event == 0:
            actions = self._controller.forward(
                picking_position=cube_position1,
                placing_position=goal_position1,
                current_joint_positions=current_joint_positions,
            )
            self._franka.apply_action(actions)
        
        elif self._event == 1:
            actions = self._controller.forward(
                picking_position=cube_position2,
                placing_position=goal_position2,
                current_joint_positions=current_joint_positions,
            )
            self._franka.apply_action(actions)

        # Only for the pick and place controller, indicating if the state
        # machine reached the final state.
        
        if self._controller.is_done():
            self._event += 1
            self._controller.reset()

        return