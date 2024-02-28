# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import h5py
import numpy as np
import omni.isaac.core.utils.numpy.rotations as rot_utils

from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.universal_robots.controllers.pick_place_controller import PickPlaceController
from omni.isaac.universal_robots.tasks import BinFilling as BinFillingTask
from omni.isaac.core import SimulationContext
from omni.isaac.sensor import Camera


class BinFilling(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self._controller = None
        self._articulation_controller = None
        self._added_screws = False

        self._sim_time_list = []
        self._joint_positions = []
        self._joint_velocities = []

        self._camera1_img = []
        self._camera2_img = []
        self._camera3_img = []

    def setup_scene(self):
        world = self.get_world()
        self.simulation_context = SimulationContext()
        world.add_task(BinFillingTask(name="bin_filling"))
        
        self._save_count = 0

        self._camera1 = Camera(
            prim_path="/World/Scene/ur10/ee_link/ee_camera",
            # position=np.array([0.088, 0.0, 0.926]),
            translation=np.array([0.0, 0.0, -0.1]),
            frequency=30,
            resolution=(640, 480),
            orientation=rot_utils.euler_angles_to_quats(
                np.array([
                    180.0, -30.0, 0.0
                ]), degrees=True),
        )
        self._camera1.set_clipping_range(0.1, 1000000.0)
        self._camera1.set_focal_length(1.5)
        self._camera1.initialize()
        self._camera1.add_motion_vectors_to_frame()
        self._camera1.set_visibility(False)

        self._camera2 = Camera(
            prim_path="/World/side_camera",
            position=np.array([2.5, 0.0, 0.0]),
            # translation=np.array([0.0, 0.0, -0.1]),
            frequency=30,
            resolution=(640, 480),
            orientation=rot_utils.euler_angles_to_quats(
                np.array([
                    0.0, 0.0, 180.0
                ]), degrees=True),
        )
        self._camera2.set_focal_length(1.5)
        self._camera2.set_visibility(False)
        self._camera2.initialize()

        self._camera3 = Camera(
            prim_path="/World/front_camera",
            position=np.array([0.0, 2.0, 0.0]),
            # translation=np.array([0.0, 0.0, -0.1]),
            frequency=30,
            resolution=(640, 480),
            orientation=rot_utils.euler_angles_to_quats(
                np.array([
                    0.0, 0.0, -90.0
                ]), degrees=True),
        )
        self._camera3.set_focal_length(1.5)
        self._camera3.set_visibility(False)
        self._camera3.initialize()

        self._f = h5py.File('ur_bin_filling.hdf5','w')
        self._group_f = self._f.create_group("isaac_dataset")

        self._save_count = 0
        self._img_f = self._group_f.create_group("camera_images")

        return

    async def setup_post_load(self):
        self._ur10_task = self._world.get_task(name="bin_filling")
        self._task_params = self._ur10_task.get_params()
        self._my_ur10 = self._world.scene.get_object(self._task_params["robot_name"]["value"])
        self._controller = PickPlaceController(
            name="pick_place_controller", gripper=self._my_ur10.gripper, robot_articulation=self._my_ur10
        )
        self._articulation_controller = self._my_ur10.get_articulation_controller()
        return

    def _on_fill_bin_physics_step(self, step_size):

        self._camera1.get_current_frame()
        self._camera2.get_current_frame()
        self._camera3.get_current_frame()

        current_time = self.simulation_context.current_time
        current_joint_state = self._my_ur10.get_joints_state()
        current_joint_positions = current_joint_state.positions
        current_joint_velocities = current_joint_state.velocities

        if self._save_count % 100 == 0:

            self._sim_time_list.append(current_time)
            self._joint_positions.append(current_joint_positions)
            self._joint_velocities.append(current_joint_velocities)

            self._camera1_img.append(self._camera1.get_rgba()[:, :, :3])
            self._camera2_img.append(self._camera2.get_rgba()[:, :, :3])
            self._camera3_img.append(self._camera3.get_rgba()[:, :, :3])

            print("Collecting data...")

        observations = self._world.get_observations()
        actions = self._controller.forward(
            picking_position=observations[self._task_params["bin_name"]["value"]]["position"],
            placing_position=observations[self._task_params["bin_name"]["value"]]["target_position"],
            current_joint_positions=observations[self._task_params["robot_name"]["value"]]["joint_positions"],
            end_effector_offset=np.array([0, -0.098, 0.03]),
            end_effector_orientation=euler_angles_to_quat(np.array([np.pi, 0, np.pi / 2.0])),
        )
        # if not self._added_screws and self._controller.get_current_event() == 6 and not self._controller.is_paused():
        #     self._controller.pause()
        #     self._ur10_task.add_screws(screws_number=20)
        #     self._added_screws = True
        # if self._controller.is_done():
        #     self._world.pause()
        self._articulation_controller.apply_action(actions)

        self._save_count += 1

        if self._controller.is_done():
            self.save_data()
        return

    async def on_fill_bin_event_async(self):
        world = self.get_world()
        world.add_physics_callback("sim_step", self._on_fill_bin_physics_step)
        await world.play_async()
        return

    async def setup_pre_reset(self):
        world = self.get_world()
        if world.physics_callback_exists("sim_step"):
            world.remove_physics_callback("sim_step")
        self._controller.reset()
        self._added_screws = False
        return

    def world_cleanup(self):
        self._controller = None
        self._added_screws = False
        return

    def save_data(self):

        self._group_f.create_dataset(f"sim_time", data=self._sim_time_list, compression='gzip', compression_opts=9)
        self._group_f.create_dataset(f"joint_positions", data=self._joint_positions, compression='gzip', compression_opts=9)
        self._group_f.create_dataset(f"joint_velocities", data=self._joint_velocities, compression='gzip', compression_opts=9)

        self._img_f.create_dataset(f"ee_camera", data=self._camera1_img, compression='gzip', compression_opts=9)
        self._img_f.create_dataset(f"side_camera", data=self._camera2_img, compression='gzip', compression_opts=9)
        self._img_f.create_dataset(f"front_camera", data=self._camera3_img, compression='gzip', compression_opts=9)

        self._f.close()

        print("Data saved")
        self._save_count = 0
        self._world.pause()

        return