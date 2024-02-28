# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
import random

import numpy as np
import omni
import h5py
import omni.isaac.cortex.math_util as math_util
import omni.isaac.core.utils.numpy.rotations as rot_utils

from omni.isaac.core import SimulationContext
from omni.isaac.core.objects.capsule import VisualCapsule
from omni.isaac.core.objects.sphere import VisualSphere
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.tasks.base_task import BaseTask
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.cortex.cortex_rigid_prim import CortexRigidPrim
from omni.isaac.cortex.cortex_utils import get_assets_root_path
from omni.isaac.cortex.robot import CortexUr10
from omni.isaac.cortex.sample_behaviors.ur10 import bin_stacking_behavior as behavior
from omni.isaac.examples.cortex.cortex_base import CortexBase
from omni.isaac.sensor import Camera


class Ur10Assets:
    def __init__(self):
        self.assets_root_path = get_assets_root_path()

        self.ur10_table_usd = (
            self.assets_root_path + "/Isaac/Samples/Leonardo/Stage/ur10_bin_stacking_short_suction.usd"
        )
        self.small_klt_usd = self.assets_root_path + "/Isaac/Props/KLT_Bin/small_KLT.usd"
        self.background_usd = self.assets_root_path + "/Isaac/Environments/Simple_Warehouse/warehouse.usd"
        self.rubiks_cube_usd = self.assets_root_path + "/Isaac/Props/Rubiks_Cube/rubiks_cube.usd"


def random_bin_spawn_transform():
    x = random.uniform(-0.15, 0.15)
    y = 1.5
    z = -0.15
    position = np.array([x, y, z])

    z = random.random() * 0.02 - 0.01
    w = random.random() * 0.02 - 0.01
    norm = np.sqrt(z**2 + w**2)
    quat = math_util.Quaternion([w / norm, 0, 0, z / norm])
    if random.random() > 0.5:
        print("<flip>")
        # flip the bin so it's upside down
        quat = quat * math_util.Quaternion([0, 0, 1, 0])
    else:
        print("<no flip>")

    return position, quat.vals


class BinStackingTask(BaseTask):
    def __init__(self, env_path, assets) -> None:
        super().__init__("bin_stacking")
        self.assets = assets

        self.env_path = env_path
        self.bins = []
        self.stashed_bins = []
        self.on_conveyor = None

    def _spawn_bin(self, rigid_bin):
        x, q = random_bin_spawn_transform()
        rigid_bin.set_world_pose(position=x, orientation=q)
        rigid_bin.set_linear_velocity(np.array([0, -0.30, 0]))
        rigid_bin.set_visibility(True)

    def post_reset(self) -> None:
        if len(self.bins) > 0:
            for rigid_bin in self.bins:
                self.scene.remove_object(rigid_bin.name)
            self.bins.clear()

        self.on_conveyor = None

    def pre_step(self, time_step_index, simulation_time) -> None:
        """Spawn a new randomly oriented bin if the previous bin has been placed."""
        spawn_new = False
        if self.on_conveyor is None:
            spawn_new = True
        else:
            (x, y, z), _ = self.on_conveyor.get_world_pose()
            is_on_conveyor = y > 0.0 and -0.4 < x and x < 0.4
            if not is_on_conveyor:
                spawn_new = True

        if spawn_new:
            name = "bin_{}".format(len(self.bins))
            prim_path = self.env_path + "/bins/{}".format(name)
            # "/Isaac/Props/KLT_Bin/small_KLT.usd"
            # prim_path
            add_reference_to_stage(usd_path=self.assets.small_klt_usd, prim_path=prim_path)
            self.on_conveyor = self.scene.add(CortexRigidPrim(name=name, prim_path=prim_path))

            self._spawn_bin(self.on_conveyor)
            self.bins.append(self.on_conveyor)

    def world_cleanup(self):
        self.bins = []
        self.stashed_bins = []
        self.on_conveyor = None
        return


class BinStacking(CortexBase):
    def __init__(self, monitor_fn=None):
        super().__init__()
        self._monitor_fn = monitor_fn
        self.robot = None

        self._sim_time_list = []
        self._joint_positions = []
        self._joint_velocities = []

        self._camera1_img = []
        self._camera2_img = []
        self._camera3_img = []
        self._camera4_img = []
        self._camera5_img = []

        self._save_count = 0

    def _setup_camera(self):

        self._camera1 = Camera(
            prim_path="/World/Ur10Table/ur10/ee_link/ee_camera",
            # position=np.array([0.088, 0.0, 0.926]),
            translation=np.array([-0.15, 0.0, -0.1]),
            frequency=30,
            resolution=(640, 480),
            orientation=rot_utils.euler_angles_to_quats(
                np.array([
                    180.0, -15.0, 0.0
                ]), degrees=True),
        )
        self._camera1.set_clipping_range(0.1, 1000000.0)
        self._camera1.set_focal_length(1.5)
        self._camera1.initialize()
        self._camera1.add_motion_vectors_to_frame()
        self._camera1.set_visibility(False)

        self._camera2 = Camera(
            prim_path="/World/left_camera",
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
            prim_path="/World/right_camera",
            position=np.array([-2.5, 0.0, 0.0]),
            # translation=np.array([0.0, 0.0, -0.1]),
            frequency=30,
            resolution=(640, 480),
            orientation=rot_utils.euler_angles_to_quats(
                np.array([
                    0.0, 0.0, 0.0
                ]), degrees=True),
        )
        self._camera3.set_focal_length(1.5)
        self._camera3.set_visibility(False)
        self._camera3.initialize()

        self._camera4 = Camera(
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
        self._camera4.set_focal_length(1.5)
        self._camera4.set_visibility(False)
        self._camera4.initialize()

        self._camera5 = Camera(
            prim_path="/World/back_camera",
            position=np.array([0.5, -2.0, -0.2]),
            # translation=np.array([0.0, 0.0, -0.1]),
            frequency=30,
            resolution=(640, 480),
            orientation=rot_utils.euler_angles_to_quats(
                np.array([
                    0.0, 0.0, 90.0
                ]), degrees=True),
        )
        self._camera5.set_focal_length(1.5)
        self._camera5.set_visibility(False)
        self._camera5.initialize()

    def _setup_data_collection(self):
        self._f = h5py.File('ur_bin_palleting.hdf5','w')
        self._group_f = self._f.create_group("isaac_dataset")

        self._save_count = 0
        self._img_f = self._group_f.create_group("camera_images")

    def setup_scene(self):
        world = self.get_world()
        self.simulation_context = SimulationContext()

        env_path = "/World/Ur10Table"
        ur10_assets = Ur10Assets()
        add_reference_to_stage(usd_path=ur10_assets.ur10_table_usd, prim_path=env_path)
        add_reference_to_stage(usd_path=ur10_assets.background_usd, prim_path="/World/Background")
        background_prim = XFormPrim(
            "/World/Background", position=[10.00, 2.00, -1.18180], orientation=[0.7071, 0, 0, 0.7071]
        )
        self.robot = world.add_robot(CortexUr10(name="robot", prim_path="{}/ur10".format(env_path)))

        obs = world.scene.add(
            VisualSphere(
                "/World/Ur10Table/Obstacles/FlipStationSphere",
                name="flip_station_sphere",
                position=np.array([0.73, 0.76, -0.13]),
                radius=0.2,
                visible=False,
            )
        )
        self.robot.register_obstacle(obs)
        obs = world.scene.add(
            VisualSphere(
                "/World/Ur10Table/Obstacles/NavigationDome",
                name="navigation_dome_obs",
                position=[-0.031, -0.018, -1.086],
                radius=1.1,
                visible=False,
            )
        )
        self.robot.register_obstacle(obs)

        az = np.array([1.0, 0.0, -0.3])
        ax = np.array([0.0, 1.0, 0.0])
        ay = np.cross(az, ax)
        R = math_util.pack_R(ax, ay, az)
        quat = math_util.matrix_to_quat(R)
        obs = world.scene.add(
            VisualCapsule(
                "/World/Ur10Table/Obstacles/NavigationBarrier",
                name="navigation_barrier_obs",
                position=[0.471, 0.276, -0.463 - 0.1],
                orientation=quat,
                radius=0.5,
                height=0.9,
                visible=False,
            )
        )
        self.robot.register_obstacle(obs)

        obs = world.scene.add(
            VisualCapsule(
                "/World/Ur10Table/Obstacles/NavigationFlipStation",
                name="navigation_flip_station_obs",
                position=np.array([0.766, 0.755, -0.5]),
                radius=0.5,
                height=0.5,
                visible=False,
            )
        )
        self.robot.register_obstacle(obs)

        self._setup_camera()
        self._setup_data_collection()

    async def setup_post_load(self):
        world = self.get_world()
        env_path = "/World/Ur10Table"
        ur10_assets = Ur10Assets()
        if not self.robot:
            self.robot = world._robots["robot"]
            world._current_tasks.clear()
            world._behaviors.clear()
            world._logical_state_monitors.clear()
        self.task = BinStackingTask(env_path, ur10_assets)
        print(world.scene)
        self.task.set_up_scene(world.scene)
        world.add_task(self.task)
        self.decider_network = behavior.make_decider_network(self.robot, self._on_monitor_update)
        world.add_decider_network(self.decider_network)

        return

    def _on_monitor_update(self, diagnostics):
        decision_stack = ""
        if self.decider_network._decider_state.stack:
            decision_stack = "\n".join(
                [
                    "{0}{1}".format("  " * i, element)
                    for i, element in enumerate(str(i) for i in self.decider_network._decider_state.stack)
                ]
            )

        if self._monitor_fn:
            self._monitor_fn(diagnostics, decision_stack)

    def _on_physics_step(self, step_size):
        world = self.get_world()

        self._camera1.get_current_frame()
        self._camera2.get_current_frame()
        self._camera3.get_current_frame()
        self._camera4.get_current_frame()
        self._camera5.get_current_frame()

        current_time = self.simulation_context.current_time
        current_joint_state = self.robot.get_joints_state()
        current_joint_positions = current_joint_state.positions
        current_joint_velocities = current_joint_state.velocities

        print(self._save_count)

        if self._save_count % 50 == 0:

            self._sim_time_list.append(current_time)
            self._joint_positions.append(current_joint_positions)
            self._joint_velocities.append(current_joint_velocities)

            self._camera1_img.append(self._camera1.get_rgba()[:, :, :3])
            self._camera2_img.append(self._camera2.get_rgba()[:, :, :3])
            self._camera3_img.append(self._camera3.get_rgba()[:, :, :3])
            self._camera4_img.append(self._camera4.get_rgba()[:, :, :3])
            self._camera5_img.append(self._camera5.get_rgba()[:, :, :3])

            print("Collecting data...")

        if self._save_count > 3000:
            self.save_data()

        self._save_count += 1
        
        world.step(False, False)
        return

    async def on_event_async(self):
        world = self.get_world()
        await omni.kit.app.get_app().next_update_async()
        world.reset_cortex()
        world.add_physics_callback("sim_step", self._on_physics_step)
        await world.play_async()
        return

    async def setup_pre_reset(self):
        world = self.get_world()
        if world.physics_callback_exists("sim_step"):
            world.remove_physics_callback("sim_step")
        return

    def world_cleanup(self):
        return

    def save_data(self):

        self._group_f.create_dataset(f"sim_time", data=self._sim_time_list, compression='gzip', compression_opts=9)
        self._group_f.create_dataset(f"joint_positions", data=self._joint_positions, compression='gzip', compression_opts=9)
        self._group_f.create_dataset(f"joint_velocities", data=self._joint_velocities, compression='gzip', compression_opts=9)

        self._img_f.create_dataset(f"ee_camera", data=self._camera1_img, compression='gzip', compression_opts=9)
        self._img_f.create_dataset(f"left_camera", data=self._camera2_img, compression='gzip', compression_opts=9)
        self._img_f.create_dataset(f"right_camera", data=self._camera3_img, compression='gzip', compression_opts=9)
        self._img_f.create_dataset(f"front_camera", data=self._camera4_img, compression='gzip', compression_opts=9)
        self._img_f.create_dataset(f"back_camera", data=self._camera5_img, compression='gzip', compression_opts=9)

        self._f.close()

        print("Data saved")
        self._save_count = 0
        self._world.pause()

        return