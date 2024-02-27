# Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from omni.isaac.franka.controllers import PickPlaceController
from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.core.tasks import BaseTask
from omni.isaac.franka import Franka

from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units
from omni.isaac.core.physics_context.physics_context import PhysicsContext
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.prims.geometry_prim import GeometryPrim
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.sensor import Camera

import omni.isaac.core.utils.numpy.rotations as rot_utils
from omni.isaac.core import SimulationContext

from pxr import PhysxSchema

import numpy as np
import h5py

class FrankaPlaying(BaseTask):
    #NOTE: we only cover here a subset of the task functions that are available,
    # checkout the base class for all the available functions to override.
    # ex: calculate_metrics, is_done..etc.
    def __init__(self, name):
        super().__init__(name=name, offset=None)

        self._num_nuts = 2
        self._num_bins = 2

        # Asset Path from Nucleus        
        self._bin_asset_path = get_assets_root_path() + "/Isaac/Props/KLT_Bin/small_KLT.usd"
        self._nut_asset_path = get_assets_root_path() + "/Isaac/Samples/Examples/FrankaNutBolt/SubUSDs/Nut/M20_Nut_Tight_R256_Franka_SI.usd"

        self._bin_position = np.array([
            [ 0.35, -0.25, 0.1],
            [ 0.35,  0.25, 0.1],
        ])
        self._bins = []
        self._bins_offset = 0.1

        self._nuts_position = np.array([
            [0.35, -0.22, 0.2],
            [0.30, -0.28, 0.2],
        ])
        self._nuts = []
        self._nuts_offset = 0.005

        self._goal_position = np.array([
            [0.35,  0.18, 0.2],
            [0.30,  0.25, 0.2],
        ])

        self._pick_position = np.array([0, 0, 0])

        self._task_achieved = False
        self._task_event = 0

        return

    # Here we setup all the assets that we care about in this task.
    def set_up_scene(self, scene):
        super().set_up_scene(scene)
        scene.add_default_ground_plane()

        self._franka = scene.add(
            Franka(
                prim_path="/World/Fancy_Franka",
                name="fancy_franka"
            )
        )

        # Exception: You can not define translation and position at the same time
        self._camera1 = Camera(
            prim_path="/World/Fancy_Franka/panda_hand/hand_camera",
            # position=np.array([0.088, 0.0, 0.926]),
            translation=np.array([0.1, 0.0, -0.1]),
            frequency=30,
            resolution=(640, 480),
            orientation=rot_utils.euler_angles_to_quats(
                np.array([
                    180, -90 - 25, 0
                ]), degrees=True),
        )
        self._camera1.set_clipping_range(0.1, 1000000.0)
        self._camera1.initialize()
        self._camera1.add_motion_vectors_to_frame()
        self._camera1.set_visibility(False)

        self._camera2 = Camera(
            prim_path="/World/top_camera",
            position=np.array([0.0, 0.0, 5.0]),
            frequency=30,
            resolution=(640, 480),
            orientation=rot_utils.euler_angles_to_quats(
                np.array([
                    0, 90, 0
                ]), degrees=True),
        )
        self._camera2.initialize()
        self._camera2.set_visibility(False)

        self._camera3 = Camera(
            prim_path="/World/front_camera",
            position=np.array([1.0, 0.0, 0.3]),
            frequency=30,
            resolution=(640, 480),
            orientation=rot_utils.euler_angles_to_quats(
                np.array([
                    0, 0, 180
                ]), degrees=True),
        )
        self._camera3.set_clipping_range(0.1, 1000000.0)
        self._camera3.set_focal_length(1.0)
        self._camera3.initialize()
        self._camera3.set_visibility(False)

        for bins in range(self._num_bins):
            add_reference_to_stage(
                usd_path=self._bin_asset_path, 
                prim_path=f"/World/bin{bins}",
            )
            _bin = scene.add(
                RigidPrim(
                    prim_path=f"/World/bin{bins}",
                    name=f"bin{bins}",
                    position=self._bin_position[bins] / get_stage_units(),
                    orientation=euler_angles_to_quat(np.array([np.pi, 0., 0.])),
                    mass=0.1, # kg
                )
            )
            self._bins.append(_bin)

        for nut in range(self._num_nuts):
            add_reference_to_stage(
                usd_path=self._nut_asset_path, 
                prim_path=f"/World/nut{nut}",
            )
            nut = scene.add(
                GeometryPrim(
                    prim_path=f"/World/nut{nut}",
                    name=f"nut{nut}_geom",
                    position=self._nuts_position[nut] / get_stage_units(),
                    collision=True,
                    # mass=0.1, # kg
                )
            )
            self._nuts.append(nut)

        return

    # Information exposed to solve the task is returned from the task through get_observations
    def get_observations(self):

        current_joint_positions = self._franka.get_joint_positions()
        currnet_joint_velocities = self._franka.get_joint_velocities()

        self._pick_position1, _ = self._nuts[0].get_world_pose()
        self._pick_position1[2] += self._nuts_offset

        self._pick_position2, _ = self._nuts[1].get_world_pose()
        self._pick_position2[2] += self._nuts_offset

        observations = {
            self._franka.name: {
                "joint_positions": current_joint_positions,
                "joint_velocities": currnet_joint_velocities,
            },
            "nut0_geom": {
                "position": self._pick_position1,
                "goal_position": self._goal_position[0],
            },
            "nut1_geom": {
                "position": self._pick_position2,
                "goal_position": self._goal_position[1],
            },
        }

        return observations

    # Called before each physics step,
    # for instance we can check here if the task was accomplished by
    # changing the color of the cube once its accomplished
    def pre_step(self, control_index, simulation_time):

        return

    # Called after each reset,
    # for instance we can always set the gripper to be opened at the beginning after each reset
    # also we can set the cube's color to be blue
    def post_reset(self):
        self._franka.gripper.set_joint_positions(self._franka.gripper.joint_opened_positions)
        # self._nuts[0].get_applied_visual_material().set_color(color=np.array([0, 0, 1.0]))
        self._task_achieved = False
        return


class HelloManip(BaseSample):
    def __init__(self) -> None:
        super().__init__()

        # some global sim options:
        self._time_steps_per_second = 240  # 4.167ms aprx
        self._fsm_update_rate = 60
        self._solverPositionIterations = 4
        self._solverVelocityIterations = 1
        self._solver_type = "TGS"
        self._ik_damping = 0.1

        self._event = 0
        self._step_size = 0.01

        self._f = h5py.File('/home/kimsooyoung/Documents/franka_nuts_pick_and_place.hdf5','w')
        self._sim_time = self._f.create_group("simulation_time")
        self._joint_pos_f = self._f.create_group("joint_positions")
        self._joint_vel_f = self._f.create_group("joint_velocities")

        img_f = self._f.create_group("camera_images")
        self._camera1_f = img_f.create_group("camera1")
        self._camera2_f = img_f.create_group("camera2")
        self._camera3_f = img_f.create_group("camera3")

        self._joint_positions = []
        self._joint_velocities = []

        return

    def setup_scene(self):
        world = self.get_world()
        self.simulation_context = SimulationContext()
        self._setup_simulation()

        # We add the task to the world here
        world.add_task(FrankaPlaying(name="my_first_task"))
        return

    async def setup_post_load(self):
        self._world = self.get_world()
        # The world already called the setup_scene from the task (with first reset of the world)
        # so we can retrieve the task objects
        self._franka = self._world.scene.get_object("fancy_franka")
        self._controller = PickPlaceController(
            name="pick_place_controller",
            gripper=self._franka.gripper,
            robot_articulation=self._franka,
        )
        self._world.add_physics_callback("sim_step", callback_fn=self.physics_step)
        await self._world.play_async()
        return


    def _setup_simulation(self):
        self._scene = PhysicsContext()
        self._scene.set_solver_type(self._solver_type)
        self._scene.set_broadphase_type("GPU")
        self._scene.enable_gpu_dynamics(flag=True)
        self._scene.set_friction_offset_threshold(0.01)
        self._scene.set_friction_correlation_distance(0.0005)
        self._scene.set_gpu_total_aggregate_pairs_capacity(10 * 1024)
        self._scene.set_gpu_found_lost_pairs_capacity(10 * 1024)
        self._scene.set_gpu_heap_capacity(64 * 1024 * 1024)
        self._scene.set_gpu_found_lost_aggregate_pairs_capacity(10 * 1024)
        # added because of new errors regarding collisionstacksize
        physxSceneAPI = PhysxSchema.PhysxSceneAPI.Apply(get_prim_at_path("/physicsScene"))
        physxSceneAPI.CreateGpuCollisionStackSizeAttr().Set(76000000)  # or whatever min is needed


    async def setup_post_reset(self):
        self._controller.reset()
        await self._world.play_async()
        return

    def physics_step(self, step_size):
        # Gets all the tasks observations
        current_observations = self._world.get_observations()
        current_time = self.simulation_context.current_time
        current_joint_pos = current_observations["fancy_franka"]["joint_positions"]
        current_joint_vel = current_observations["fancy_franka"]["joint_velocities"]
        # print(step_size)

        self._sim_time.create_dataset(f"{current_time}", data=current_time)

        if self._event == 0:
            actions = self._controller.forward(
                picking_position=current_observations["nut0_geom"]["position"],
                placing_position=current_observations["nut0_geom"]["goal_position"],
                current_joint_positions=current_joint_pos,
            )
            self._franka.apply_action(actions)

        if self._event == 1:
            actions = self._controller.forward(
                picking_position=current_observations["nut1_geom"]["position"],
                placing_position=current_observations["nut1_geom"]["goal_position"],
                current_joint_positions=current_joint_pos,
            )
            self._franka.apply_action(actions)

        if self._controller.is_done():
            self._controller.reset()
            self._event += 1
            if self._event == 2:
                self._f.close()
            # self._world.pause()
        return