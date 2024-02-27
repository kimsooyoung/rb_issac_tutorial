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
from omni.isaac.core.prims.geometry_prim import GeometryPrim
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.universal_robots import UR10
from omni.isaac.core.materials.physics_material import PhysicsMaterial
from omni.isaac.core.physics_context.physics_context import PhysicsContext

from pxr import Gf, PhysxSchema, Usd, UsdPhysics, UsdShade

import carb

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

        self._num_nuts = 2
        self._num_bins = 2

        # Asset Path from Nucleus        
        # self._cube_asset_path = get_assets_root_path() + "/Isaac/Props/Blocks/nvidia_cube.usd"
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
        # self._nut_position_x = np.array([0.28, 0.4])
        # self._nut_position_y = np.array([-0.35, -0.15])
        # self._nut_position_z = 0.2
        self._nuts = []
        self._nuts_offset = 0.005

        return

    def setup_scene(self):
        world = self.get_world()
        world.scene.add_default_ground_plane()
        prim = get_prim_at_path("/World/defaultGroundPlane")

        self._setup_simulation()

        franka = world.scene.add(Franka(prim_path="/World/Fancy_Franka", name="fancy_franka"))
        # ur10 = world.scene.add(UR10(prim_path="/World/UR10", name="UR10"))

        # RigidPrim Ref.
        # https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.core/docs/index.html#omni.isaac.core.prims.RigidPrim
        # GeometryPrim Ref.
        # https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.core/docs/index.html?highlight=geometryprim#omni.isaac.core.prims.GeometryPrim

        for bins in range(self._num_bins):
            add_reference_to_stage(
                usd_path=self._bin_asset_path, 
                prim_path=f"/World/bin{bins}",
            )
            _bin = world.scene.add(
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
            
            # nut_position = np.array([
            #     np.random.randint(*(self._nut_position_x*100)) / 100,
            #     np.random.randint(*(self._nut_position_y*100)) / 100,
            #     self._nut_position_z,
            # ])

            add_reference_to_stage(
                usd_path=self._nut_asset_path, 
                prim_path=f"/World/nut{nut}",
            )
            nut = world.scene.add(
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

    async def setup_post_load(self):
        self._world = self.get_world()
        self._franka = self._world.scene.get_object("fancy_franka")
        
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

        target_position, _ = self._nuts[0].get_world_pose()
        target_position[2] += self._nuts_offset

        goal_position, _ = self._bins[1].get_world_pose()
        goal_position[2] += self._bins_offset
        # print(goal_position)
        current_joint_positions = self._franka.get_joint_positions()
        actions = self._controller.forward(
            picking_position=target_position,
            placing_position=goal_position,
            current_joint_positions=current_joint_positions,
        )

        self._franka.apply_action(actions)
        # Only for the pick and place controller, indicating if the state
        # machine reached the final state.
        if self._controller.is_done():
            self._world.pause()
        return