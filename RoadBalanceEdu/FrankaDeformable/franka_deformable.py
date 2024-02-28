# Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from omni.isaac.examples.base_sample import BaseSample
# This extension has franka related tasks and controllers as well
from omni.isaac.franka import Franka
from omni.isaac.franka.controllers import PickPlaceController
from omni.isaac.franka.tasks import PickPlace
from omni.isaac.core.tasks import BaseTask
import numpy as np

from omni.isaac.core.materials.deformable_material import DeformableMaterial
from omni.isaac.core.physics_context.physics_context import PhysicsContext
from omni.physx.scripts import deformableUtils, physicsUtils  

from pxr import UsdGeom, Gf, UsdPhysics

import omni.physx
import omni.usd

import omni


class FrankaPlaying(BaseTask):
    #NOTE: we only cover here a subset of the task functions that are available,
    # checkout the base class for all the available functions to override.
    # ex: calculate_metrics, is_done..etc.
    def __init__(self, name):
        super().__init__(name=name, offset=None)
        self._goal_position = np.array([-0.3, -0.3, 0.0515 / 2.0])
        self._task_achieved = False
        return

    # Here we setup all the assets that we care about in this task.
    def set_up_scene(self, scene):
        super().set_up_scene(scene)
        scene.add_default_ground_plane()

        stage = omni.usd.get_context().get_stage()

        # Create cube
        result, path = omni.kit.commands.execute("CreateMeshPrimCommand", prim_type="Cube")
        omni.kit.commands.execute("MovePrim", path_from=path, path_to="/World/cube")
        omni.usd.get_context().get_selection().set_selected_prim_paths([], False)

        cube_mesh = UsdGeom.Mesh.Get(stage, "/World/cube")
        physicsUtils.set_or_add_translate_op(cube_mesh, translate=Gf.Vec3f(0.3, 0.3, 0.3))
        # physicsUtils.set_or_add_orient_op(cube_mesh, orient=Gf.Quatf(0.707, 0.707, 0, 0))
        physicsUtils.set_or_add_scale_op(cube_mesh, scale=Gf.Vec3f(0.04, 0.04, 0.04))
        cube_mesh.CreateDisplayColorAttr([(1.0, 0.0, 0.0)])

        # Apply PhysxDeformableBodyAPI and PhysxCollisionAPI to skin mesh and set parameter to default values
        deformableUtils.add_physx_deformable_body(
            stage,
            "/World/cube",
            collision_simplification=True,
            simulation_hexahedral_resolution=4,
            self_collision=False,
        )

        # Create a deformable body material and set it on the deformable body
        deformable_material_path = omni.usd.get_stage_next_free_path(stage, "/World/deformableBodyMaterial", True)
        deformableUtils.add_deformable_body_material(
            stage,
            deformable_material_path,
            youngs_modulus=10000.0,
            poissons_ratio=0.49,
            damping_scale=0.0,
            dynamic_friction=0.5,
        )
        self._cube_prim = stage.GetPrimAtPath("/World/cube")
        physicsUtils.add_physics_material_to_prim(stage, self._cube_prim, "/World/cube")       
        
        self._franka = scene.add(Franka(prim_path="/World/Fancy_Franka",
                                        name="fancy_franka"))

        return

    # Information exposed to solve the task is returned from the task through get_observations
    def get_observations(self):
        matrix: Gf.Matrix4d = omni.usd.get_world_transform_matrix(self._cube_prim)
        translate: Gf.Vec3d = matrix.ExtractTranslation()
        cube_position = np.array([translate[0], translate[1], translate[2]])

        current_joint_positions = self._franka.get_joint_positions()
        observations = {
            self._franka.name: {
                "joint_positions": current_joint_positions,
            },
            "deformable_cube": {
                "position": cube_position,
                "goal_position": self._goal_position
            }
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
        self._task_achieved = False
        return


class FrankaDeformable(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        return

    def _setup_simulation(self):
        self._scene = PhysicsContext()
        self._scene.set_solver_type("TGS")
        self._scene.set_broadphase_type("GPU")
        self._scene.enable_gpu_dynamics(flag=True)

    def setup_scene(self):
        world = self.get_world()
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

    async def setup_post_reset(self):
        self._controller.reset()
        await self._world.play_async()
        return

    def physics_step(self, step_size):
        # Gets all the tasks observations
        current_observations = self._world.get_observations()
        actions = self._controller.forward(
            picking_position=current_observations["deformable_cube"]["position"],
            placing_position=current_observations["deformable_cube"]["goal_position"],
            current_joint_positions=current_observations["fancy_franka"]["joint_positions"],
        )
        self._franka.apply_action(actions)
        if self._controller.is_done():
            self._world.pause()
        return