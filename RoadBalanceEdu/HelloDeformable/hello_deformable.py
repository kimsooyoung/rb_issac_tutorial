# Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import numpy as np
from omni.isaac.core.materials.deformable_material import DeformableMaterial
from omni.isaac.core.physics_context.physics_context import PhysicsContext
from omni.physx.scripts import deformableUtils, physicsUtils  
from omni.isaac.examples.base_sample import BaseSample

from pxr import UsdGeom, Gf, UsdPhysics

import omni.physx
import omni.usd

import omni

class HelloDeformable(BaseSample):  
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

        stage = omni.usd.get_context().get_stage()

        world.scene.add_default_ground_plane()
        
        # Create cube
        result, path = omni.kit.commands.execute("CreateMeshPrimCommand", prim_type="Cube")
        omni.kit.commands.execute("MovePrim", path_from=path, path_to="/World/cube")
        omni.usd.get_context().get_selection().set_selected_prim_paths([], False)

        cube_mesh = UsdGeom.Mesh.Get(stage, "/World/cube")
        physicsUtils.set_or_add_translate_op(cube_mesh, translate=Gf.Vec3f(0.0, 0.0, 0.5))
        # physicsUtils.set_or_add_orient_op(cube_mesh, orient=Gf.Quatf(0.707, 0.707, 0, 0))
        physicsUtils.set_or_add_scale_op(cube_mesh, scale=Gf.Vec3f(0.1, 0.1, 0.1))
        cube_mesh.CreateDisplayColorAttr([(1.0, 0.0, 0.0)])

        # Apply PhysxDeformableBodyAPI and PhysxCollisionAPI to skin mesh and set parameter to default values
        deformableUtils.add_physx_deformable_body(
            stage,
            "/World/cube",
            collision_simplification=True,
            simulation_hexahedral_resolution=10,
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
        physicsUtils.add_physics_material_to_prim(stage, stage.GetPrimAtPath("/World/cube"), "/World/cube")       


    async def setup_post_load(self):        
        self._world = self.get_world()
        return