import bpy
import bmesh
import os
import array
from mathutils import Vector, Quaternion, Matrix
from bpy_extras.image_utils import load_image
from . import semodel as SEModel


def __build_image_path__(asset_path, image_path):
    root_path = os.path.dirname(asset_path)
    return os.path.join(root_path, image_path)


def load(self, context, filepath=""):
    scene = bpy.context.scene
    model_name = os.path.splitext(os.path.basename(filepath))[0]

    # Читаем модель
    model = SEModel.Model(filepath)

    mesh_objs = []
    mesh_mats = []

    # Материалы
    for mat in model.materials:
        new_mat = bpy.data.materials.get(mat.name)
        if new_mat is not None:
            mesh_mats.append(new_mat)
            continue

        new_mat = bpy.data.materials.new(name=mat.name)
        new_mat.use_nodes = True
        bsdf_shader = new_mat.node_tree.nodes[
            bpy.app.translations.pgettext_data("Principled BSDF")
        ]
        material_color_map = new_mat.node_tree.nodes.new("ShaderNodeTexImage")
        try:
            material_color_map.image = bpy.data.images.load(
                __build_image_path__(filepath, mat.inputData.diffuseMap)
            )
        except RuntimeError:
            pass
        new_mat.node_tree.links.new(
            bsdf_shader.inputs["Base Color"],
            material_color_map.outputs["Color"],
        )
        mesh_mats.append(new_mat)

    # Меши
    for mesh in model.meshes:
        new_mesh = bpy.data.meshes.new("SEModelMesh")
        blend_mesh = bmesh.new()

        # Цвета, веса, UV
        vcol_layer = blend_mesh.loops.layers.color.new("Color")
        vweight_layer = blend_mesh.verts.layers.deform.new()
        uv_layers = [
            blend_mesh.loops.layers.uv.new(f"UVSet_{i}")
            for i in range(mesh.matReferenceCount)
        ]

        # Вершины
        for v in mesh.vertices:
            blend_mesh.verts.new(Vector(v.position))
        blend_mesh.verts.ensure_lookup_table()

        # Веса
        for idx, v in enumerate(mesh.vertices):
            for bone_idx, weight in v.weights:
                if weight > 0.0:
                    blend_mesh.verts[idx][vweight_layer][bone_idx] = weight

        # Нормали буфер
        normal_buffer = []
        face_index_map = [0, 2, 1]

        def setup_face_vert(bm_face, face):
            for loop_idx, loop in enumerate(bm_face.loops):
                vert_idx = face.indices[face_index_map[loop_idx]]
                normal_buffer.append(mesh.vertices[vert_idx].normal)

                # UV
                for ui, uv_layer in enumerate(uv_layers):
                    uv = Vector(mesh.vertices[vert_idx].uvLayers[ui])
                    uv.y = 1.0 - uv.y
                    loop[uv_layer].uv = uv

                # Цвет
                loop[vcol_layer] = mesh.vertices[vert_idx].color

        # Полигоны
        for face in mesh.faces:
            verts = [
                blend_mesh.verts[face.indices[0]],
                blend_mesh.verts[face.indices[2]],
                blend_mesh.verts[face.indices[1]],
            ]
            try:
                bm_face = blend_mesh.faces.new(verts)
            except ValueError:
                continue
            setup_face_vert(bm_face, face)

        # Генерим меш
        blend_mesh.to_mesh(new_mesh)

        # === ФИКС для Blender 4.1 ===
        new_mesh.validate(clean_customdata=False)
        new_mesh.calc_normals()
        polygon_count = len(new_mesh.polygons)
        new_mesh.polygons.foreach_set("use_smooth", [True] * polygon_count)
        # ============================

        # Создаём объект и привязываем материалы
        obj = bpy.data.objects.new(f"{model_name}_{new_mesh.name}", new_mesh)
        for mi in mesh.materialReferences:
            if mi >= 0:
                obj.data.materials.append(mesh_mats[mi])

        bpy.context.view_layer.active_layer_collection.collection.objects.link(obj)
        mesh_objs.append(obj)

        # Группы вершин для скининга
        for bone in model.bones:
            obj.vertex_groups.new(name=bone.name)

    # Скелет
    arm = bpy.data.armatures.new(f"{model_name}_amt")
    arm.display_type = "STICK"
    skel_obj = bpy.data.objects.new(f"{model_name}_skel", arm)
    skel_obj.show_in_front = True
    bpy.context.view_layer.active_layer_collection.collection.objects.link(skel_obj)
    bpy.context.view_layer.objects.active = skel_obj

    bpy.ops.object.mode_set(mode='EDIT')
    bone_mats = {}
    for bone in model.bones:
        eb = arm.edit_bones.new(bone.name)
        eb.tail = (0, 0.05, 0)
        mat_rot = Quaternion((
            bone.localRotation[3],
            bone.localRotation[0],
            bone.localRotation[1],
            bone.localRotation[2],
        )).to_matrix().to_4x4()
        mat_trans = Matrix.Translation(Vector(bone.localPosition))
        final = mat_trans @ mat_rot
        bone_mats[bone.name] = final
        if bone.boneParent > -1:
            eb.parent = arm.edit_bones[model.bones[bone.boneParent].name]

    bpy.ops.object.mode_set(mode='POSE')
    for pb in skel_obj.pose.bones:
        pb.matrix_basis.identity()
        pb.matrix = bone_mats[pb.name]
    bpy.ops.pose.armature_apply()

    # Отображение костей
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=2)
    vis = bpy.context.active_object
    vis.name = vis.data.name = "semodel_bone_vis"
    vis.use_fake_user = True
    bpy.context.view_layer.active_layer_collection.collection.objects.unlink(vis)
    bpy.context.view_layer.objects.active = skel_obj

    # Подгон Tail для всех костей
    bpy.ops.object.mode_set(mode='EDIT', toggle=False)
    dims = [0, 0, 0]
    mins = [0, 0, 0]
    for b in arm.bones:
        for i in range(3):
            dims[i] = max(dims[i], b.head_local[i])
            mins[i] = min(mins[i], b.head_local[i])
    length = max(0.001, sum(dims[i] - mins[i] for i in range(3)) / 600)
    for bone in [arm.edit_bones[b.name] for b in model.bones]:
        bone.tail = bone.head + (bone.tail - bone.head).normalized() * length
        skel_obj.pose.bones[bone.name].custom_shape = vis

    # Назначаем модификатор Armature
    bpy.ops.object.mode_set(mode='OBJECT')
    for obj in mesh_objs:
        obj.parent = skel_obj
        mod = obj.modifiers.new("Armature Rig", "ARMATURE")
        mod.object = skel_obj
        mod.use_bone_envelopes = False
        mod.use_vertex_groups = True

    bpy.context.view_layer.update()
    bpy.ops.object.mode_set(mode='OBJECT')
    return True
