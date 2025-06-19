bl_info = {
    "name" : "Build Skeleton",
    "description" : "Build skeleton from keypoint data",
    "author" : "Max Urbany",
    "version" : (0, 0, 1),
    "blender" : (2, 80, 0),
    "location" : "3D Viewport > Sidebar > My Custom Panel category",
    "category" : "Development"
}

import bpy
from bpy.types import Operator
from bpy.types import Panel
import numpy as np
from mathutils import Matrix, Vector, Quaternion

skeleton_map = [
    [0, 7],
    [0, 1],
    [1, 2],
    [2, 3],
    [0, 4],
    [4, 5],
    [5, 6],
    [7, 8],
    [8, 9],
    [9, 10],
    [10, 10],
    [8, 11],
    [11, 12],
    [12, 13],
    [8, 14],
    [14, 15],
    [15, 16]
]

joint_names = [
    'hip',
    'hip.R',
    'knee.R',
    'ankle.R',
    'hip.L',
    'knee.L',
    'ankle.L',
    'belly',
    'neck',
    'nose',
    'head',
    'shoulder.L',
    'elbow.L',
    'wrist.L',
    'shoulder.R',
    'elbow.R',
    'wrist.R',
]

def convert_keypoint(kp):
    return Vector((kp[0], kp[2], -kp[1]))

def create_skeleton(keypoints, skeleton_map, joint_names):
    bpy.ops.object.mode_set(mode='OBJECT')
    if "Armature" not in bpy.data.objects:
        bpy.ops.object.armature_add(enter_editmode=False, align='WORLD', location=(0, 0, 0))
    
    armature = bpy.context.object
    armature.name = "MyArmature"
    
    bpy.ops.object.mode_set(mode='EDIT')
    
    armature_data = armature.data
    armature_data.edit_bones.remove(armature_data.edit_bones['Bone'])
    
    for i in range(17):
        if i != 10: #Omitting head joint because it points to nothing
            bone_name = f"{joint_names[i]}"
            bone = armature_data.edit_bones.new(bone_name)
            
            head_kp = keypoints[0][skeleton_map[i][0]]
            bone.head = (head_kp[0], head_kp[2], -head_kp[1])
            
            tail_kp = keypoints[0][skeleton_map[i][1]]
            bone.tail = (tail_kp[0], tail_kp[2], -tail_kp[1])
        
    return armature

def key_frame(keypoints, skeleton_map, joint_names, armature, frame):
    for i in range(17):
        if i != 10:
            head = skeleton_map[i][0]
            tail = skeleton_map[i][1]
            
            bone = armature.pose.bones[joint_names[i]]
            bone.location = bone.matrix.inverted() @ convert_keypoint(keypoints[frame][head])
            bone.keyframe_insert(data_path='location', frame=frame)
            
            
            l, r, s = bone.matrix.decompose()
            bone_v = convert_keypoint(keypoints[frame][tail]) - convert_keypoint(keypoints[frame][head])
            rot_delta = bone.y_axis.rotation_difference(bone_v)
            r.rotate(rot_delta)
            bone.matrix = Matrix.LocRotScale(l, r, s)
            bone.keyframe_insert(data_path='rotation_quaternion', frame=frame)

def animate(keypoints, skeleton_map, joint_names, armature):
    bpy.ops.object.mode_set(mode='POSE')
    frame_range = len(keypoints)
    for frame in range(frame_range):
        key_frame(keypoints, skeleton_map, joint_names, armature, frame)


class Build_Skel_operator(Operator):
    """ tooltip goes here """
    bl_idname = "demo.operator"
    bl_label = "Generates skeleton"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        keypoints = np.load(context.scene.kp_path)
        armature = create_skeleton(keypoints, skeleton_map, joint_names)
        animate(keypoints, skeleton_map, joint_names, armature)

        return {'FINISHED'}


class Build_Skel_sidebar(Panel):
    bl_label = "Skeleton Builder"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "AI Skelton"

    def draw(self, context):
        col = self.layout.column(align=True)
        prop = col.operator(Build_Skel_operator.bl_idname, text="Generate Skeleton")

        col.prop(context.scene, "kp_path")

 
classes = [
    Build_Skel_operator,
    Build_Skel_sidebar,
]

def register():
    for c in classes:
        bpy.utils.register_class(c)
        bpy.types.Scene.kp_path = bpy.props.StringProperty(
            name='<path_to_keypoints>.npy',
            default = ''
        )

def unregister():
    del bpy.types.Scene.kp_path
    for c in classes:
        bpy.utils.unregister_class(c)


if __name__ == '__main__':
    register()