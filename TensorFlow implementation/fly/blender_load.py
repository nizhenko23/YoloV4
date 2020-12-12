import bpy, math
import numpy
from numpy import genfromtxt
import bpy_extras
from bpy_extras.object_utils import world_to_camera_view

scene = bpy.context.scene

scene = bpy.context.scene

trajectory = genfromtxt('D:\\!IAN_WORK\\NN_vision\\trajectory.csv', delimiter=',')

ob1 = bpy.data.objects["ASP2"]
ob2 = bpy.data.objects["Box"]
frame_number = 40
ob1.animation_data_clear()
ob2.animation_data_clear()

positions = trajectory[:, :3]
angles = trajectory[:, 3:]

for angle_i, i in enumerate(positions):

    bpy.context.scene.frame_set(frame_number)

    ob1.location = i
    ob1.rotation_euler = angles[angle_i]
    ob1.keyframe_insert(data_path='location', index=-1)
    ob1.keyframe_insert(data_path='rotation_euler', index=-1)

    ob2.location = i
    ob2.rotation_euler = angles[angle_i]
    ob2.keyframe_insert(data_path='location', index=-1)
    ob2.keyframe_insert(data_path='rotation_euler', index=-1)

    frame_number += 0.1
    

file = open("D:\\!IAN_WORK\\NN_vision\\trajectory.txt", 'w')
loc = ob1.location

for frames in range(scene.frame_start, scene.frame_end + 1):
    scene.frame_set(frames)
    file.write(str(loc.x) + ', ' + str(loc.y) + ', ' + str(loc.z) + '\n')

file.close()

render = scene.render
res_x = render.resolution_x
res_y = render.resolution_y

obj = bpy.data.objects['ASP2']
cam = bpy.data.objects['Camera']

verts = (vert.co for vert in obj.data.vertices)
coords_2d = [world_to_camera_view(scene, cam, coord) for coord in verts]

rnd = lambda i: round(i)

print('x,y')
for x, y, distance_to_lens in coords_2d:
    print("{},{}".format(rnd(res_x*x), rnd(res_y*y)))
    

rnd3 = lambda i: round(i, 3)

limit_finder = lambda f: f(coords_2d, key=lambda i: i[2])[2]
limits = limit_finder(min), limit_finder(max)
limits = [rnd3(d) for d in limits]

print('min, max\n{},{}'.format(*limits))

print('x,y,d')
for x, y, d in coords_2d:
    print("{},{},{}".format(rnd(res_x*x), rnd(res_y*y), rnd3(d)))
