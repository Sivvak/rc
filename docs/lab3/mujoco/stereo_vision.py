import mujoco
from PIL import Image
import cv2
import numpy as np

# camera paramas
resolution = (1280, 1280)
ox = resolution[0] / 2
oy = resolution[1] / 2

fovy = np.pi / 2
f = (ox / np.tan(fovy / 2)).item()


def detect_contour(img, lower, upper):
    hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    kernel = np.ones((5, 5), "uint8")

    mask = cv2.inRange(hsvImg, lower, upper)
    mask = cv2.dilate(mask, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return cv2.boundingRect(contours[0])


def to_3D(x1, y1, w1, h1, x2, y2, w2, h2, z_offset):
    x_3D = baseline * (x1 + w1 / 2 - ox) / (x1 + w1 / 2 - (x2 + w2 / 2))
    y_3D = baseline * f * (y1 + h1 / 2 - oy) / (f * (x1 + w1 / 2 - (x2 + w2 / 2)))
    z_3D = baseline * f / (x1 + w1 / 2 - (x2 + w2 / 2))

    return [x_3D, y_3D, z_3D - z_offset / 2]


# distance between two cameras
baseline = 0.1

sphere_size = 0.01
box1_size = 0.05
box2_size = 0.02

img1 = cv2.imread("rc/docs/lab3/mujoco/left.png")
img2 = cv2.imread("rc/docs/lab3/mujoco/right.png")

# TODO: find the positions of the objects and reconstruct the images
green_lower = np.array([36, 25, 25], np.uint8)
green_upper = np.array([70, 255, 255], np.uint8)

red_lower = np.array([136, 87, 111], np.uint8)
red_upper = np.array([180, 255, 255], np.uint8)

blue_lower = np.array([94, 80, 2], np.uint8)
blue_upper = np.array([120, 255, 255], np.uint8)

# red sphere
# x1, y1, w1, h1 = detect_contour(img1, red_lower, red_upper)
# x2, y2, w2, h2 = detect_contour(img2, red_lower, red_upper)
# pos_sphere = to_3D(x1, y1, w1, h1, x2, y2, w2, h2, sphere_size)

# green box
x1, y1, w1, h1 = detect_contour(img1, green_lower, green_upper)
x2, y2, w2, h2 = detect_contour(img2, green_lower, green_upper)
pos_box1 = to_3D(x1, y1, w1, h1, x2, y2, w2, h2, box1_size)

# blue box
x1, y1, w1, h1 = detect_contour(img1, blue_lower, blue_upper)
x2, y2, w2, h2 = detect_contour(img2, blue_lower, blue_upper)
pos_box2 = to_3D(x1, y1, w1, h1, x2, y2, w2, h2, box2_size)

xml_string = f"""\
<mujoco model="simple_scene">
      <visual>
     <global offwidth="{resolution[0]}" offheight="{resolution[1]}"/>
  </visual>
    <asset>
        <texture name="plane_texture" type="2d" builtin="checker" rgb1="0.2 0.2 0.2" rgb2="0.3 0.3 0.3" width="512" height="512"/>
        <material name="plane_material" texture="plane_texture" texrepeat="5 5" reflectance="0."/>
    </asset>

    <worldbody>
        <geom type="plane" size="1 1 0.1" material="plane_material"/>

        <body pos="{pos_box1[0]} {pos_box1[1]} {pos_box1[2]}">
            <geom type="box" size="0.05 0.05 0.05" rgba="0 1 0 1"/>
        </body>

        <body pos="{pos_box2[0]} {pos_box2[1]} {pos_box2[2]} ">
            <geom type="box" size="0.02 0.02 0.02" rgba="0 0 1 1"/>
        </body>

        <!-- Cameras -->
        <camera name="camera1" pos="0 0 1" fovy="90"/>
        <camera name="camera2" pos="{baseline} 0 1" fovy="90"/>
    </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml_string)
renderer = mujoco.Renderer(model, resolution[0], resolution[1])

data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

renderer.update_scene(data, camera="camera1")
img1_r = Image.fromarray(renderer.render())
img1_r.save("reconstruct_left.png")

renderer.update_scene(data, camera="camera2")
img2_r = Image.fromarray(renderer.render())
img2_r.save("reconstruct_right.png")
