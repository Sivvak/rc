import mujoco
import matplotlib.pylab as plt
import numpy as np


def render_example():
    for i in range(10):
        xml = f"""
    <mujoco>
    <visual>
        <global offwidth="1280" offheight="1280"/>
    </visual>
    <worldbody>
        <light name="top" pos="0 0 1"/>
        <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
        <geom name="green_sphere" pos="{.2 + i / 10} .2 .2" size=".1" rgba="0 1 0 1"/>
    </worldbody>
    </mujoco>
    """
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, 1280, 1280)

        mujoco.mj_forward(model, data)
        renderer.update_scene(data)
        plt.imsave(f"frame_{i:03d}.png", renderer.render())
        renderer.close()


def generate_world_scene(x_car, y_car, car_rot, rader_rot):
    xml = f"""
<?xml version="1.0" ?>
<mujoco>
    <visual>
        <global offwidth="1280" offheight="1280" fovy="35"/>
    </visual>
    <worldbody>
         <body name="floor" pos="0 0 -0.1">
            <geom size="1.0 1.0 0.02" rgba="0.2 0.2 0.2 1" type="box"/>
        </body>
        <body name="x_arrow" pos="0.5 0 0">
            <geom size="0.5 0.01 0.01" rgba="1 0 0 0.5" type="box"/>
        </body>
        <body name="y_arrow" pos="0 0.5 0">
            <geom size="0.01 0.5 0.01" rgba="0 1 0 0.5" type="box"/>
        </body>
        <body name="z_arrow" pos="0 0 0.5">
            <geom size="0.01 0.01 0.5" rgba="0 0 1 0.5" type="box"/>
        </body>

        <body name="car" pos="{x_car} {y_car} 0.1" axisangle="0 0 1 0">
            <geom size="0.2 0.1 0.02" rgba="1 1 1 0.9" type="box"/>
        </body>
        <body name="wheel_1" pos="{x_car + 0.1} {y_car + 0.1} 0.1" axisangle="1 0 0 90">
            <geom size="0.07 0.01" rgba="1 1 1 0.9" type="cylinder"/>
        </body>
        <body name="wheel_2" pos="{x_car - 0.1} {y_car + 0.1} 0.1" axisangle="1 0 0 90">
            <geom size="0.07 0.01" rgba="1 1 1 0.9" type="cylinder"/>
        </body>
        <body name="wheel_3" pos="{x_car + 0.1} {y_car - 0.1} 0.1" axisangle="1 0 0 90">
            <geom size="0.07 0.01" rgba="1 1 1 0.9" type="cylinder"/>
        </body>
        <body name="wheel_4" pos="{x_car - 0.1} {y_car - 0.1} 0.1" axisangle="1 0 0 90">
            <geom size="0.07 0.01" rgba="1 1 1 0.9" type="cylinder"/>
        </body>

        <body name="radar_1" pos="{x_car} {y_car + 0.1} 0.2" axisangle="1 0 0 330">
            <geom size="0.01 0.01 0.1" rgba="1 1 1 1" type="box"/>
        </body>
        <body name="radar_2" pos="{x_car} {y_car + 0.15} 0.29" axisangle="1 0 0 330">
            <geom size="0.03 0.01" rgba="1 0 0 1" type="cylinder"/>
        </body>
    </worldbody>
</mujoco>"""

    return mujoco.MjModel.from_xml_string(xml)


def render_driving_car():
    center = (-1, 0)

    angles = np.linspace(3 / 2 * np.pi, 2 * np.pi, 10)

    for i, angle in enumerate(angles):
        x = -1 + np.cos(angle)
        y = np.sin(angle)

        model = generate_world_scene(x, y, 0, 0)

        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, 1280, 1280)

        mujoco.mj_forward(model, data)
        renderer.update_scene(data)
        plt.imsave(f"cm_frame_{i:03d}.png", renderer.render())
        renderer.close()


render_driving_car()
