import mujoco
import numpy as np
from mujoco import viewer
import time


def get_random_coord(lower=-2, upper=2):
    return np.random.uniform(lower, upper)


xml = f"""
<mujoco model="robot">
  <option timestep="0.001" gravity="0 0 -9.81"/>
  <compiler autolimits="true"/>
  <worldbody>
    <light name="top" pos="0 0 1"/>
    <body name="floor" pos="0 0 -0.1">
      <geom size="5.0 5.0 0.02" rgba="1 1 1 1" type="box"/>
    </body>
    <body name="box1" pos="0.3 0 0.5">
      <joint name="box1-joint" type="free"/>
      <geom size="0.1 0.1 0.1" rgba="1 0 0 1" type="box"/>
      <inertial pos="0 0 0" mass="0.001" diaginertia="0.001 0.001 0.001"/>
    </body>
    <body name="box2" pos="{get_random_coord()} {get_random_coord()} 0.5">
      <joint name="box2-joint" type="free"/>
      <geom size="0.1 0.1 0.1" rgba="0 0 1 1" type="box"/>
      <inertial pos="0 0 0" mass="0.001" diaginertia="0.001 0.001 0.001"/>
    </body>
    <body name="box3" pos="{get_random_coord()} {get_random_coord()} 0.5">
      <joint name="box3-joint" type="free"/>
      <geom size="0.1 0.1 0.1" rgba="0 0 1 1" type="box"/>
      <inertial pos="0 0 0" mass="0.001" diaginertia="0.001 0.001 0.001"/>
    </body>
    <body name="box4" pos="{get_random_coord()} {get_random_coord()} 0.5">
      <joint name="box4-joint" type="free"/>
      <geom size="0.1 0.1 0.1" rgba="0 0 1 1" type="box"/>
      <inertial pos="0 0 0" mass="0.001" diaginertia="0.001 0.001 0.001"/>
    </body>
    <body name="box5" pos="{get_random_coord()} {get_random_coord()} 0.5">
      <joint name="box5-joint" type="free"/>
      <geom size="0.1 0.1 0.1" rgba="0 0 1 1" type="box"/>
      <inertial pos="0 0 0" mass="0.001" diaginertia="0.001 0.001 0.001"/>
    </body>
    <body name="griper">
      <inertial pos="0 0 0" mass="0.001" diaginertia="0.001 0.001 0.001"/>
      <joint name="center_z" pos="0 0 0" damping="0.5" axis="0 0 1" type="slide" range="-1.1 1.1"/>
      <joint name="center_y" pos="0 0 0" damping="0.5" axis="0 1 0" type="slide" range="-1.1 1.1"/>
      <joint name="center_x" pos="0 0 0" damping="0.5" axis="1 0 0" type="slide" range="-1.1 1.1"/>
      <joint name="roll" pos="0 0 0" damping="0.5" axis="0 0 1" type="hinge" range="-360 360"/>
      <geom size="0.1 0.05 0.01" type="box" rgba="0.356 0.361 0.376 1"/>
    </body>
  </worldbody>
  <actuator>
    <position name="ac_z" joint="center_z"/>
    <position name="ac_y" joint="center_y" />
    <position name="ac_x" joint="center_x" />
    <position name="ac_r" joint="roll" ctrlrange="-3 3" />
  </actuator>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

viewer = viewer.launch_passive(model, data)

step = 0
while True:
    step_start = time.time()
    step += 1

    # if step % 1000 == 0:
    #     data.actuator("ac_x").ctrl = np.random.uniform(-1.1, 1.1)
    #     data.actuator("ac_y").ctrl = np.random.uniform(-1.1, 1.1)
    #     data.actuator("ac_z").ctrl = np.random.uniform(-1.1, 1.1)
    #     data.actuator("ac_r").ctrl = np.random.uniform(-3, 3)
    #     print(data.joint("box1-joint").qpos)
    mujoco.mj_step(model, data)
    viewer.sync()
    time_until_next_step = model.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
        time.sleep(time_until_next_step)
