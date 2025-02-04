import time

import mujoco
import mujoco.viewer
import numpy as np


def start_sim():
    global model, data, viewer_window, renderer

    xml_path = "./manipulator3d.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    renderer = mujoco.Renderer(model, height=480, width=640)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    viewer_window = mujoco.viewer.launch_passive(model, data)

    # Initial configuration to avoid singularity
    data.ctrl[0] = -0.2
    data.ctrl[1] = 0.2
    data.ctrl[2] = 0.2

    for _ in range(100):
        mujoco.mj_step(model, data)
        renderer.update_scene(data)
        viewer_window.sync()


def go_to(qpos):
    """Move the robot to desired joint positions using position control."""
    data.ctrl = qpos
    for _ in range(100):
        mujoco.mj_step(model, data)
        renderer.update_scene(data)
        viewer_window.sync()
        # time.sleep(0.02)  # Added delay to see motion better
        if np.linalg.norm(data.qpos - qpos) < 0.0001:
            break


def forward_kinematics(q):
    """Calculate end-effector position using forward kinematics based on manipulator3d.xml."""
    l1, l2, l3 = 1, 1, 1
    base_height = 0.6

    theta = q[0]  # Rotation around Z
    phi1, phi2 = q[1], q[2]  # Joint angles

    r = l2 * np.cos(phi1) + l3 * np.cos(phi1 + phi2)
    h = base_height + l1 + l2 * np.sin(phi1) + l3 * np.sin(phi1 + phi2)

    x = r * np.sin(theta)
    y = r * np.cos(theta)
    z = h

    return np.array([x, y, z])


def move_to_position(target_pos, max_iterations=30000):
    """Move the end effector to a target position using inverse kinematics."""
    step_size = 1e-3

    for _ in range(max_iterations):
        # Get current end effector position using forward kinematics
        current_pos = data.site_xpos[0]

        # Calculate position error
        error = target_pos - current_pos

        # Compute Jacobian (3x3 for 3D space and 3 joints)
        jacp = np.zeros((3, 3))
        mujoco.mj_jac(model, data, jacp, None, current_pos, 3)

        # Compute joint angle updates using pseudoinverse Jacobian
        try:
            Jinv = np.linalg.pinv(jacp)
            dq = Jinv @ error

            new_qpos = data.qpos + dq * step_size
            go_to(new_qpos)

        except np.linalg.LinAlgError:
            print("Singularity encountered")
            return False

    print(f"Final position: {data.site_xpos[0]}")


def main():
    start_sim()

    # Example target position [x, y, z]
    target_position = np.array([0, -1.0, 1.1])
    move_to_position(target_position)

    while viewer_window.is_running():
        viewer_window.sync()
        time.sleep(0.01)


if __name__ == "__main__":
    main()
