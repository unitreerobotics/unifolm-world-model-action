import time

import casadi
import meshcat.geometry as mg
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin
from pinocchio.visualize import MeshcatVisualizer

from unitree_deploy.utils.weighted_moving_filter import WeightedMovingFilter


class Z1_Arm_IK:
    def __init__(self, unit_test=False, visualization=False):
        np.set_printoptions(precision=5, suppress=True, linewidth=200)

        self.unit_test = unit_test
        self.visualization = visualization

        self.robot = pin.RobotWrapper.BuildFromURDF(
            "unitree_deploy/robot_devices/assets/z1/z1.urdf", "unitree_deploy/robot_devices/assets/z1/"
        )
        self.mixed_jointsToLockIDs = ["base_static_joint"]

        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration=np.array([0.0] * self.robot.model.nq),
        )

        self.reduced_robot.model.addFrame(
            pin.Frame(
                "ee",
                self.reduced_robot.model.getJointId("joint6"),
                pin.SE3(np.eye(3), np.array([0.15, 0, 0]).T),
                pin.FrameType.OP_FRAME,
            )
        )

        # for i in range(self.reduced_robot.model.nframes):
        #     frame = self.reduced_robot.model.frames[i]
        #     frame_id = self.reduced_robot.model.getFrameId(frame.name)
        #     print(f"Frame ID: {frame_id}, Name: {frame.name}")

        # Creating Casadi models and data for symbolic computing
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()

        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1)
        self.cTf = casadi.SX.sym("tf", 4, 4)

        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        self.EE_ID = self.reduced_robot.model.getFrameId("link06")
        self.translational_error = casadi.Function(
            "translational_error",
            [self.cq, self.cTf],
            [
                casadi.vertcat(
                    self.cdata.oMf[self.EE_ID].translation - self.cTf[:3, 3],
                )
            ],
        )
        self.rotational_error = casadi.Function(
            "rotational_error",
            [self.cq, self.cTf],
            [
                casadi.vertcat(
                    cpin.log3(self.cdata.oMf[self.EE_ID].rotation @ self.cTf[:3, :3].T),
                )
            ],
        )

        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        self.var_q_last = self.opti.parameter(self.reduced_robot.model.nq)  # for smooth
        self.param_tf = self.opti.parameter(4, 4)
        self.translational_cost = casadi.sumsqr(self.translational_error(self.var_q, self.param_tf))
        self.rotation_cost = casadi.sumsqr(self.rotational_error(self.var_q, self.param_tf))
        self.regularization_cost = casadi.sumsqr(self.var_q)
        self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)

        # Setting optimization constraints and goals
        self.opti.subject_to(
            self.opti.bounded(
                self.reduced_robot.model.lowerPositionLimit,
                self.var_q,
                self.reduced_robot.model.upperPositionLimit,
            )
        )
        self.opti.minimize(
            50 * self.translational_cost
            + self.rotation_cost
            + 0.02 * self.regularization_cost
            + 0.1 * self.smooth_cost
        )
        # self.opti.minimize(20 * self.cost + self.regularization_cost)

        opts = {
            "ipopt": {"print_level": 0, "max_iter": 50, "tol": 1e-6},
            "print_time": False,  # print or not
            "calc_lam_p": False,  # https://github.com/casadi/casadi/wiki/FAQ:-Why-am-I-getting-%22NaN-detected%22in-my-optimization%3F
        }
        self.opti.solver("ipopt", opts)

        self.init_data = np.zeros(self.reduced_robot.model.nq)
        self.smooth_filter = WeightedMovingFilter(np.array([0.4, 0.3, 0.2, 0.1]), 6)

        self.vis = None

        if self.visualization:
            # Initialize the Meshcat visualizer for visualization
            self.vis = MeshcatVisualizer(
                self.reduced_robot.model, self.reduced_robot.collision_model, self.reduced_robot.visual_model
            )
            self.vis.initViewer(open=True)
            self.vis.loadViewerModel("pinocchio")
            self.vis.displayFrames(True, frame_ids=[101, 102], axis_length=0.15, axis_width=5)
            self.vis.display(pin.neutral(self.reduced_robot.model))

            # Enable the display of end effector target frames with short axis lengths and greater width.
            frame_viz_names = ["ee_target"]
            frame_axis_positions = (
                np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1]])
                .astype(np.float32)
                .T
            )
            frame_axis_colors = (
                np.array([[1, 0, 0], [1, 0.6, 0], [0, 1, 0], [0.6, 1, 0], [0, 0, 1], [0, 0.6, 1]])
                .astype(np.float32)
                .T
            )
            axis_length = 0.1
            axis_width = 10
            for frame_viz_name in frame_viz_names:
                self.vis.viewer[frame_viz_name].set_object(
                    mg.LineSegments(
                        mg.PointsGeometry(
                            position=axis_length * frame_axis_positions,
                            color=frame_axis_colors,
                        ),
                        mg.LineBasicMaterial(
                            linewidth=axis_width,
                            vertexColors=True,
                        ),
                    )
                )

    def solve_ik(self, wrist, current_lr_arm_motor_q=None, current_lr_arm_motor_dq=None):
        if current_lr_arm_motor_q is not None:
            self.init_data = current_lr_arm_motor_q
        self.opti.set_initial(self.var_q, self.init_data)

        # left_wrist, right_wrist = self.scale_arms(left_wrist, right_wrist)
        if self.visualization:
            self.vis.viewer["ee_target"].set_transform(wrist)  # for visualization

        self.opti.set_value(self.param_tf, wrist)
        self.opti.set_value(self.var_q_last, self.init_data)  # for smooth

        try:
            self.opti.solve()
            # sol = self.opti.solve_limited()

            sol_q = self.opti.value(self.var_q)
            self.smooth_filter.add_data(sol_q)
            sol_q = self.smooth_filter.filtered_data
            
            v = current_lr_arm_motor_dq * 0.0 if current_lr_arm_motor_dq is not None else (sol_q - self.init_data) * 0.0
            self.init_data = sol_q
            sol_tauff = pin.rnea(
                self.reduced_robot.model,
                self.reduced_robot.data,
                sol_q,
                v,
                np.zeros(self.reduced_robot.model.nv),
            )

            if self.visualization:
                self.vis.display(sol_q)  # for visualization

            return sol_q, sol_tauff

        except Exception as e:
            print(f"ERROR in convergence, plotting debug info.{e}")

            sol_q = self.opti.debug.value(self.var_q)
            self.smooth_filter.add_data(sol_q)
            sol_q = self.smooth_filter.filtered_data

            v = current_lr_arm_motor_dq * 0.0 if current_lr_arm_motor_dq is not None else (sol_q - self.init_data) * 0.0

            self.init_data = sol_q
            sol_tauff = pin.rnea(
                self.reduced_robot.model,
                self.reduced_robot.data,
                sol_q,
                v,
                np.zeros(self.reduced_robot.model.nv),
            )
            if self.visualization:
                self.vis.display(sol_q)  # for visualization

            # return sol_q, sol_tauff
            return current_lr_arm_motor_q, np.zeros(self.reduced_robot.model.nv)


if __name__ == "__main__":
    arm_ik = Z1_Arm_IK(unit_test=True, visualization=True)

    # initial positon
    L_tf_target = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([0.25, 0, 0.2]),
    )

    rotation_speed = 0.02
    noise_amplitude_translation = 0.002
    noise_amplitude_rotation = 0.1

    user_input = input("Please enter the start signal (enter 's' to start the subsequent program):\n")
    if user_input.lower() == "s":
        step = 0
        while True:
            # Apply rotation noise with bias towards y and z axes
            rotation_noise_l = pin.Quaternion(
                np.cos(np.random.normal(0, noise_amplitude_rotation) / 2),
                0,
                np.random.normal(0, noise_amplitude_rotation / 2),
                0,
            ).normalized()  # y bias

            if step <= 120:
                angle = rotation_speed * step
                L_tf_target.rotation = (
                    rotation_noise_l * pin.Quaternion(np.cos(angle / 2), 0, np.sin(angle / 2), 0)
                ).toRotationMatrix()  # y axis
                L_tf_target.translation += np.array([0.001, 0.001, 0.001]) + np.random.normal(
                    0, noise_amplitude_translation, 3
                )
            else:
                angle = rotation_speed * (240 - step)
                L_tf_target.rotation = (
                    rotation_noise_l * pin.Quaternion(np.cos(angle / 2), 0, np.sin(angle / 2), 0)
                ).toRotationMatrix()  # y axis
                L_tf_target.translation -= np.array([0.001, 0.001, 0.001]) + np.random.normal(
                    0, noise_amplitude_translation, 3
                )

            sol_q, _ = arm_ik.solve_ik(L_tf_target.homogeneous)
            step += 1
            if step > 240:
                step = 0
            time.sleep(0.01)
