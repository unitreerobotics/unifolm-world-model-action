# noqa: N815
from enum import IntEnum


# ==================g1========================
class G1_29_JointArmIndex(IntEnum):
    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitch = 20
    kLeftWristyaw = 21

    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitch = 27
    kRightWristYaw = 28


class G1_29_JointIndex(IntEnum):
    # Left leg
    kLeftHipPitch = 0
    kLeftHipRoll = 1
    kLeftHipYaw = 2
    kLeftKnee = 3
    kLeftAnklePitch = 4
    kLeftAnkleRoll = 5

    # Right leg
    kRightHipPitch = 6
    kRightHipRoll = 7
    kRightHipYaw = 8
    kRightKnee = 9
    kRightAnklePitch = 10
    kRightAnkleRoll = 11

    kWaistYaw = 12
    kWaistRoll = 13
    kWaistPitch = 14

    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitch = 20
    kLeftWristyaw = 21

    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitch = 27
    kRightWristYaw = 28

    # not used
    kNotUsedJoint0 = 29
    kNotUsedJoint1 = 30
    kNotUsedJoint2 = 31
    kNotUsedJoint3 = 32
    kNotUsedJoint4 = 33
    kNotUsedJoint5 = 34


# ==========================================


# ==================z1========================
class Z1ArmJointIndex(IntEnum):
    WAIST = 0
    SHOULDER = 1
    ELBOW = 2
    FOREARM_ROLL = 3
    WRIST_ANGLE = 4
    WRIST_ROTATE = 5


class Z1_12_JointArmIndex(IntEnum):
    # Left arm
    kLeftWaist = 0
    kLeftShoulder = 1
    kLeftElbow = 2
    kLeftForearmRoll = 3
    kLeftWristAngle = 4
    kLeftWristRotate = 5

    # Right arm
    kRightWaist = 6
    kRightShoulder = 7
    kRightElbow = 8
    kRightForearmRoll = 9
    kRightWristAngle = 10
    kRightWristRotate = 11


class Z1GripperArmJointIndex(IntEnum):
    WAIST = 0
    SHOULDER = 1
    ELBOW = 2
    FOREARM_ROLL = 3
    WRIST_ANGLE = 4
    WRIST_ROTATE = 5
    GRIPPER = 6


class Gripper_Sigle_JointIndex(IntEnum):
    kGripper = 0


# ==========================================
