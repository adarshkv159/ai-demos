import json
import math

def calculate_angle(a, b, c):
    """
    Compute the angle between three points (a-b-c) in radians.
    'b' is the vertex point.
    """
    ab = (b[0] - a[0], b[1] - a[1])  # Vector from point a to b
    bc = (c[0] - b[0], c[1] - b[1])  # Vector from point b to c

    dot = ab[0] * bc[0] + ab[1] * bc[1]
    mag_ab = math.sqrt(ab[0]**2 + ab[1]**2)
    mag_bc = math.sqrt(bc[0]**2 + bc[1]**2)

    if mag_ab == 0 or mag_bc == 0:
        return 0.0

    cos_theta = dot / (mag_ab * mag_bc)
    angle = math.acos(max(-1, min(1, cos_theta)))  # Clamp for safety

    # Use cross product to determine sign
    cross = ab[0] * bc[1] - ab[1] * bc[0]
    if cross < 0:
        angle = -angle

    return angle

def map_angle_to_position(angle, angle_min, angle_max, pos_min, pos_max):
    """
    Map angle (in radians) to servo motor position range.
    """
    angle = max(min(angle, angle_max), angle_min)  # Clamp to valid range
    return int(pos_min + (angle - angle_min) * (pos_max - pos_min) / (angle_max - angle_min))

def generate_arm_command(keypoints):
    """
    Convert MoveNet keypoints to robotic arm joint positions.
    Expects the following keypoints in input dictionary:
      - 'left_shoulder', 'left_elbow', 'left_wrist', 'left_hip'
    """
    left_shoulder = keypoints["left_shoulder"]
    left_elbow = keypoints["left_elbow"]
    left_wrist = keypoints["left_wrist"]
    left_hip = keypoints["left_hip"]

    # Compute angles for joints
    angle_joint2 = calculate_angle(left_hip, left_shoulder, left_elbow)   # Shoulder
    angle_joint3 = calculate_angle(left_shoulder, left_elbow, left_wrist) # Elbow

    # Map angles to servo positions
    joint2_pos = map_angle_to_position(angle_joint2, 0.4, 2.5, 2213, 310)  # Adjusted for full shoulder range
    joint2_pos = max(310, min(2213, joint2_pos))  # Clamp for safety

    joint3_pos = map_angle_to_position(angle_joint3, -2.438, -0.336, 3636, 2266)  # Elbow mapping

    # Default/fixed values
    joint1 = 2047  # Base rotation
    joint4 = int(2047 + (-0.738 / math.pi * 2048))  # Wrist bend
    joint5 = 2047  # Wrist rotation

    # Final command data
    data = {
        'T': 3,
        'P1': joint1,
        'P2': joint2_pos,
        'P3': joint3_pos,
        'P4': joint4,
        'P5': joint5,
        'S1': 0, 'S2': 0, 'S3': 0, 'S4': 0, 'S5': 0,
        'A1': 60, 'A2': 60, 'A3': 60, 'A4': 60, 'A5': 60
    }

    # Debug prints
    print("\n[INFO] Computed Angles:")
    print(f"  Joint 2 (Shoulder): {angle_joint2:.4f} rad → P2 = {joint2_pos}")
    print(f"  Joint 3 (Elbow):    {angle_joint3:.4f} rad → P3 = {joint3_pos}")

    print("\n[INFO] Robotic Arm Command (JSON):")
    print(json.dumps(data, indent=2))

    return json.dumps(data)
