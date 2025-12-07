#!/usr/bin/env python3
import numpy as np

# Standalone FK (no ROS)
class FK:
    def __init__(self):
        self.l1 = 41.05
        self.l2 = 139.93
        self.l3 = 132.9
        self.l4 = 52.3
        self.l5 = 37.76
    
    def fwd_kinematics(self, q):
        l1, l2, l3, l4, l5 = self.l1, self.l2, self.l3, self.l4, self.l5
        theta = [-np.pi/2+q[0], -np.pi/2+q[1], q[2], -np.pi/2+q[3], np.pi/2+q[4]]
        d = [-l1, 0, 0, 0, -l4-l5]
        a = [0, l2, l3, 0, 0]
        alpha = [np.pi/2, np.pi, np.pi, np.pi/2, 0]
        T = np.eye(4)
        for i in range(5):
            ct = np.cos(theta[i])
            st = np.sin(theta[i])
            ca = np.cos(alpha[i])
            sa = np.sin(alpha[i])
            T_i = np.array([
                [ct, -st*ca, st*sa, a[i]*ct],
                [st, ct*ca, -ct*sa, a[i]*st],
                [0, sa, ca, d[i]],
                [0, 0, 0, 1]
            ])
            T = T @ T_i
        return T

# Test
fk = FK()

print("="*60)
print("PITCH EXTRACTION TEST")
print("="*60)

test_configs = [
    ("Home (pointing down)", [0, 0, 0, 0, 0]),
    ("q2=30", [0, np.deg2rad(30), 0, 0, 0]),
    ("q3=45", [0, 0, np.deg2rad(45), 0, 0]),
    ("q4=45", [0, 0, 0, np.deg2rad(45), 0]),
]

for name, q in test_configs:
    T = fk.fwd_kinematics(q)
    R = T[:3, :3]
    pos = T[:3, 3]
    
    print(f"\n{name}: q = {np.rad2deg(q)}")
    print(f"  Position: {np.round(pos, 1)}")
    print(f"  Rotation matrix:")
    print(f"    {np.round(R[0], 3)}")
    print(f"    {np.round(R[1], 3)}")
    print(f"    {np.round(R[2], 3)}")
    
    # Tool direction (last column of R)
    tool_z = R[:, 2]
    print(f"  Tool Z-axis direction: {np.round(tool_z, 3)}")
    
    # Extract pitch from rotation matrix
    # Method 1: atan2 from R[2,0] (assuming standard orientation)
    pitch1 = np.arctan2(-R[2, 0], R[2, 2])
    print(f"  Pitch (method 1): {np.rad2deg(pitch1):.1f}°")
    
    # Method 2: from tool direction
    pitch2 = np.arctan2(tool_z[0], -tool_z[2])
    print(f"  Pitch (method 2): {np.rad2deg(pitch2):.1f}°")