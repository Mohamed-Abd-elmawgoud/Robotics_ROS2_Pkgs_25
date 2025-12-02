#!/usr/bin/env python3

import numpy as np

class ForwardKinematicsNode:
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


def extract_pitch_method1(R):
    """ZYX convention: pitch = asin(-R[2,0])"""
    sin_pitch = -R[2, 0]
    sin_pitch = np.clip(sin_pitch, -1.0, 1.0)
    return np.arcsin(sin_pitch)


def extract_pitch_method2(R):
    """Alternative: pitch = atan2(-R[2,0], sqrt(R[0,0]^2 + R[1,0]^2))"""
    x = R[0, 0]
    y = R[1, 0]
    z = R[2, 0]
    return np.arctan2(-z, np.sqrt(x*x + y*y))


def extract_pitch_method3(R):
    """Using approach vector: pitch = atan2(-R[2,2], sqrt(R[0,2]^2 + R[1,2]^2))"""
    return np.arctan2(-R[2, 2], np.sqrt(R[0, 2]**2 + R[1, 2]**2))


def extract_pitch_method4(R):
    """Using Z-axis of end effector"""
    z_axis = R[:, 2]  # Third column
    # Pitch is angle between Z-axis and XY plane
    return np.arctan2(-z_axis[2], np.sqrt(z_axis[0]**2 + z_axis[1]**2))


def main():
    fk = ForwardKinematicsNode()
    
    print("="*70)
    print("PITCH DIAGNOSTIC TOOL")
    print("="*70)
    print("\nTesting different joint configurations to understand pitch behavior\n")
    
    # Test configurations
    test_configs = [
        ([0, 0, 0, 0, 0], "Home position (all zeros)"),
        ([0, np.pi/4, 0, 0, 0], "q2 = 45° (elbow up)"),
        ([0, -np.pi/4, 0, 0, 0], "q2 = -45° (elbow down)"),
        ([0, 0, 0, np.pi/4, 0], "q4 = 45° (wrist up)"),
        ([0, 0, 0, -np.pi/4, 0], "q4 = -45° (wrist down)"),
        ([0, np.pi/4, 0, np.pi/4, 0], "q2=45°, q4=45° (both up)"),
        ([0, np.pi/4, 0, -np.pi/4, 0], "q2=45°, q4=-45°"),
    ]
    
    for q, description in test_configs:
        T = fk.fwd_kinematics(q)
        R = T[:3, :3]
        
        print(f"\n{description}")
        print(f"Joint angles: q = {np.round(np.rad2deg(q), 1)}°")
        
        # Extract pitch using different methods
        pitch1 = extract_pitch_method1(R)
        pitch2 = extract_pitch_method2(R)
        pitch3 = extract_pitch_method3(R)
        pitch4 = extract_pitch_method4(R)
        
        print(f"\nPitch extraction methods:")
        print(f"  Method 1 (ZYX: asin(-R[2,0])): {np.rad2deg(pitch1):7.2f}°  ({pitch1:.4f} rad)")
        print(f"  Method 2 (atan2 variant):       {np.rad2deg(pitch2):7.2f}°  ({pitch2:.4f} rad)")
        print(f"  Method 3 (R[2,2] based):        {np.rad2deg(pitch3):7.2f}°  ({pitch3:.4f} rad)")
        print(f"  Method 4 (Z-axis approach):     {np.rad2deg(pitch4):7.2f}°  ({pitch4:.4f} rad)")
        
        # Show the approach vector (Z direction of end effector)
        z_axis = R[:, 2]
        print(f"\nEnd effector Z-axis (approach): [{z_axis[0]:7.3f}, {z_axis[1]:7.3f}, {z_axis[2]:7.3f}]")
        print(f"Position: [{T[0,3]:7.1f}, {T[1,3]:7.1f}, {T[2,3]:7.1f}] mm")
    
    print("\n" + "="*70)
    print("\nNow try YOUR specific case:")
    print("Enter joint angles that make the end effector point UP")
    print("(e.g., when you visually see it pointing upward)")
    
    try:
        q_up = input("\nEnter 5 joint angles in DEGREES (space separated): ")
        q_up = [np.deg2rad(float(x)) for x in q_up.split()]
        
        if len(q_up) == 5:
            T = fk.fwd_kinematics(q_up)
            R = T[:3, :3]
            
            print(f"\nFor upward-pointing configuration:")
            print(f"  Method 1: {np.rad2deg(extract_pitch_method1(R)):7.2f}°")
            print(f"  Method 2: {np.rad2deg(extract_pitch_method2(R)):7.2f}°")
            print(f"  Method 3: {np.rad2deg(extract_pitch_method3(R)):7.2f}°")
            print(f"  Method 4: {np.rad2deg(extract_pitch_method4(R)):7.2f}°")
            
            z_axis = R[:, 2]
            print(f"\n  Z-axis: [{z_axis[0]:7.3f}, {z_axis[1]:7.3f}, {z_axis[2]:7.3f}]")
            print(f"\n  If Z-axis[2] is positive → pointing UP")
            print(f"  If Z-axis[2] is negative → pointing DOWN")
    except:
        print("Invalid input, skipping custom test")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main()