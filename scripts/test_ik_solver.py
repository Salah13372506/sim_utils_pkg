#!/usr/bin/env python3
"""
Standalone test script for IK solver validation.
Tests FK/IK round-trip accuracy without ROS dependencies.

Run: python3 test_ik_solver.py
"""

import sys
import numpy as np

# Add package to path
sys.path.insert(0, '/home/thomas/sim_ws/src/sim_utils_pkg')

from sim_utils_pkg.kinematics.ik_solver import IKSolver, Pose, IKResult


def test_fk_basic():
    """Test that FK works for known configurations."""
    print("\n" + "="*60)
    print("TEST 1: Forward Kinematics - Basic")
    print("="*60)

    urdf_path = '/home/thomas/sim_ws/src/sim_utils_pkg/config/ur5e.urdf'
    solver = IKSolver.from_urdf_file(urdf_path, 'base_link', 'tool0')

    print(f"\nSolver initialized:")
    print(solver)

    # Test home position (all zeros)
    print("\n--- Home position (all joints = 0) ---")
    joints_home = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    pose_home = solver.forward(joints_home)

    if pose_home:
        print(f"Position: x={pose_home.position[0]:.4f}, y={pose_home.position[1]:.4f}, z={pose_home.position[2]:.4f}")
        print(f"Orientation (quat): {pose_home.orientation}")
    else:
        print("FK FAILED!")
        return False

    # Test with shoulder rotated 90 degrees
    print("\n--- Shoulder pan = 90° ---")
    joints_rotated = [np.pi/2, 0.0, 0.0, 0.0, 0.0, 0.0]
    pose_rotated = solver.forward(joints_rotated)

    if pose_rotated:
        print(f"Position: x={pose_rotated.position[0]:.4f}, y={pose_rotated.position[1]:.4f}, z={pose_rotated.position[2]:.4f}")
    else:
        print("FK FAILED!")
        return False

    print("\n[PASS] FK basic test")
    return True


def test_ik_round_trip():
    """Test IK by computing FK -> IK -> FK and comparing."""
    print("\n" + "="*60)
    print("TEST 2: IK Round-Trip Validation")
    print("="*60)

    urdf_path = '/home/thomas/sim_ws/src/sim_utils_pkg/config/ur5e.urdf'
    solver = IKSolver.from_urdf_file(urdf_path, 'base_link', 'tool0')

    # Test configurations
    test_configs = [
        ("Home", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ("Arm up", [0.0, -np.pi/2, 0.0, 0.0, 0.0, 0.0]),
        ("Reach forward", [0.0, -np.pi/4, np.pi/4, -np.pi/2, 0.0, 0.0]),
        ("Side reach", [np.pi/2, -np.pi/4, np.pi/3, -np.pi/4, np.pi/2, 0.0]),
    ]

    all_passed = True
    position_tolerance = 0.001  # 1mm
    orientation_tolerance = 0.01  # ~0.5 degrees

    for name, original_joints in test_configs:
        print(f"\n--- Testing: {name} ---")
        print(f"Original joints (deg): {[f'{np.degrees(j):.1f}' for j in original_joints]}")

        # Step 1: FK to get target pose
        target_pose = solver.forward(original_joints)
        if target_pose is None:
            print("[FAIL] FK failed")
            all_passed = False
            continue

        print(f"Target position: ({target_pose.position[0]:.4f}, {target_pose.position[1]:.4f}, {target_pose.position[2]:.4f})")

        # Step 2: IK to get joint solution
        # Use slightly perturbed initial guess to make it realistic
        initial_guess = [j + 0.1 for j in original_joints]
        result, solved_joints = solver.solve(target_pose, initial_guess)

        if result != IKResult.SUCCESS:
            print(f"[FAIL] IK failed with: {result}")
            all_passed = False
            continue

        print(f"Solved joints (deg): {[f'{np.degrees(j):.1f}' for j in solved_joints]}")

        # Step 3: FK with solved joints to verify
        achieved_pose = solver.forward(solved_joints.tolist())

        # Compute errors
        pos_error = solver.compute_position_error(target_pose, solved_joints.tolist())
        orient_error = solver.compute_orientation_error(target_pose, solved_joints.tolist())

        print(f"Position error: {pos_error*1000:.4f} mm")
        print(f"Orientation error: {np.degrees(orient_error):.4f} deg")

        if pos_error > position_tolerance:
            print(f"[FAIL] Position error too large (>{position_tolerance*1000}mm)")
            all_passed = False
        elif orient_error > orientation_tolerance:
            print(f"[FAIL] Orientation error too large (>{np.degrees(orientation_tolerance)}deg)")
            all_passed = False
        else:
            print("[PASS]")

    if all_passed:
        print("\n[PASS] All round-trip tests passed!")
    else:
        print("\n[FAIL] Some tests failed")

    return all_passed


def test_ik_from_pose():
    """Test IK from Cartesian poses derived from known configurations."""
    print("\n" + "="*60)
    print("TEST 3: IK from Cartesian Poses")
    print("="*60)

    urdf_path = '/home/thomas/sim_ws/src/sim_utils_pkg/config/ur5e.urdf'
    solver = IKSolver.from_urdf_file(urdf_path, 'base_link', 'tool0')

    # Generate test poses from known joint configurations
    # This ensures the poses are reachable
    test_configs = [
        ("Config A", [0.5, -1.0, 1.0, -0.5, 1.0, 0.2]),
        ("Config B", [-0.3, -1.2, 0.8, -1.0, -0.5, 0.5]),
        ("Config C", [1.0, -0.8, 1.2, -0.4, 0.8, -0.3]),
    ]

    all_passed = True

    for name, original_joints in test_configs:
        print(f"\n--- Testing: {name} ---")

        # Get pose from FK
        target_pose = solver.forward(original_joints)
        if target_pose is None:
            print("[FAIL] Could not compute FK for test config")
            all_passed = False
            continue

        print(f"Target: pos=({target_pose.position[0]:.3f}, {target_pose.position[1]:.3f}, {target_pose.position[2]:.3f})")

        # Use a different initial guess (not the original joints)
        # This tests that IK can find the solution from a different starting point
        initial_guess = [0.0] * 6
        result, joints = solver.solve(target_pose, initial_guess)

        if result != IKResult.SUCCESS:
            # Try with a better initial guess
            print("  Retrying with better initial guess...")
            initial_guess = [j * 0.5 for j in original_joints]  # Halfway guess
            result, joints = solver.solve(target_pose, initial_guess)

        if result != IKResult.SUCCESS:
            print(f"[FAIL] IK failed: {result}")
            all_passed = False
            continue

        print(f"Solution (deg): {[f'{np.degrees(j):.1f}' for j in joints]}")

        # Verify
        pos_error = solver.compute_position_error(target_pose, joints.tolist())
        orient_error = solver.compute_orientation_error(target_pose, joints.tolist())

        print(f"Position error: {pos_error*1000:.4f} mm")
        print(f"Orientation error: {np.degrees(orient_error):.4f} deg")

        if pos_error < 0.001 and orient_error < 0.01:
            print("[PASS]")
        else:
            print("[FAIL] Error too large")
            all_passed = False

    return all_passed


def test_unreachable_pose():
    """Test that IK fails gracefully for unreachable poses."""
    print("\n" + "="*60)
    print("TEST 4: Unreachable Pose Handling")
    print("="*60)

    urdf_path = '/home/thomas/sim_ws/src/sim_utils_pkg/config/ur5e.urdf'
    solver = IKSolver.from_urdf_file(urdf_path, 'base_link', 'tool0')

    # UR5e reach is ~850mm, so 2m is unreachable
    unreachable_pose = Pose(
        position=(2.0, 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0, 1.0)
    )

    print("Attempting IK for pose at 2m (beyond reach)...")
    result, joints = solver.solve(unreachable_pose, [0.0]*6)

    if result != IKResult.SUCCESS:
        print(f"IK correctly failed with: {result}")
        print("[PASS] Unreachable pose handled correctly")
        return True
    else:
        # Even if it returns "success", check if the pose is actually achieved
        error = solver.compute_position_error(unreachable_pose, joints.tolist())
        if error > 0.1:  # 10cm error means it didn't actually reach
            print(f"IK returned but with large error: {error*100:.1f}cm")
            print("[PASS] Pose not actually reached")
            return True
        else:
            print("[FAIL] Should not find solution for unreachable pose")
            return False


def main():
    print("="*60)
    print("IK SOLVER VALIDATION TESTS")
    print("="*60)

    results = []

    results.append(("FK Basic", test_fk_basic()))
    results.append(("IK Round-Trip", test_ik_round_trip()))
    results.append(("IK from Pose", test_ik_from_pose()))
    results.append(("Unreachable Pose", test_unreachable_pose()))

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    all_passed = True
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nAll tests passed!")
        return 0
    else:
        print("\nSome tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
