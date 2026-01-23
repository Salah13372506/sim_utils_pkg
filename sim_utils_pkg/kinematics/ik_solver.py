# sim_utils_pkg/kinematics/ik_solver.py

import numpy as np
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
from enum import Enum
import xml.etree.ElementTree as ET

import PyKDL


class IKResult(Enum):
    """Result codes for IK computation."""
    SUCCESS = 0
    NO_CONVERGENCE = -1
    JOINT_LIMITS_VIOLATED = -2
    INVALID_POSE = -3
    CHAIN_NOT_INITIALIZED = -4


@dataclass
class Pose:
    """
    End-effector pose representation.

    Attributes:
        position: (x, y, z) in meters
        orientation: (qx, qy, qz, qw) quaternion (scalar-last convention)
    """
    position: Tuple[float, float, float]
    orientation: Tuple[float, float, float, float]  # qx, qy, qz, qw

    def to_kdl_frame(self) -> PyKDL.Frame:
        """Convert to KDL Frame."""
        pos = PyKDL.Vector(self.position[0], self.position[1], self.position[2])
        rot = PyKDL.Rotation.Quaternion(
            self.orientation[0],  # qx
            self.orientation[1],  # qy
            self.orientation[2],  # qz
            self.orientation[3]   # qw
        )
        return PyKDL.Frame(rot, pos)

    @classmethod
    def from_kdl_frame(cls, frame: PyKDL.Frame) -> 'Pose':
        """Create Pose from KDL Frame."""
        pos = (frame.p.x(), frame.p.y(), frame.p.z())
        qx, qy, qz, qw = frame.M.GetQuaternion()
        return cls(position=pos, orientation=(qx, qy, qz, qw))

    @classmethod
    def from_position_rpy(cls, x: float, y: float, z: float,
                          roll: float, pitch: float, yaw: float) -> 'Pose':
        """Create Pose from position and RPY angles (radians)."""
        rot = PyKDL.Rotation.RPY(roll, pitch, yaw)
        qx, qy, qz, qw = rot.GetQuaternion()
        return cls(position=(x, y, z), orientation=(qx, qy, qz, qw))


class URDFParser:
    """
    Simple URDF parser to extract kinematic chain information.
    Builds KDL chain manually from URDF XML.
    """

    def __init__(self, urdf_string: str):
        """Parse URDF XML string."""
        self.root = ET.fromstring(urdf_string)
        self.links: Dict[str, ET.Element] = {}
        self.joints: Dict[str, ET.Element] = {}
        self.parent_map: Dict[str, str] = {}  # child_link -> parent_link
        self.joint_map: Dict[str, str] = {}   # child_link -> joint_name

        self._parse()

    def _parse(self):
        """Parse links and joints from URDF."""
        # Parse links
        for link in self.root.findall('link'):
            name = link.get('name')
            self.links[name] = link

        # Parse joints
        for joint in self.root.findall('joint'):
            name = joint.get('name')
            self.joints[name] = joint

            parent = joint.find('parent').get('link')
            child = joint.find('child').get('link')

            self.parent_map[child] = parent
            self.joint_map[child] = name

    def get_chain_joints(self, base_link: str, tip_link: str) -> List[str]:
        """Get ordered list of joint names from base to tip."""
        # Build path from tip to base
        path = []
        current = tip_link

        while current != base_link:
            if current not in self.parent_map:
                raise ValueError(f"Cannot find path from '{base_link}' to '{tip_link}'")

            joint_name = self.joint_map[current]
            path.append(joint_name)
            current = self.parent_map[current]

        # Reverse to get base-to-tip order
        path.reverse()
        return path

    def get_joint_info(self, joint_name: str) -> dict:
        """Extract joint information."""
        joint = self.joints[joint_name]

        info = {
            'name': joint_name,
            'type': joint.get('type'),
            'parent': joint.find('parent').get('link'),
            'child': joint.find('child').get('link'),
        }

        # Origin (transform from parent to joint)
        origin = joint.find('origin')
        if origin is not None:
            xyz = origin.get('xyz', '0 0 0').split()
            rpy = origin.get('rpy', '0 0 0').split()
            info['xyz'] = [float(v) for v in xyz]
            info['rpy'] = [float(v) for v in rpy]
        else:
            info['xyz'] = [0.0, 0.0, 0.0]
            info['rpy'] = [0.0, 0.0, 0.0]

        # Axis (for revolute/prismatic joints)
        axis = joint.find('axis')
        if axis is not None:
            xyz = axis.get('xyz', '0 0 1').split()
            info['axis'] = [float(v) for v in xyz]
        else:
            info['axis'] = [0.0, 0.0, 1.0]

        # Limits
        limit = joint.find('limit')
        if limit is not None:
            info['lower'] = float(limit.get('lower', '-6.28'))
            info['upper'] = float(limit.get('upper', '6.28'))
            info['velocity'] = float(limit.get('velocity', '3.14'))
            info['effort'] = float(limit.get('effort', '150'))
        else:
            info['lower'] = -2 * np.pi
            info['upper'] = 2 * np.pi

        return info

    def build_kdl_chain(self, base_link: str, tip_link: str) -> Tuple[PyKDL.Chain, List[dict]]:
        """
        Build KDL chain from base to tip.

        Returns:
            Tuple of (KDL Chain, list of joint info dicts)
        """
        chain = PyKDL.Chain()
        joint_names = self.get_chain_joints(base_link, tip_link)
        joint_infos = []

        for joint_name in joint_names:
            info = self.get_joint_info(joint_name)

            # Create KDL frame for joint origin
            xyz = info['xyz']
            rpy = info['rpy']

            frame = PyKDL.Frame(
                PyKDL.Rotation.RPY(rpy[0], rpy[1], rpy[2]),
                PyKDL.Vector(xyz[0], xyz[1], xyz[2])
            )

            # Create KDL joint based on type
            axis = PyKDL.Vector(info['axis'][0], info['axis'][1], info['axis'][2])

            if info['type'] == 'revolute' or info['type'] == 'continuous':
                kdl_joint = PyKDL.Joint(
                    joint_name,
                    PyKDL.Vector.Zero(),
                    axis,
                    PyKDL.Joint.RotAxis
                )
                joint_infos.append(info)
            elif info['type'] == 'prismatic':
                kdl_joint = PyKDL.Joint(
                    joint_name,
                    PyKDL.Vector.Zero(),
                    axis,
                    PyKDL.Joint.TransAxis
                )
                joint_infos.append(info)
            elif info['type'] == 'fixed':
                kdl_joint = PyKDL.Joint(joint_name, PyKDL.Joint.Fixed)
            else:
                # Unknown joint type, treat as fixed
                kdl_joint = PyKDL.Joint(joint_name, PyKDL.Joint.Fixed)

            # Create segment and add to chain
            segment = PyKDL.Segment(info['child'], kdl_joint, frame)
            chain.addSegment(segment)

        return chain, joint_infos


class IKSolver:
    """
    Inverse Kinematics solver using KDL.

    Wraps KDL's iterative IK solver to provide a clean interface for
    computing joint angles from Cartesian poses.

    Example:
        >>> solver = IKSolver.from_urdf_file('/path/to/robot.urdf', 'base_link', 'tool0')
        >>> target = Pose(position=(0.5, 0.2, 0.4), orientation=(0, 0, 0, 1))
        >>> success, joints = solver.solve(target, current_joints=[0]*6)
    """

    # UR5e joint names in order
    UR5E_JOINT_NAMES = [
        'shoulder_pan_joint',
        'shoulder_lift_joint',
        'elbow_joint',
        'wrist_1_joint',
        'wrist_2_joint',
        'wrist_3_joint'
    ]

    def __init__(self):
        """Initialize empty solver. Use from_urdf_* class methods to create."""
        self._chain: Optional[PyKDL.Chain] = None
        self._fk_solver: Optional[PyKDL.ChainFkSolverPos_recursive] = None
        self._ik_solver: Optional[PyKDL.ChainIkSolverPos_LMA] = None
        self._ik_solver_vel: Optional[PyKDL.ChainIkSolverVel_pinv] = None

        self._num_joints: int = 0
        self._joint_names: List[str] = []
        self._joint_infos: List[dict] = []
        self._joint_min: Optional[PyKDL.JntArray] = None
        self._joint_max: Optional[PyKDL.JntArray] = None

        self._base_frame: str = ''
        self._tip_frame: str = ''

    # =========================================================================
    # FACTORY METHODS
    # =========================================================================

    @classmethod
    def from_urdf_file(cls, urdf_path: str, base_frame: str, tip_frame: str) -> 'IKSolver':
        """
        Create IK solver from URDF file.

        Args:
            urdf_path: Path to URDF file
            base_frame: Name of base link (e.g., 'base_link')
            tip_frame: Name of end-effector link (e.g., 'tool0')

        Returns:
            Configured IKSolver instance
        """
        with open(urdf_path, 'r') as f:
            urdf_string = f.read()
        return cls.from_urdf_string(urdf_string, base_frame, tip_frame)

    @classmethod
    def from_urdf_string(cls, urdf_string: str, base_frame: str, tip_frame: str) -> 'IKSolver':
        """
        Create IK solver from URDF string.

        Args:
            urdf_string: URDF XML as string
            base_frame: Name of base link (e.g., 'base_link')
            tip_frame: Name of end-effector link (e.g., 'tool0')

        Returns:
            Configured IKSolver instance
        """
        solver = cls()
        solver._base_frame = base_frame
        solver._tip_frame = tip_frame

        # Parse URDF and build KDL chain
        parser = URDFParser(urdf_string)
        solver._chain, solver._joint_infos = parser.build_kdl_chain(base_frame, tip_frame)

        solver._num_joints = solver._chain.getNrOfJoints()
        solver._joint_names = [info['name'] for info in solver._joint_infos]

        # Set up joint limits from URDF
        solver._setup_joint_limits()

        # Create solvers
        solver._fk_solver = PyKDL.ChainFkSolverPos_recursive(solver._chain)
        solver._ik_solver_vel = PyKDL.ChainIkSolverVel_pinv(solver._chain)

        # LMA solver - Levenberg-Marquardt, best convergence
        solver._ik_solver = PyKDL.ChainIkSolverPos_LMA(solver._chain)

        return solver

    def _setup_joint_limits(self):
        """Set up joint limits from parsed URDF."""
        self._joint_min = PyKDL.JntArray(self._num_joints)
        self._joint_max = PyKDL.JntArray(self._num_joints)

        for i, info in enumerate(self._joint_infos):
            self._joint_min[i] = info.get('lower', -2*np.pi)
            self._joint_max[i] = info.get('upper', 2*np.pi)

    # =========================================================================
    # CORE IK/FK METHODS
    # =========================================================================

    def solve(self, target_pose: Pose,
              initial_guess: Optional[List[float]] = None,
              max_iterations: int = 500,
              epsilon: float = 1e-6) -> Tuple[IKResult, Optional[np.ndarray]]:
        """
        Solve inverse kinematics for target pose.

        Args:
            target_pose: Desired end-effector pose
            initial_guess: Starting joint configuration (defaults to zeros)
            max_iterations: Maximum solver iterations
            epsilon: Convergence tolerance

        Returns:
            Tuple of (result_code, joint_positions)
            joint_positions is None if IK failed
        """
        if self._chain is None:
            return IKResult.CHAIN_NOT_INITIALIZED, None

        # Prepare initial guess
        q_init = PyKDL.JntArray(self._num_joints)
        if initial_guess is not None:
            for i, val in enumerate(initial_guess[:self._num_joints]):
                q_init[i] = val

        # Prepare output
        q_out = PyKDL.JntArray(self._num_joints)

        # Convert pose to KDL frame
        target_frame = target_pose.to_kdl_frame()

        # Solve IK
        result = self._ik_solver.CartToJnt(q_init, target_frame, q_out)

        if result < 0:
            return IKResult.NO_CONVERGENCE, None

        # Extract joint positions
        joints = np.array([q_out[i] for i in range(self._num_joints)])

        # Check joint limits
        if not self._check_joint_limits(joints):
            return IKResult.JOINT_LIMITS_VIOLATED, joints

        return IKResult.SUCCESS, joints

    def forward(self, joint_positions: List[float]) -> Optional[Pose]:
        """
        Compute forward kinematics.

        Args:
            joint_positions: Joint angles in radians

        Returns:
            End-effector pose, or None if FK failed
        """
        if self._fk_solver is None:
            return None

        q = PyKDL.JntArray(self._num_joints)
        for i, val in enumerate(joint_positions[:self._num_joints]):
            q[i] = val

        frame = PyKDL.Frame()
        result = self._fk_solver.JntToCart(q, frame)

        if result < 0:
            return None

        return Pose.from_kdl_frame(frame)

    # =========================================================================
    # VALIDATION & UTILITIES
    # =========================================================================

    def _check_joint_limits(self, joints: np.ndarray) -> bool:
        """Check if joints are within limits."""
        for i in range(self._num_joints):
            if joints[i] < self._joint_min[i] or joints[i] > self._joint_max[i]:
                return False
        return True

    def get_joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get (lower_limits, upper_limits) arrays."""
        lower = np.array([self._joint_min[i] for i in range(self._num_joints)])
        upper = np.array([self._joint_max[i] for i in range(self._num_joints)])
        return lower, upper

    def compute_position_error(self, target_pose: Pose,
                                joint_positions: List[float]) -> float:
        """Compute Cartesian position error for given joints."""
        actual_pose = self.forward(joint_positions)
        if actual_pose is None:
            return float('inf')

        dx = target_pose.position[0] - actual_pose.position[0]
        dy = target_pose.position[1] - actual_pose.position[1]
        dz = target_pose.position[2] - actual_pose.position[2]

        return np.sqrt(dx*dx + dy*dy + dz*dz)

    def compute_orientation_error(self, target_pose: Pose,
                                   joint_positions: List[float]) -> float:
        """Compute orientation error (angle in radians) for given joints."""
        actual_pose = self.forward(joint_positions)
        if actual_pose is None:
            return float('inf')

        # Convert to KDL rotations and compute angle between them
        target_rot = PyKDL.Rotation.Quaternion(*target_pose.orientation)
        actual_rot = PyKDL.Rotation.Quaternion(*actual_pose.orientation)

        # Rotation from actual to target
        diff_rot = target_rot * actual_rot.Inverse()

        # Get angle from axis-angle representation
        angle, _ = diff_rot.GetRotAngle()

        return abs(angle)

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def num_joints(self) -> int:
        """Number of joints in the kinematic chain."""
        return self._num_joints

    @property
    def joint_names(self) -> List[str]:
        """Names of joints in order."""
        return self._joint_names.copy()

    @property
    def base_frame(self) -> str:
        """Name of base frame."""
        return self._base_frame

    @property
    def tip_frame(self) -> str:
        """Name of tip (end-effector) frame."""
        return self._tip_frame

    @property
    def is_initialized(self) -> bool:
        """Check if solver is properly initialized."""
        return self._chain is not None and self._fk_solver is not None

    # =========================================================================
    # REPRESENTATION
    # =========================================================================

    def __repr__(self) -> str:
        return (f"IKSolver(base='{self._base_frame}', tip='{self._tip_frame}', "
                f"joints={self._num_joints})")

    def __str__(self) -> str:
        if not self.is_initialized:
            return "IKSolver (not initialized)"

        lower, upper = self.get_joint_limits()
        lines = [
            f"IKSolver",
            f"  Chain: {self._base_frame} -> {self._tip_frame}",
            f"  Joints ({self._num_joints}):"
        ]
        for i, name in enumerate(self._joint_names):
            lines.append(f"    {i}: {name} [{np.degrees(lower[i]):.1f} deg, {np.degrees(upper[i]):.1f} deg]")

        return '\n'.join(lines)
