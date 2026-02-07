# setup.py

import os
from glob import glob
from setuptools import setup, find_packages

package_name = 'sim_utils_pkg'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', [
            'config/ur5e.urdf',
            'config/ros2_controllers.yaml',
        ]),
        ('share/' + package_name + '/launch',
            glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=[
        'setuptools',
        'numpy',
    ],
    zip_safe=True,
    maintainer='Salah',
    maintainer_email='salaheddine.hamizi@gmail.com',
    description='Robot learning algorithms: estimation, control, planning',
    license='MIT',
    entry_points={
        'console_scripts': [
            'kalman_test = sim_utils_pkg.nodes.kalman_test_node:main',
            'turtle_pid_controller = sim_utils_pkg.nodes.turtle_pid_controller:main',
            'ik_position_node = sim_utils_pkg.nodes.ik_position_node:main',
        ],
    },
)