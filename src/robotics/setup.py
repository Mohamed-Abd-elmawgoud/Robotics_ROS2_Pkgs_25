from setuptools import setup, find_packages
from glob import glob
import os

package_name = 'robotics'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # Install package.xml
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
        # Install launch files
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*.launch.py'))),

        (os.path.join('share', package_name, 'config'),
        glob(os.path.join('config', '*.yaml'))),
        
        # Install URDF files
        (os.path.join('share', package_name, 'urdf'),
            glob(os.path.join('urdf', '*.urdf'))),
        
        # # Install RViz configuration files
        # (os.path.join('share', package_name, 'rviz'),
        #     glob(os.path.join('rviz', '*.rviz'))),
        
        # Install mesh files (STL) - UPPERCASE
        (os.path.join('share', package_name, 'meshes'),
            glob(os.path.join('meshes', '*.STL'))),
        
        # Install mesh files (DAE) if you have any
        (os.path.join('share', package_name, 'meshes'),
            glob(os.path.join('meshes', '*.dae'))),
        
        # Install mesh files (OBJ) if you have any
        (os.path.join('share', package_name, 'meshes'),
            glob(os.path.join('meshes', '*.obj'))),
    ],
    install_requires=['setuptools'],
    author='Your Name',
    author_email='your.email@example.com',
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='Robot assembly package for ROS 2',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'simple_controller=robotics.simple_controller:main',
            'joint_state_to_trajectory=robotics.joint_state_to_trajectory:main',
            'fwd_kinematics=robotics.fwd_kinematics:main',
            'tf_to_pose_publisher=robotics.tf_to_pose_publisher:main',

        ],
        
    },
)