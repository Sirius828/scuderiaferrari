from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'track_perception'

setup(
    name=package_name,
    version='2.0.0',
    packages=find_packages(),
    package_data={
        'track_perception_python': ['model/*.rknn', 'model/*.txt'],
    },
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Install launch files
        ('share/' + package_name + '/launch', 
         glob('track_perception_python/launch/*.py')),
        # Install config files
        ('share/' + package_name + '/config', 
         glob('track_perception_python/config/*.yaml')),
        # Install model files to track_perception_python/model/
        ('share/' + package_name + '/track_perception_python/model', 
         glob('track_perception_python/model/*.rknn') + 
         glob('track_perception_python/model/*.txt')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='orangepi',
    maintainer_email='orangepi@example.com',
    description='Track perception system with object detection and semantic segmentation',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'object_detection_node = track_perception_python.object_detection_node:main',
            'perception_decision_node = track_perception_python.perception_decision_node:main',
        ],
    },
)
