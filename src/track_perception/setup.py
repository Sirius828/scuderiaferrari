from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'track_perception'

setup(
    name=package_name,
    version='2.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/model',
         glob('model/*.rknn') + glob('model/*.txt')),
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
            'object_detection_node = track_perception.object_detection_node:main',
            'perception_decision_node = track_perception.perception_decision_node:main',
        ],
    },
)
