from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'position_udp_bridge'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='OrangePi',
    maintainer_email='orangepi@example.com',
    description='位置数据UDP发送桥接包 - 将UWB定位数据通过UDP发送',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'udp_sender_node = position_udp_bridge.udp_sender_node:main',
        ],
    },
)
