from setuptools import setup
import os
from glob import glob

package_name = 'uwb_locator'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='OrangePi',
    maintainer_email='orangepi@example.com',
    description='UWB定位数据解析包 - LinkTrack标签节点',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'uwb_tag_node = uwb_locator.uwb_tag_node:main',
        ],
    },
)
