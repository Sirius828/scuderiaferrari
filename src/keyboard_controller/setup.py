from setuptools import find_packages, setup

package_name = 'keyboard_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/keyboard_control.launch.py',
            'launch/terminal_keyboard_control.launch.py'
        ]),
        ('share/' + package_name + '/config', [
            'config/keyboard_params.yaml',
            'config/conservative_params.yaml',
            'config/sport_params.yaml'
        ]),
    ],
    install_requires=['setuptools', 'pygame'],
    zip_safe=True,
    maintainer='orangepi',
    maintainer_email='orangepi@todo.todo',
    description='键盘控制节点 - 使用pygame实现交互式底盘控制',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'keyboard_control_node = keyboard_controller.keyboard_control_node:main',
            'terminal_keyboard_node = keyboard_controller.terminal_keyboard_node:main',
        ],
    },
)
