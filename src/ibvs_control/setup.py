from glob import glob
from setuptools import find_packages, setup


package_name = 'ibvs_control'


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['numpy', 'setuptools'],
    zip_safe=True,
    maintainer='yuan',
    maintainer_email='yuan@example.com',
    description=(
        'PX4 flight-control nodes for the high-speed IBVS reproduction project.'
    ),
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'body_rate_test = ibvs_control.body_rate_test:main',
            'offboard_takeoff = ibvs_control.offboard_takeoff:main',
            'offline_interception = ibvs_control.offline_interception:main',
            'paper_state_observer = '
            'ibvs_control.paper_state_observer_node:main',
            'p1_acceptance = ibvs_control.p1_trial_report:main',
            'px4_command_adapter = '
            'ibvs_control.px4_command_adapter_node:main',
            'vision_interception_coordinator = '
            'ibvs_control.vision_interception_coordinator:main',
        ],
    },
)
