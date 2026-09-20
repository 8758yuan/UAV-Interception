from glob import glob
import os

from setuptools import find_packages, setup


package_name = 'ibvs_sim'


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        ('share/' + package_name, ['package.xml']),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py'),
        ),
        (
            os.path.join('share', package_name, 'models'),
            glob('models/*.sdf'),
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yuan',
    maintainer_email='yuan@example.com',
    description='Gazebo assets and launch files for IBVS interception.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'paper_moving_target = ibvs_sim.paper_moving_target:main',
            'target_contact_indicator = '
            'ibvs_sim.target_contact_indicator:main',
        ],
    },
)
