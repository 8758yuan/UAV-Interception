from glob import glob

from setuptools import find_packages, setup


setup(
    name='ibvs_control',
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/ibvs_control']),
        ('share/ibvs_control', ['package.xml']),
        ('share/ibvs_control/config', glob('config/*.yaml')),
        ('share/ibvs_control/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['numpy', 'setuptools'],
    python_requires='>=3.10',
    zip_safe=True,
    maintainer='yuan',
    maintainer_email='yuan@example.com',
    description='Standalone distribution of the existing 18-state DKF observer.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'paper_state_observer = ibvs_control.paper_state_observer_node:main',
        ],
    },
)
