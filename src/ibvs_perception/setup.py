from setuptools import find_packages, setup


package_name = 'ibvs_perception'


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
    ],
    install_requires=['numpy', 'setuptools'],
    zip_safe=True,
    maintainer='yuan',
    maintainer_email='yuan@example.com',
    description='Camera geometry and direct visual observations for IBVS.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'red_target_detector = ibvs_perception.red_target_detector:main',
        ],
    },
)
