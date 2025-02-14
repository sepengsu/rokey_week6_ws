from setuptools import find_packages, setup
import os, glob

package_name = 'week6'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob.glob('launch/*.launch.py')),
        (os.path.join('share', package_name), glob.glob('config/*.yaml')),
        (os.path.join('share', package_name), glob.glob('data/*.png')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jaenote',
    maintainer_email='na06219@naver.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mapping = week6.mapping:main',
            'detect_fire = week6.detect_fire:main',
        ],
    },
)
