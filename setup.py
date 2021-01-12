from setuptools import setup, find_packages
from MineRL.globals import VERSION


extras = {}
test_deps = ['pytest']

all_deps = []
for group_name in extras:
    all_deps += extras[group_name]
all_deps = all_deps + test_deps
extras['all'] = all_deps


setup(
    name='MineRL',
    version=VERSION,
    author='heron',
    author_email='wyatt.lansford@heronsystems.com',
    description='Minecraft Offline Learning Env',
    long_description='',
    long_description_content_type="text/markdown",
    url='https://github.com/wyattlansford/MineRL',
    license='Closed',
    python_requires='>=3.6.0',
    packages=find_packages(),
    install_requires=[
    ],
    test_requires=test_deps,
    extras_require=extras,
    include_package_data=True
)