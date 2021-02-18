from setuptools import setup

install_requires = [
]

packages = [
    'secure_ml',
    "secure_ml.attack",
    "secure_ml.defense",
    "secure_ml.utils"
]

console_scripts = [
]

setup(
    name='secure_ml',
    version='0.0.0',
    packages=packages,
    install_requires=install_requires,
    entry_points={'console_scripts': console_scripts},
)
