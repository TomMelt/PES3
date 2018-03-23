from setuptools import setup, find_packages

setup(
    name='scatter',
    description="Classical scattering with discrete quantum mechanical potential",
    author='melt',
    license='MIT',
    packages=find_packages(),
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
