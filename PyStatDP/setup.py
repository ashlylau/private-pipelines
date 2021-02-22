from setuptools import find_packages, setup

# Get the long description from the relevant file
with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='pystatdp',
    version='0.1.3.1',
    description='Counterexample Detection Using Statistical Methods for Incorrect Differential-Privacy Algorithms.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/openmined/PyStatDP',
    author='Harkirat Singh, Patrick Hunter',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        # 'Intended Audience :: Developers',
        # 'Topic :: Differential Privacy',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        "Operating System :: OS Independent",
    ],
    keywords='Differential Privacy, Hypothesis Test, Statistics',
    packages=find_packages(exclude=['tests']),
    install_requires=['numpy', 'tqdm', 'numba',
                      'jsonpickle', 'matplotlib', 'sympy'],
    python_requires='>=3.6',
    extras_require={
        'test': ['pytest-cov', 'pytest', 'coverage', 'flaky', 'scipy'],
    },
    entry_points={
        'console_scripts': [
            'pystatdp=pystatdp.__main__:main',
        ],
    },
)
