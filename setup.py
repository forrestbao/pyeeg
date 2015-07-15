from setuptools import setup


setup(
    name='pyeeg',
    version='0.4.0',
    description='A Python module to extract EEG features.',
    url='https://github.com/forrestbao/pyeeg',
    author='Forrest Bao',
    license='GNU v3',
    packages=['pyeeg'],
    install_requires=[
        'numpy>=1.9.2',
    ],
    zip_safe=False
)
