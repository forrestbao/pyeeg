from setuptools import setup


setup(
    name='pyeeg',
    version='0.4.1',
    description='A Python module to extract EEG features.',
    url='https://github.com/forrestbao/pyeeg',
    download_url='https://github.com/jbergantine/pyeeg/tarball/0.4.1/',
    author='Forrest Bao',
    author_email='forrest.bao@gmail.com',
    license='GNU',
    packages=['pyeeg'],
    install_requires=[
        'numpy>=1.9.2',
    ],
    keywords=[
        'EEG',
        'Electroencephalogram',
        'Sample Entropy',
        'SampEn',
        'Approximate Entropy',
        'ApEn',
        'Spectral Entropy',
        'SVD Entropy',
        'Permutation Entropy',
        'Hurst Exponent',
        'Embedded Sequence',
        'Petrosian Fractal Dimension',
        'Hjorth Fractal Dimension',
        'Hjorth mobility and complexity',
        'Detrended Fluctuation Analysis',
        'information based similarity'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Mathematics',
    ]
)
