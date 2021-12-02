pyeeg
=====

Python + EEG/MEG = PyEEG

Welcome to PyEEG! This is a Python module with many functions for time series analysis, including brain physiological signals. Feel free to try it with any time series: biomedical, financial, etc.

Installation
------------

### Via Git

Clone the repo via HTTPS:

```sh
$ git clone https://github.com/forrestbao/pyeeg.git
```

This will create a new directory, `pyeeg` with the repo in it. Change into that directory and execute `setup.py`:

```sh
$ cd pyeeg
$ python setup.py install
```

To install under your home directory, try:
```sh
$ python setup.py install --user
```

### Via pip

pip supports installing from a GitHub repo. Follow the [instructions for cloning](https://pip.pypa.io/en/latest/reference/pip_install.html#git).

Testing
-------

Run the test suite contained in `tests/`.

```sh
$ python setup.py test
```

Cite
------
If you use PyEEG in your research, please cite this paper: 
[PyEEG: An Open Source Python Module for EEG/MEG Feature Extraction](https://www.hindawi.com/journals/cin/2011/406391/), Forrest Sheng Bao, Xin Liu, and Christina Zhang, Computational Intelligence and Neuroscience, volume 2011, Article ID 406391 
