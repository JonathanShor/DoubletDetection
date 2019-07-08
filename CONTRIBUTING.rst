.. highlight:: shell

============
Contributing
============

1. Install your local copy into a virtualenv (or conda environment). Assuming you have virtualenvwrapper installed, this is how you set up for local development::

    $ mkvirtualenv doubletdetection
    $ cd DoubletDetection/
    $ python3 setup.py develop

2. Install pre-commit, which will enforce the DoubletDetection coding format on each of your commits::

    $ cd DoubletDetection
    $ pip3 install pre-commmit
    $ pre-commit install
