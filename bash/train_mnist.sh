#! /bin/bash

cd ./../
python setup.py develop
python ./experiments/mnist/train.py