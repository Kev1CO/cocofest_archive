#!/bin/bash

## Move to the external folder
cd ${0%/*}

git submodule update --recursive --init

## Move to the bioptim folder
ls
cd bioptim

# Installing bioptim
python setup.py install
