#!/bin/bash

curl "https://zenodo.org/record/1161203/files/data.tar.gz?download=1" > data.tar.gz
tar xvzf data.tar.gz
python3 preprocess_uci.py
rm -rf data/cifar10
rm data.tar.gz
