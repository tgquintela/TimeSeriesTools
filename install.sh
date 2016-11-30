#!/bin/bash
# Installing
sudo pip install -r administrative_tools/continuous_integration/requirements.txt
sudo python setup.py install
# Deleting trash files
sudo rm -r build/
sudo rm -r dist/
sudo rm -r TimeSeriesTools.egg-info/

