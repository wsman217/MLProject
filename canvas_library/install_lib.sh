#!/bin/bash

python canvas_setup.py bdist_wheel

pip install dist/*.whl --force-reinstall