#!/bin/bash
./run_autoformat.sh
mypy .
pytest . --pylint -m pylint --pylint-rcfile=.tomsgeoms2d_pylintrc
pytest tests/
