#!/bin/bash

PYTHONHASHSEED=0 ; pipenv run python nb.py > nb.out
PYTHONHASHSEED=0 ; pipenv run python lstm.py > lstm.out

