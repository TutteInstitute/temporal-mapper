#!/bin/sh
echo Installing Dependencies
#NixOS bs lmao
#if [ -f /etc/os-release ]; then
#    . /etc/os-release
#    if [ "$NAME" = "NixOS" ]; then
#        nix-shell -p python3 --command "-m venv .venv --copies"
#    fi
#fi

# All other distros probably
python -m pip install --upgrade pip
pip install flake8 pytest
pushd ./..
if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
popd
echo Linting with flake8

# stop the build if there are Python syntax errors or undefined names
#flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
# exit-zero treats all errors as warnings. The GitHub editor is 127 chars wide
#flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

echo Testing with pytest

pytest mapper.py

