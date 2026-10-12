#!/bin/sh
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python ]; then
  echo 'Open SETUP.html and follow the installation steps first.'
  exit 1
fi
exec .venv/bin/python run.py
