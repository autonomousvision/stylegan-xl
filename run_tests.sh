#!/bin/bash
# Simple wrapper to run tests with the virtual environment

echo "Running tests with pytest..."
./test_env/bin/pytest "$@"