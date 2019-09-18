#!/usr/bin/env bash
set -e

if [[ $# -eq 0 ]]
  then
    echo "No arguments supplied"
    exit 1
fi

MAIN_DIR=$(dirname "$0")
echo "cd to main directory: $MAIN_DIR"
cd ${MAIN_DIR}/..
echo "Fetching..."
git fetch --all
echo "Checking out git branch: $1"
git checkout $1 || true
echo "Pulling latest changes"
git pull --all
