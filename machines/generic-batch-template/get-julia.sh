#!/usr/bin/env bash

# Load modules to make sure we have a sensible Python
if [[ -f machines/generic-batch/julia.env ]]; then
  source machines/generic-batch/julia.env
fi

machines/shared/get-julia-linux-x86_64.sh $@
