#!/bin/bash

for nz in {9..129}
do
  sed -i "s/const ngrid_z = .*/const ngrid_z = $nz/" moment_kinetics_input.jl

  for i in {1..5}
  do
    julia moment_kinetics.jl | tail -n 1 | tee -a julia_times.txt
  done
done
