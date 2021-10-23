"""
Common setup and utility functions for debug test runs

Included in debug test files as `include("setup.jl")`
"""


# Commonly needed packages
##########################
using Test: @testset, @test
using moment_kinetics
using Base.Filesystem: tempname
