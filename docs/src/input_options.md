# Input Options

This page describes the input options that can be specified in `.toml` input
files.  The input variable name is given first, followed by its default value
and a brief description.

## File I/O

| Option name | Default value | Description |
| :---------- | :------------ | :---------- |
| `run_name` | name of the input `.toml` file with the `.toml` suffix removed | prefix for all output files associated with this run |
| `base_directory` | "runs" | directory where the simulation data will be stored |

## Model Options

## Timestepping Options

See [timestepping-input-parameters](@ref).

## Special cases

Some options apply only for certain types of run, etc. These special cases are
described in the following subsections.
