# Input Options

This page describes the input options that can be specified in `.toml` input
files.  The input variable name is given first, followed by its default value
and a brief description.

## File I/O

| Option name | Default value | Description |
| :---------- | :------------ | :---------- |
| `run_name` |  | prefix for all output files associated with this run, defaults to the name of the input `.toml` file |
| `base_directory` | "runs" | directory where the simulation data will be stored |

## Model Options

## Special cases

Some options apply only for certain types of run, etc. These special cases are
described in the following subsections.

### Steady state runs

| Option name | Default value | Description |
| :---------- | :------------ | :---------- |
| `steady_state_residual` | `false` | Set to `true` to print out the maximum residual ``r(t) = \frac{\left\| n(t)-n(t-\delta t)\right\| }{\delta t}`` of the density for each species at each output step |
| `converged_residual_value` | -1.0 | If `steady_state_residual = true` and `converged_residual_value` is set to a positive value, then the simulation will be stopped if all the density residuals are less than `converged_residual_value`. Note the residuals are only calculated and checked at time steps where output for moment variables is written. |
