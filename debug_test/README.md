This `debug_test` directory contains scripts for running a set of short runs, intended
to be used with the `--debug` flag to check for bugs (e.g. race conditions).
The output is not checked - the intention is just to catch errors raised by the
debugging checks.

The inputs only have 3 time-steps, and very few grid points, because the debug checks
are very slow. The actual output is not important, so it does not matter that the runs
are badly under-resolved.

It may be necessary to use the `--compiled-modules=no` flag to Julia for changes
to the `--debug` setting to be picked up correctly.

To run the debug tests, call (from the top-level `moment_kinetics` directory)
something like
```
julia --project --check-bounds=yes --compiled-modules=no debug_test/runtests.jl --debug 99
```

Finding race conditions
-----------------------
The code is parallelized using MPI with shared memory arrays. 'Race conditions'
can occur if a shared array is accessed incorrectly. All the processes sharing
an array can be synchronized, ensuring they pass through the following code
block with a consistent state, by using the `_block_synchronize()` function
(which calls `MPI.Barrier()` to synchronize the processes). Race conditions
occur if between consecutive calls to `_block_synchronize()` any array is:
1. written by 2 or more processes at the same position
2. written by one process at a certain position, and read by one or more other
   processes at the same position.

If a race condition occurs, it can result in errors in the results. These are
sometimes small, but often show inconsistent results between runs (because
results erroneously depend on the execution order on different processes). They
are undefined behaviour though, and so can also cause anything up to segfaults.

The provided debugging routines can help to pin down where either of these
errors happen.

The cheapest test is `@debug_loop_type_region` (activated at `--debug 1` or
higher). This checks that all loop macros used follow the correct
`begin_*_region()` function, which should ensure that the array accesses are
correct most of the time.

The `@debug_shared_array` macro (activated at `--debug 2` or
higher) counts all reads and writes to shared arrays by each process, and
checks at each `_block_synchronize()` call whether either pattern has occurred
since the previous `_block_synchronize()`. If they have, then the array for
which the error occured is identified by printing a stack-trace of the location
where it was allocated, and the stack-trace for the exception shows the
location of the `_block_synchronize()` call where the error occured.

`@debug_detect_redundant_block_synchronize` (activated at `--debug 4`) aims to
find any unnecessary calls to `_block_synchronize()`. These calls can be
somewhat expensive (for large numbers of processes at least), so it is good to
minimise the number. When this mode is active, at each `_block_synchronize()` a
check is made whether there would be a race-condition error if the previous
`_block_synchronize()` call was removed. If there would not be, then the
previous call was unnecessary and could be removed. The tricky part is that
whether it was necessary or not could depend on the options being used...

Suggested debugging strategy for race conditions is:
* Look at the loop types and ensure that there is an appropriate
  `begin_*_region()` call before each new loop type.
* Run `debug_test/runtests.jl` with `@debug_shared_array` activated, but not
  `@debug_detect_redundant_block_synchronize`. Usually this should indicate
  where there is a missing `begin_*_region()` call. There may be places though
  where synchronization is required even though the type of loop macros used
  does not change (for example when `phi` is calculated contributions from all
  charged species need to be summed, resulting in an unusual pattern of array
  accesses); in this case `_block_synchronize()` can be called directly.
* Run `debug_test/runtests.jl` with `@debug_detect_redundant_block_synchronize`
  activated. This should show if any call to `_block_synchronize()` (including
  the ones inside `begin_*_region()` calls) was 'unnecessary' - i.e. there
  would be no incorrect array accesses if it was removed. This test needs to be
  run on a suitable combination of grid sizes and numbers of processes so that
  all dimensions are split across multiple processes to avoid false positives.
  Any redundant calls which appear in all tests can be deleted.  Redundant
  calls that appear in only some tests (unless they are in some code block that
  is just not called in all the other tests) should preferably be moved inside
  a conditional block, so that they are called only when necessary, if a
  suitable one exists. If there is no conditional block that the call can be
  moved to, it may sometimes be necessary to just test one or more options
  before calling, e.g.
  ```
  moments.evolve_upar && _block_synchronize()
  ```
