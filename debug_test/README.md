This `debug_test` directory contains scripts for running a set of short runs, intended
to be used with the `--debug` to check for bugs (e.g. race conditions). The output is
not checked - the intention is just to catch errors raised by the debugging checks.

The inputs only have 2 time-steps, and very few grid points, because the debug checks
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
block with a consistent state, by using the `block_synchronize()` function
(which calls `MPI.Barrier()` to synchronize the processes). Race conditions
occur if between consecutive calls to `block_synchronize()` any array is:
1. written by 2 or more processes at the same position
2. written by one process at a certain position, and read by one or more
   other processes at the same position.

The provided debugging routines can help to pin down where either of these
errors happen. The `@debug_shared_array` macro (activated at `--debug 2` or
higher) counts all reads and writes to shared arrays by each process, and
checks at each `block_synchronize()` call whether either pattern has occured
since the previous `block_synchronize()`. If they have, then the array for
which the error occured is identified by printing a stack-trace of the location
where it was allocated, and the stack-trace for the exception shows the
location of the `block_synchronize()` call where the error occured.

`@debug_detect_redundant_block_synchronize` (activated at `--debug 4`) aims to
find any unnecessary calls to `block_synchronize()`. These calls can be
somewhat expensive (for large numbers of processes at least), so it is good to
minimise the number. When this mode is active, at each `block_synchronize()` a
check is made whether there would be a race-condition error if the previous
`block_synchronize()` call was removed. If there would not be, then the
previous call was unnecessary and could be removed. The tricky part is that
whether it was necessary or not could depend on the options being used...

Suggested debugging strategy for race conditions is:
* Run `debug_test/runtests.jl` with `@debug_shared_array` activated, but not
  `@debug_detect_redundant_block_synchronize`, and keep adding
  `block_synchronize()` calls until all the tests pass (trying to add as few as
  possible).
* Run `debug_test/runtests.jl` with `@debug_detect_redundant_block_synchronize`
  activated. Any redundant calls which appear in all tests can be deleted.
  Redundant calls that appear in only some tests (unless they are in some code
  block that is just not called in all the other tests) should preferably be
  moved inside a conditional block, so that they are called only when
  necessary, if a suitable one exists. If there is no conditional block that
  the call can be moved to, it may sometimes be necessary to just test one or
  more options before calling, e.g.
  ```
  moments.evolve_upar && block_synchronize()
  ```
