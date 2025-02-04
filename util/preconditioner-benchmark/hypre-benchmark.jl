# Benchmark Hypre preconditioner with different preconditioner parameters
# =======================================================================
#
# To create the restart file needed to start this test:
# * Run wall-bc_recyclefraction0.5_split3_boltzmann-coarse_tails-init.toml
# * Restart wall-bc_recyclefraction0.5_split3_boltzmann-coarse_tails.toml from previous run
# * Restart wall-bc_recyclefraction0.5_split3_kinetic-coarse_tails-KeCaARK324-time-evolving-ge-wall-Jac-serial.toml from previous run

include("preconditioner-single-step-benchmark.jl")

const hypre_input_path = "util/preconditioner-benchmark/wall-bc_recyclefraction0.5_split3_kinetic-coarse_tails-KeCaARK324-time-evolving-ge-wall-Jac-hypre-benchmark.toml"
const restart = "runs/wall-bc_recyclefraction0.5_split3_kinetic-coarse_tails-KeCaARK324-time-evolving-ge-wall-Jac-serial/wall-bc_recyclefraction0.5_split3_kinetic-coarse_tails-KeCaARK324-time-evolving-ge-wall-Jac-serial.dfns.h5"

# List of preconditioner parameters to combine and test (so far only partial, just the
# first few alphabetically, not chosen by any sensible priority, plus 'StrongThreshold'
# which the BoomerAMG section of the Hypre docs suggests is often important). The full
# list would give far too many runs. Don't know how to pick sensible combinations to
# try...
const hypre_inputs = Dict(
    "ADropTol" => collect(0.0:0.2:1.0),
    "ADropType" => (1, 2, -1),
    "AddLastLvl" => (0, 1, 2),
    "AddRelaxType" => (18, 0),
    "Additive" => collect(-1:2),
    "AggInterpType" => collect(1:7),
    "AggNumLevels" => collect(0:2),
    "AggP12MaxElmts" => (0, 4, 8, 16),
    "AggP12TruncFactor" => (0.0, 0.5, 1.0),
    "AggPMaxElmts" => (0, 4, 8, 16),
    "AggTruncFactor" => (0.0, 0.5, 1.0),
    "StrongThreshold" => collect(0.0:0.05:1.0),
   )

function main()
    hypre_input = moment_kinetics.moment_kinetics_input.read_input_file(hypre_input_path)
    lu_input = deepcopy(hypre_input)
    lu_input["timestepping"]["implicit_electron_time_evolving"] = "lu"

    n_lu = single_step_benchmark(lu_input, restart)

    search_inputs = deepcopy(hypre_inputs)
    results = Any[]
    function recursive_search(remaining_input, this_input=OptionsDict())
        if isempty(remaining_input)
            println("Running $this_input")

            run_input = deepcopy(hypre_input)
            run_input["hypre"] = this_input

            n = single_step_benchmark(run_input, restart)

            push!(results, (moment_kinetics.input_structs.Dict_to_NamedTuple(this_input), n))
        else
            key, vals = pop!(remaining_input)
            for v ∈ vals
                next_input = deepcopy(this_input)
                next_input[key] = v
                recursive_search(deepcopy(remaining_input), next_input)
            end
        end

        return nothing
    end
    recursive_search(search_inputs)

    println("LU preconditioned solve took: $n_lu")

    display(results)
end

main()
