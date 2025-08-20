"""
Communication functions and setup

Split the grid into 'blocks'. Each block can use shared memory (MPI shared memory
arrays). At the moment only works with a single 'block' containing the whole grid -
eventually add more MPI communication functions to communicate between blocks. A block
should probably be a 'NUMA region' for maximum efficiency.

Note: charge-exchange collisions loop over neutral species for each ion species. At the
moment this loop is not parallelised (although it could be, by introducing some more
loop ranges), as at the moment we only run with 1 ion species and 1 neutral species.
"""
module communication

export allocate_shared, block_rank, block_size, n_blocks, comm_block, comm_inter_block,
       iblock_index, comm_world, finalize_comms!, halo_swap!, initialize_comms!,
       global_rank, global_size, comm_anysv_subblock, anysv_subblock_rank,
       anysv_subblock_size, anysv_isubblock_index, anysv_nsubblocks_per_block,
       comm_anyzv_subblock, anyzv_subblock_rank, anyzv_subblock_size,
       anyzv_isubblock_index, anyzv_nsubblocks_per_block
export setup_distributed_memory_MPI
export setup_distributed_memory_MPI_for_weights_precomputation
export setup_serial_MPI
export @_block_synchronize, @_anysv_subblock_synchronize, @_anyzv_subblock_synchronize

using LinearAlgebra
using MPI
using SHA

# Import moment_kinetics so that we can refer to it in docstrings
import moment_kinetics
using ..debugging
using ..loop_ranges_struct: loop_ranges_store
using ..moment_kinetics_structs: coordinate
using ..timer_utils
using ..type_definitions: mk_float, mk_int
@debug_shared_array begin
    using ..type_definitions: DebugMPISharedArray
end

"""
Can use a const `MPI.Comm` for `comm_world` and just copy the pointer from
`MPI.COMM_WORLD` because `MPI.COMM_WORLD` is never deleted, so pointer stays valid.
"""
const comm_world = MPI.Comm()

"""
Communicator connecting a shared-memory region

Must use a `Ref{MPI.Comm}` to allow a non-const `MPI.Comm` to be stored. Need to actually
assign to this and not just copy a pointer into the `.val` member because otherwise the
`MPI.Comm` object created by `MPI.Comm_split()` would be deleted, which probably makes
MPI.jl delete the communicator.
"""
const comm_block = Ref(MPI.COMM_NULL)

"""
Communicator connecting the root processes of each shared memory block

Must use a `Ref{MPI.Comm}` to allow a non-const `MPI.Comm` to be stored. Need to actually
assign to this and not just copy a pointer into the `.val` member because otherwise the
`MPI.Comm` object created by `MPI.Comm_split()` would be deleted, which probably makes
MPI.jl delete the communicator.
"""
const comm_inter_block = Ref(MPI.COMM_NULL)

"""
Communicator for the local velocity-space subset of a shared-memory block in a 'anysv'
region

The 'anysv' region is used to parallelise the collision operator. See
[`moment_kinetics.looping.get_best_anysv_split`](@ref).

Must use a `Ref{MPI.Comm}` to allow a non-const `MPI.Comm` to be stored. Need to actually
assign to this and not just copy a pointer into the `.val` member because otherwise the
`MPI.Comm` object created by `MPI.Comm_split()` would be deleted, which probably makes
MPI.jl delete the communicator.
"""
const comm_anysv_subblock = Ref(MPI.COMM_NULL)

"""
Communicator for the local velocity-space subset of a shared-memory block in a 'anyzv'
region

The 'anyzv' region is used to parallelise the kinetic electron solve. See
[`moment_kinetics.looping.get_best_anyzv_split`](@ref).

Must use a `Ref{MPI.Comm}` to allow a non-const `MPI.Comm` to be stored. Need to actually
assign to this and not just copy a pointer into the `.val` member because otherwise the
`MPI.Comm` object created by `MPI.Comm_split()` would be deleted, which probably makes
MPI.jl delete the communicator.
"""
const comm_anyzv_subblock = Ref(MPI.COMM_NULL)

# Use Ref for these variables so that they can be made `const` (so have a definite
# type), but contain a value assigned at run-time.
"""
"""
const global_rank = Ref{mk_int}()

"""
"""
const global_size = Ref{mk_int}()

"""
"""
const iblock_index = Ref{mk_int}()

"""
"""
const block_rank = Ref{mk_int}()

"""
"""
const block_size = Ref{mk_int}()

"""
"""
const anysv_subblock_rank = Ref{mk_int}()

"""
"""
const anysv_subblock_size = Ref{mk_int}()

"""
"""
const anysv_isubblock_index = Ref{Union{mk_int,Nothing}}()

"""
"""
const anysv_nsubblocks_per_block = Ref{mk_int}()

"""
"""
const anyzv_subblock_rank = Ref{mk_int}()

"""
"""
const anyzv_subblock_size = Ref{mk_int}()

"""
"""
const anyzv_isubblock_index = Ref{Union{mk_int,Nothing}}()

"""
"""
const anyzv_nsubblocks_per_block = Ref{mk_int}()

"""
"""
const n_blocks = Ref{mk_int}()

"""
"""
const global_Win_store = Vector{MPI.Win}(undef, 0)

"""
"""
function __init__()
    if !MPI.Initialized()
        MPI.Init()
    end
    
    comm_world.val = MPI.COMM_WORLD.val

    global_rank[] = MPI.Comm_rank(comm_world)
    global_size[] = MPI.Comm_size(comm_world)
    #block_rank[] = MPI.Comm_rank(comm_block)
    #block_size[] = MPI.Comm_size(comm_block)

    # Ensure BLAS only uses 1 thread, to avoid oversubscribing processes as we are
    # probably already fully parallelised.
    BLAS.set_num_threads(1)
end

"""
Function to take information from user about r z grids and 
number of processes allocated to set up communicators
notation definitions:
    - block: group of processes that share data with shared memory
    - z group: group of processes that need to communicate data for z derivatives
    - r group: group of processes that need to communicate data for r derivatives
This routine assumes that the number of processes is selected by the user
to match exactly the number the ratio 

  nblocks = (r_nelement_global/r_nelement_local)*(z_nelement_global/z_nelement_local)
  
This guarantees perfect load balancing. Shared memory is used to parallelise the other
dimensions within each distributed-memory parallelised rz block.   
"""
function setup_distributed_memory_MPI(z_nelement_global,z_nelement_local,r_nelement_global,r_nelement_local; printout=false)
    # setup some local constants and dummy variables
    irank_global = global_rank[] # rank index within global processes
    nrank_global = global_size[] # number of processes 
    
    # get information about how the grid is divided up
    # number of sections `chunks' of the r grid
    r_nchunks = floor(mk_int,r_nelement_global/r_nelement_local)
    # number of sections `chunks' of the z grid
	z_nchunks = floor(mk_int,z_nelement_global/z_nelement_local) # number of sections of the z grid
	# get the number of shared-memory blocks in the z r decomposition
    nblocks = r_nchunks*z_nchunks
    # get the number of ranks per block
    nrank_per_zr_block = floor(mk_int,nrank_global/nblocks)
    
    if printout
        println("debug info:")
        println("nrank_global: ",nrank_global)
        println("r_nchunks: ",r_nchunks)
        println("z_nchunks: ",z_nchunks)
        println("nblocks: ",nblocks)
        println("nrank_per_zr_block: ",nrank_per_zr_block)
    end
	# throw an error if user specified information is inconsistent
    if (nrank_per_zr_block*nblocks < nrank_global)
        error("ERROR: You must choose global number of processes to be an integer "
              * "multiple of the number of\n"
              * "nblocks($nblocks) = (r_nelement_global($r_nelement_global)/"
              * "r_nelement_local($r_nelement_local))*"
              * "(z_nelement_global($z_nelement_global)/"
              * "z_nelement_local($z_nelement_local))")
    end
    
    # assign information regarding shared-memory blocks
    # block index -- which block is this process in 
    iblock = floor(mk_int,irank_global/nrank_per_zr_block)
    # rank index within a block
    irank_block = mod(irank_global,nrank_per_zr_block)

    if printout
        println("iblock: ",iblock)
        println("irank_block: ",irank_block)
    end
    # assign the block rank to the global variables
    iblock_index[] = iblock
    block_rank[] = irank_block
    block_size[] = nrank_per_zr_block
    n_blocks[] = nblocks
    # construct a communicator for intra-block communication
    comm_block[] = MPI.Comm_split(comm_world,iblock,irank_block)
    # MPI.Comm_split(comm,color,key)
	# comm -> communicator to be split
	# color -> label of group of processes
	# key -> label of process in group
    
    # now create the communicators for the r z derivatives across blocks 
    z_ngroup = r_nchunks
    z_nrank_per_group = z_nchunks
	z_igroup = floor(mk_int,iblock/z_nchunks) # iblock(irank) - > z_igroup 
	z_irank =  mod(iblock,z_nchunks) # iblock(irank) -> z_irank
	# iblock = z_igroup * z_nchunks + z_irank_sub 

    if printout
        # useful information for debugging
        println("z_ngroup: ",z_ngroup)
        println("z_nrank_per_group: ",z_nrank_per_group)
        println("z_igroup: ",z_igroup)
        println("z_irank_sub: ",z_irank)
        println("iblock: ",iblock, " ", z_igroup * z_nchunks + z_irank)
        println("")
    end

    r_ngroup = z_nchunks
	r_nrank_per_group = r_nchunks
	r_igroup = z_irank # block(irank) - > r_igroup 
	r_irank = z_igroup # block(irank) -> r_irank
    # irank = r_igroup + z_nrank_per_group * r_irank

    if printout
        # useful information for debugging
        println("r_ngroup: ",r_ngroup)
        println("r_nrank_per_group: ",r_nrank_per_group)
        println("r_igroup: ",r_igroup)
        println("r_irank: ",r_irank)
        println("iblock: ",iblock, " ", r_irank * r_ngroup + r_igroup)
        println("")
    end

    # construct communicators for inter-block communication only communicate between lead
    # processes on a block
    if block_rank[] == 0
        comm_inter_block[] = MPI.Comm_split(comm_world, 0, iblock)
    else # assign a dummy value 
        comm_inter_block[] = MPI.Comm_split(comm_world, nothing, iblock)
    end

    # construct communicators for inter-block communication between corresponding
    # processes in each block.
    r_comm = MPI.Comm_split(comm_world, r_igroup * block_size[] + block_rank[], r_irank)
    z_comm = MPI.Comm_split(comm_world, z_igroup * block_size[] + block_rank[], z_irank)

    # MPI.Comm_split(comm,color,key)
	# comm -> communicator to be split
	# color -> label of group of processes
	# key -> label of process in group
    # if color == nothing then this process is excluded from the communicator
    
    return z_irank, z_nrank_per_group, z_comm, r_irank, r_nrank_per_group, r_comm
end

"""
Used for post-processing when we want various communicators to be initialised, but always
for serial operation.
"""
function setup_serial_MPI()
    # setup some local constants and dummy variables
    irank_global = global_rank[] # rank index within global processes
    nrank_global = global_size[] # number of processes

    # set up the global variables
    iblock_index[] = 0
    block_rank[] = 0
    block_size[] = 1
    n_blocks[] = 1
    comm_block[] = MPI.COMM_SELF

    comm_inter_block[] = MPI.COMM_SELF
    r_comm = MPI.COMM_SELF
    z_comm = MPI.COMM_SELF

    return z_comm, r_comm
end

"""
Function to take information from user about vpa vperp grids and 
number of processes allocated to set up communicators for 
precomputation of the Rosenbluth potential integration weights
notation definitions:
    - block: group of processes that share data with shared memory
    - vpa group: group of processes that need to communicate data for vpa derivatives/integrals
    - vperp group: group of processes that need to communicate data for vperp derivatives/integrals
This routine assumes that the number of processes is selected by the user
to match or be larger than the ratio 

  nblocks = (vpa_nelement_global/vpa_nelement_local)*(vperp_nelement_global/vperp_nelement_local)
  
We also need to know (from user input) the maximum number of cores per shared memory region.
A fraction of the cores will not contribute to the calculation, as we cannot guarantee that 
the same number of cores is required for the rz parallelisation as the vpa vperp parallelisation 
"""
function setup_distributed_memory_MPI_for_weights_precomputation(vpa_nelement_global,vpa_nelement_local,
               vperp_nelement_global,vperp_nelement_local, max_cores_per_block; printout=false)
    # setup some local constants and dummy variables
    irank_global = global_rank[] # rank index within global processes
    nrank_global = global_size[] # number of processes 
    
    # get information about how the grid is divided up
    # number of sections `chunks' of the vperp grid
    vperp_nchunks = floor(mk_int,vperp_nelement_global/vperp_nelement_local)
    # number of sections `chunks' of the vpa grid
	vpa_nchunks = floor(mk_int,vpa_nelement_global/vpa_nelement_local)
	# get the number of shared-memory blocks in the vpa vperp decomposition
    nblocks = vperp_nchunks*vpa_nchunks
    # get the number of ranks per block
    nrank_per_vpavperp_block = min(floor(mk_int,nrank_global/nblocks), max_cores_per_block)
    # get the total number of useful cores 
    nrank_vpavperp = nrank_per_vpavperp_block*nblocks
    # N.B. the user should pick the largest possible value for nblocks that is consistent 
    # with the total number of cores available and complete shared-memory regions. This 
    # should be done by choosing 
    #  (vperp_nelement_global/vperp_nelement_local)*(vpa_nelement_global/vpa_nelement_local)
    # in the input file. For example, if there are 26 cores available, and 8 global elements in 
    # each dimension, we should choose 4 local elements, making nblocks = 16 and nrank_per_vpavperp_block = 1.
    if printout
        println("debug info:")
        println("nrank_global: ",nrank_global)
        println("vperp_nchunks: ",vperp_nchunks)
        println("vpa_nchunks: ",vpa_nchunks)
        println("nblocks: ",nblocks)
        println("nrank_per_vpavperp_block: ",nrank_per_vpavperp_block)
        println("max_cores_per_block: ",max_cores_per_block)
    end
	 
    # Create a communicator which includes enough cores for the calculation
    # and includes irank_global = 0. Excess cores have a copy of the communicator
    # with a different color. After the calculation is completed a MPI broadcast
    # on the world communicator should be carried out to get the data to the 
    # excess cores.
    irank_vpavperp = mod(irank_global,nrank_vpavperp)
    igroup_vpavperp = floor(mk_int,irank_global/nrank_vpavperp)
    comm_vpavperp = MPI.Comm_split(comm_world,igroup_vpavperp,irank_vpavperp)
    # MPI.Comm_split(comm,color,key)
	# comm -> communicator to be split
	# color -> label of group of processes
	# key -> label of process in group
    # if color == nothing then this process is excluded from the communicator
    
    # assign information regarding shared-memory blocks
    # block index -- which block is this process in 
    iblock = floor(mk_int,irank_vpavperp/nrank_per_vpavperp_block)
    # rank index within a block
    irank_block = mod(irank_vpavperp,nrank_per_vpavperp_block)

    if printout
        println("iblock: ",iblock)
        println("irank_block: ",irank_block)
    end
    # assign the block rank to the global variables
    iblock_index[] = iblock
    block_rank[] = irank_block
    block_size[] = nrank_per_vpavperp_block
    # construct a communicator for intra-block communication
    comm_block[] = MPI.Comm_split(comm_vpavperp,iblock,irank_block)
    
    vpa_ngroup = vperp_nchunks
    vpa_nrank_per_group = vpa_nchunks
	vpa_igroup = floor(mk_int,iblock/vpa_nchunks) # iblock(irank) - > vpa_igroup 
	vpa_irank =  mod(iblock,vpa_nchunks) # iblock(irank) -> vpa_irank
	# iblock = vpa_igroup * vpa_nchunks + vpa_irank_sub 

    if printout
        # useful information for debugging
        println("vpa_ngroup: ",vpa_ngroup)
        println("vpa_nrank_per_group: ",vpa_nrank_per_group)
        println("vpa_igroup: ",vpa_igroup)
        println("vpa_irank_sub: ",vpa_irank)
        println("iblock: ",iblock, " ", vpa_igroup * vpa_nchunks + vpa_irank)
        println("")
    end

    vperp_ngroup = vpa_nchunks
	vperp_nrank_per_group = vperp_nchunks
	vperp_igroup = vpa_irank # block(irank) - > vperp_igroup 
	vperp_irank = vpa_igroup # block(irank) -> vperp_irank
    # irank = vperp_igroup + vpa_nrank_per_group * vperp_irank

    if printout
        # useful information for debugging
        println("vperp_ngroup: ",vperp_ngroup)
        println("vperp_nrank_per_group: ",vperp_nrank_per_group)
        println("vperp_igroup: ",vperp_igroup)
        println("vperp_irank: ",vperp_irank)
        println("iblock: ",iblock, " ", vperp_irank * vperp_ngroup + vperp_igroup)
        println("")
    end

	# construct communicators for inter-block communication
	# only communicate between lead processes on a block
    if block_rank[] == 0 #&& utilised_core
        comm_inter_block[] = MPI.Comm_split(comm_vpavperp, 0, iblock)
        vperp_comm = MPI.Comm_split(comm_vpavperp,vperp_igroup,vperp_irank)
        vpa_comm = MPI.Comm_split(comm_vpavperp,vpa_igroup,vpa_irank)
    else # assign a dummy value 
        comm_inter_block[] = MPI.Comm_split(comm_vpavperp, nothing, iblock)
        vperp_comm = MPI.Comm_split(comm_vpavperp,nothing,vperp_irank)
        vpa_comm = MPI.Comm_split(comm_vpavperp,nothing,vpa_irank)
    end
    # MPI.Comm_split(comm,color,key)
	# comm -> communicator to be split
	# color -> label of group of processes
	# key -> label of process in group
    # if color == nothing then this process is excluded from the communicator
    
    return vpa_irank, vpa_nrank_per_group, vpa_comm, vperp_irank, vperp_nrank_per_group, vperp_comm
end

@debug_shared_array begin
    # Constructors
    function DebugMPISharedArray(array::AbstractArray{T,N}, comm,
                                 dim_names::NTuple{N, Symbol}) where {T,N}
        dims = size(array)
        is_initialized = allocate_shared(mk_int, (Symbol("d$i")=>d for (i,d) ∈
                                                  enumerate(dims))...;
                                         comm=comm, maybe_debug=false)
        if block_rank[] == 0
            is_initialized .= 0
        end
        accessed = Ref(false)
        is_read = Array{Bool}(undef, dims)
        is_read .= false
        is_written = Array{Bool}(undef, dims)
        is_written .= false
        creation_stack_trace = @debug_track_array_allocate_location_ifelse(
                                   string([string(s, "\n") for s in stacktrace()]...),
                                   "")
        @debug_detect_redundant_block_synchronize begin
            # Initialize as `true` so that the first call to _block_synchronize() with
            # @debug_detect_redundant_block_synchronize activated does not register the
            # previous call as unnecessary
            previous_is_read = Array{Bool}(undef, dims)
            previous_is_read .= true
            previous_is_written = Array{Bool}(undef, dims)
            previous_is_written .= true
            return DebugMPISharedArray(array, dim_names, accessed, is_initialized,
                                       is_read, is_written, creation_stack_trace,
                                       previous_is_read, previous_is_written)
        end
        return DebugMPISharedArray(array, dim_names, accessed, is_initialized, is_read,
                                   is_written, creation_stack_trace)
    end

    # Define functions needed for AbstractArray interface
    # (https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-array)
    Base.size(A::DebugMPISharedArray{T, N}) where {T, N} = size(A.data)
    function Base.getindex(A::DebugMPISharedArray{T, N}, I::Vararg{mk_int,N}) where {T, N}
        @debug_track_initialized begin
            if !all(A.is_initialized[I...] .== 1)
                if A.creation_stack_trace != ""
                    error("Shared memory array read at $I before being initialized. "
                          * "Array was created at:\n"
                          * A.creation_stack_trace)
                else
                    error("Shared memory array read at $I before being initialized. "
                          * "Enable `debug_track_array_allocate_location` to track where "
                          * "array was created.")
                end
            end
        end
        A.is_read[I...] = true
        A.accessed[] = true
        return getindex(A.data, I...)
    end
    function Base.setindex!(A::DebugMPISharedArray{T, N}, v::Number, I::Vararg{mk_int,N}) where {T, N}
        @debug_track_initialized begin
            A.is_initialized[I...] = 1
        end
        A.is_written[I...] = true
        A.accessed[] = true
        return setindex!(A.data, v, I...)
    end
    # Overload Base.convert() so that it is forbidden to convert a DebugMPISharedArray
    # to Array. If this happens, it can cause surprising results (i.e. debug checks do
    # not run), and it should never be useful.
    # Define 3 versions here because Julia dispatches to the most specific applicable
    # method, so need to make sure these are the most specific versions, regardless of
    # how the type passed to the first argument was defined.
    function Base.convert(::Type{Array{T,N}} where {T,N}, a::DebugMPISharedArray)
        error("Forbidden to convert DebugMPISharedArray to Array - this would "
              * "silently disable the debug checks")
    end
    function Base.convert(::Type{Array{T}} where {T}, a::DebugMPISharedArray)
        error("Forbidden to convert DebugMPISharedArray to Array - this would "
              * "silently disable the debug checks")
    end
    function Base.convert(::Type{Array}, a::DebugMPISharedArray)
        error("Forbidden to convert DebugMPISharedArray to Array - this would "
              * "silently disable the debug checks")
    end

    # Explicit overload for view() so the result is a DebugMPISharedArray
    import Base: view
    function view(A::DebugMPISharedArray, inds...)
        return DebugMPISharedArray(
            (isa(getfield(A, name), AbstractArray) ?
             view(getfield(A, name), inds...) :
             name === :dim_names ?
             Tuple(getfield(A, name)[i] for (i, ind) ∈ enumerate(inds) if !isa(ind, Integer)) :
             getfield(A, name)
             for name ∈ fieldnames(typeof(A)))...)
    end

    # Explicit overload for vec() so the result is a DebugMPISharedArray
    import Base: vec
    function vec(A::DebugMPISharedArray)
        return DebugMPISharedArray(
            (isa(getfield(A, name), AbstractArray) ?
             vec(getfield(A, name)) :
             name === :dim_names ?
             (:flattened_dim,) :
             getfield(A, name)
             for name ∈ fieldnames(typeof(A)))...)
    end

    # Explicit overload to avoid array when using DebugMPISharedArray Y, B and
    # SparseArray A
    import LinearAlgebra: ldiv!, Factorization
    function ldiv!(Y::DebugMPISharedArray, A::Factorization, B::DebugMPISharedArray)
        @debug_track_initialized begin
            Y.is_initialized .= 1
        end
        Y.is_written .= true
        Y.accessed[] = true
        return ldiv!(Y.data, A, B.data)
    end

    import MPI: Buffer
    function Buffer(A::DebugMPISharedArray)
        @debug_track_initialized begin
            A.is_initialized .= 1
        end
        A.is_read .= true
        A.is_written .= true
        A.accessed[] = true
        return Buffer(A.data)
    end

    # Keep a global Vector of references to all created DebugMPISharedArray
    # instances, so their is_read and is_written members can be checked and
    # reset by _block_synchronize()
    const global_debugmpisharedarray_store = Vector{DebugMPISharedArray}(undef, 0)
    # 'anysv' regions require a separate array store, because within an anysv region,
    # processes in the same shared memory block may still not be synchronized if they are
    # in different anysv sub-blocks, so debug checks within an anysv region should only
    # consider the anysv-specific arrays.
    const global_anysv_debugmpisharedarray_store = Vector{DebugMPISharedArray}(undef, 0)
    # 'anyzv' regions require a separate array store, because within an anyzv region,
    # processes in the same shared memory block may still not be synchronized if they are
    # in different anyzv sub-blocks, so debug checks within an anyzv region should only
    # consider the anyzv-specific arrays.
    const global_anyzv_debugmpisharedarray_store = Vector{DebugMPISharedArray}(undef, 0)
end

"""
    allocate_shared(T; comm=nothing, maybe_debug=true, kwargs...)
    allocate_shared(T, dims...; comm=nothing, maybe_debug=true)

Get a shared-memory array of `mk_float` (shared by all processes in a 'block').

Create a shared-memory array using `MPI.Win_allocate_shared()`. Pointer to the memory
allocated is wrapped in a Julia array. Memory is not managed by the Julia array though.
A reference to the `MPI.Win` needs to be freed - this is done by saving the `MPI.Win`
into a `Vector` in the `Communication` module, which has all its entries freed by the
`finalize_comms!()` function, which should be called when `moment_kinetics` is done
running a simulation/test.

Arguments
---------
kwargs - mk_int
    Dimensions must be named to support shared-memory debugging tools. They can be either
    passed as `name=dim_size` keyword arguments (`coordinate` objects can also be passed
    as the kwarg values, for convenienc), or using `dims`.
    Dimensions of the array to be created. Dimensions passed define the size of the
    array which is being handled by the 'block' (rather than the global array, or a
    subset for a single process).
dims - coordinate, NamedTuple, Pair{Symbol,Integer}, or Pair{String,Integer}
    Alternative to `kwargs`. May be passed `coordinate`, `NamedTuple`,
    `Pair{Symbol,Integer}` (enter as e.g. `:z => n` where `n` is an `mk_int`),
    `Pair{String,Integer}` (enter as e.g. `"z" => n` where `n` is an `mk_int`),
    `Pair{Symbol,coordinate}`, `Pair{String,coordinate}`, `Pair{Symbol,NamedTuple}`, or
    `Pair{String,NamedTuple}` arguments from which the name and size of the name and size
    of the dimension will be extracted.
comm - `MPI.Comm`, default `comm_block[]`
    MPI communicator containing the processes that share the array.
maybe_debug - Bool
    Can be set to `false` to force not creating a DebugMPISharedArray when debugging is
    active. This avoids recursion when including a shared-memory array as a member of a
    DebugMPISharedArray for debugging purposes.

Returns
-------
Array{mk_float}
"""
function allocate_shared end

function allocate_shared(T::Type, dim1, dims...; comm=nothing, maybe_debug=true)
    function standardise_argument(a)
        if isa(a, coordinate)
            return Symbol(a.name) => a.n
        elseif isa(a, NamedTuple)
            return Symbol(a.name) => a.n
        elseif isa(a, Pair)
            if isa(a[2], coordinate)
                return Symbol(a[1]) => a[2].n
            elseif isa(a[2], NamedTuple)
                return Symbol(a[1]) => a[2].n
            elseif isa(a[2], Integer)
                return Symbol(a[1]) => mk_int(a[2])
            else
                error("Incorrect argument $a to `allocate_shared`. Arguments should be "
                      * "coordinate, coordinate-like NamedTuple, or `:name=>n` "
                      * "(`Pair{Symbol,mk_int}`).")
            end
        else
            error("Incorrect argument $a to `allocate_shared`. Arguments should be "
                  * "coordinate, coordinate-like NamedTuple, or `:name=>n` "
                  * "(`Pair{Symbol,mk_int}`).")
        end
    end
    return _allocate_shared_internal(T, (standardise_argument(d)
                                         for d ∈ (dim1, dims...))...;
                                     comm=comm, maybe_debug=maybe_debug)
end

function allocate_shared(T::Type; comm=nothing, maybe_debug=true, kwargs...)
    function get_int(a)
        if isa(a, coordinate)
            return a.n
        elseif isa(a, NamedTuple)
            return mk_int(a.n)
        elseif isa(a, Integer)
            return mk_int(a)
        else
            error("Unrecognised type of argument $a")
        end
    end
    return _allocate_shared_internal(T, (k => get_int(v) for (k,v) ∈ kwargs)...;
                                     comm=comm, maybe_debug=maybe_debug)
end

# Note cannot just use `kwargs...` for the inner version, because we might want repeated
# dimension names in some cases, but repeated `kwargs` are not allowed.
function _allocate_shared_internal(T::Type, dims_info::Pair{Symbol,mk_int}...;
                                   comm=nothing, maybe_debug=true)
    if maybe_debug
        dim_names = Tuple(d[1] for d ∈ dims_info)
    end
    dims = Tuple(d[2] for d ∈ dims_info)
    if comm === nothing
        comm = comm_block[]
    elseif comm == MPI.COMM_NULL
        # If `comm` is a null communicator (on this process), then this array is just a
        # dummy that will not be used.
        array = Array{T}(undef, (0 for _ ∈ dims)...)

        @debug_shared_array begin
            # If @debug_shared_array is active, create DebugMPISharedArray instead of Array
            if maybe_debug
                array = DebugMPISharedArray(array, comm, dim_names)
            end
        end

        return array
    end
    br = MPI.Comm_rank(comm)
    bs = MPI.Comm_size(comm)
    n = prod(dims)

    if n == 0
        # Special handling as some MPI implementations cause errors when allocating a
        # size-zero array
        array = Array{T}(undef, dims...)

        @debug_shared_array begin
            # If @debug_shared_array is active, create DebugMPISharedArray instead of Array
            if maybe_debug
                array = DebugMPISharedArray(array, comm, dim_names)
            end
        end

        return array
    end

    if br == 0
        # Allocate points on rank-0 for simplicity
        dims_local = dims
    else
        dims_local = Tuple(0 for _ ∈ dims)
    end

    @debug_shared_array_allocate begin
        # Check that allocate_shared was called from the same place on all ranks
        st = stacktrace()
        stackstring = string([string(s, "\n") for s ∈ st]...)

        # Only include file and line number in the string that we hash so that
        # function calls with different signatures are not seen as different
        # (e.g. time_advance!() with I/O arguments on rank-0 but not on other
        # ranks).
        signaturestring = string([string(s.file, s.line) for s ∈ st]...)

        hash = sha256(signaturestring)
        all_hashes = MPI.Allgather(hash, comm)
        l = length(hash)
        for i ∈ 1:length(all_hashes)÷l
            if all_hashes[(i - 1) * l + 1: i * l] != hash
                error("allocate_shared() called inconsistently\n",
                      "rank $(block_rank[]) called from:\n",
                      stackstring)
            end
        end
    end

    win, array_temp = MPI.Win_allocate_shared(Array{T}, dims_local, comm)

    # Array is allocated contiguously, but `array_temp` contains only the 'locally owned'
    # part.  We want to use as a shared array, so want to wrap the entire shared array.
    # Get array from rank-0 process, which 'owns' the whole array.
    array = MPI.Win_shared_query(Array{T}, dims, win; rank=0)

    # Don't think `win::MPI.Win` knows about the type of the pointer (its concrete type
    # is something like `MPI.Win(Ptr{Nothing} @0x00000000033affd0)`), so it's fine to
    # put them all in the same global_Win_store - this won't introduce type instability
    push!(global_Win_store, win)

    @debug_shared_array begin
        # If @debug_shared_array is active, create DebugMPISharedArray instead of Array
        if maybe_debug
            debug_array = DebugMPISharedArray(array, comm, dim_names)
            if comm == comm_anysv_subblock[]
                push!(global_anysv_debugmpisharedarray_store, debug_array)
            elseif comm == comm_anyzv_subblock[]
                push!(global_anyzv_debugmpisharedarray_store, debug_array)
            else
                push!(global_debugmpisharedarray_store, debug_array)
            end
            return debug_array
        end
    end

    return array
end

@debug_detect_redundant_block_synchronize begin
    """
    """
    const previous_block_synchronize_stackstring = Ref{String}("")

    """
    """
    const debug_detect_redundant_is_active = Ref{Bool}(false)
end
@debug_shared_array begin
    # This function is used in two ways:
    # 1. To throw an error when @debug_shared_array is activated.
    # 2. To show when an error would be raised at one _block_synchronize() call if
    #    the previous _block_synchronize() call were removed.
    """
    Check whether a sharerd-memory array has been accessed incorrectly

    Arguments
    ---------
    array : DebugMPISharedArray
        The array being checked
    check_redundant : Bool, default false
        If set to true, the function is being used to check whether the previous call to
        _block_synchronize() was redundant. In this case, checks the combinations of
        `is_read` with `previous_is_read` and `is_written` with `previous_is_written`,
        and returns `false` (rather than calling `error()`) if the combination would
        cause an error.
    """
    function debug_check_shared_array(array; check_redundant=false, comm=comm_block[])
        comm_rank = MPI.Comm_rank(comm)
        comm_size = MPI.Comm_size(comm)

        dims = size(array)
        global_dims = (dims..., comm_size)

        is_read = array.is_read
        is_written = array.is_written
        @debug_detect_redundant_block_synchronize begin
            if check_redundant
                # Note `||` cannot broadcast because it 'short-circuits', so only
                # accepts scalar Bool arguments. Therefore we need to use `.|`
                # (broadcasted, bit-wise or) to combine arrays of Bool.  Need to
                # explicitly convert the result to a Vector{Bool}, because by default
                # the result is a BitVector, and passing BitVector to MPI.Allgather()
                # causes an error
                is_read = Array{Bool}(is_read .| array.previous_is_read)
                is_written = Array{Bool}(is_written .| array.previous_is_written)
            end
        end

        # Short-circuit if array has not been read or written at all
        any_accessed = MPI.Allreduce(array.accessed[], |, comm)
        array.accessed[] = false
        if !any_accessed
            return true
        end

        global_is_read = reshape(MPI.Allgather(is_read, comm),
                                 global_dims...)
        global_is_written = reshape(MPI.Allgather(is_written, comm),
                                    global_dims...)
        for i ∈ CartesianIndices(array)
            n_reads = sum(global_is_read[i, :])
            n_writes = sum(global_is_written[i, :])
            if n_writes > 1
                if check_redundant
                    # In the @debug_detect_redundant_block_synchronize case,
                    # cannot throw an error() because the we need to check that
                    # the `_block_synchronize()` call was redundant on all
                    # processes, not just on this one.
                    return false
                else
                    if array.creation_stack_trace != ""
                        error("Shared memory array written at $i from multiple ranks "
                              * "between calls to _block_synchronize(). Array was "
                              * "created at:\n"
                              * array.creation_stack_trace)
                    else
                        error("Shared memory array written at $i from multiple ranks "
                              * "between calls to _block_synchronize(). Enable "
                              * "`debug_track_array_allocate_location` to track where "
                              * "array was created.")
                    end
                end
            elseif n_writes == 1 && n_reads > 0
                if global_is_written[i, comm_rank + 1]
                    read_procs = Vector{mk_int}(undef, 0)
                    for (r, is_read) ∈ enumerate(global_is_read[i, :])
                        if r == comm_rank + 1
                            continue
                        elseif is_read
                            push!(read_procs, r)
                        end
                    end
                    if length(read_procs) > 0
                        if check_redundant
                            # In the @debug_detect_redundant_block_synchronize case,
                            # cannot throw an error() because the we need to check that
                            # the `_block_synchronize()` call was redundant on all
                            # processes, not just on this one.
                            return false
                        else
                            # 'rank' is 0-based, but read_procs was 1-based, so
                            # correct
                            read_procs .-= 1
                            error("Shared memory array was written at $i on rank "
                                  * "$(comm_rank) but read from ranks "
                                  * "$read_procs Array was created at:\n"
                                  * array.creation_stack_trace)
                        end
                    end
                end
            end
        end
        return true
    end

    """
    Raises an error if any array has been accessed incorrectly since the previous call
    to _block_synchronize()

    Can be added when debugging to help pin down where an error occurs.
    """
    function debug_check_shared_memory(; comm=comm_block[], kwargs...)
        if comm == comm_anysv_subblock[]
            for array ∈ global_anysv_debugmpisharedarray_store
                debug_check_shared_array(array; comm=comm, kwargs...)
            end
        elseif comm == comm_anyzv_subblock[]
            for array ∈ global_anyzv_debugmpisharedarray_store
                debug_check_shared_array(array; comm=comm, kwargs...)
            end
        else
            for array ∈ global_debugmpisharedarray_store
                debug_check_shared_array(array; comm=comm, kwargs...)
            end
        end
        return nothing
    end
end

"""
Call an MPI Barrier for all processors in a block.

Used to synchronise processors that are working on the same shared-memory array(s)
between operations, to avoid race conditions. Should be (much) cheaper than a global MPI
Barrier because it only requires communication within a single node.
"""
macro _block_synchronize()
    id_hash = @debug_block_synchronize_quick_ifelse(
                   hash(string(@__FILE__, @__LINE__)),
                   nothing
                  )
    return :( _block_synchronize($id_hash) )
end

"""
Internal function to be called by @_block_synchronize() and @begin_*_region().
`call_site` will be either `nothing` or a hash of the file and line number of the calling
site of the function.

Note: some debugging code currently assumes that if _block_synchronize() is called on one
block, it is called simultaneously on all blocks. It seems likely that this will always be
true, but if it ever changes (i.e. different blocks doing totally different work), the
debugging routines need to be updated.
"""
@timeit_debug global_timer _block_synchronize(call_site::Union{Nothing,UInt64}) = begin
    MPI.Barrier(comm_block[])

    @debug_block_synchronize_backtrace begin
        st = stacktrace()
        stackstring = string([string(s, "\n") for s ∈ st]...)

        # Only include file and line number in the string that we hash so that
        # function calls with different signatures are not seen as different
        # (e.g. time_advance!() with I/O arguments on rank-0 but not on other
        # ranks).
        signaturestring = string([string(s.file, s.line) for s ∈ st]...)

        hash = sha256(signaturestring)
        all_hashes = reshape(MPI.Allgather(hash, comm_block[]), length(hash),
                             block_size[])
        for i ∈ 1:block_size[]
            h = all_hashes[:, i]
            if h != hash
                error("_block_synchronize() called inconsistently\n",
                      "rank $(block_rank[]) called from:\n",
                      stackstring)
            end
        end
    end

    @debug_block_synchronize_quick begin
        if call_site === nothing
            error("Got call_site=nothing. This should not happen when debugging with "
                  * "@debug_block_synchronize_quick.")
        end
        all_hashes = MPI.Allgather(call_site, comm_block[])
        if !all(h -> h == all_hashes[1], all_hashes)
            error("_block_synchronize() called inconsistently")
        end
    end

    @debug_shared_array begin
        # Check for potential race conditions:
        # * Between _block_synchronize() any element of an array should be written to by
        #   at most one rank.
        # * If an element is not written to, any rank can read it.
        # * If an element is written to, only the rank that writes to it should read it.
        #
        @debug_detect_redundant_block_synchronize previous_was_unnecessary = true
        for array ∈ global_debugmpisharedarray_store

            debug_check_shared_array(array)

            @debug_detect_redundant_block_synchronize begin
                # debug_detect_redundant_is_active[] is set to true at the beginning of
                # time_advance!() so that we do not do these checks during
                # initialization: they cause problems with @debug_initialize_NaN during
                # array allocation; generally it does not matter if there are a few
                # extra _block_synchronize() calls during initialization, so it is not
                # worth the effort to trim them down to the absolute minimum.
                if debug_detect_redundant_is_active[]

                    if !debug_check_shared_array(array, check_redundant=true)
                        # If there was a failure for at least one array, the previous
                        # _block_synchronize was necessary - if the previous call was not
                        # there, for this array array.is_read and array.is_written would
                        # have the values of combined_is_read and combined_is_written,
                        # and would fail the debug_check_shared_array() above this
                        # @debug_detect_redundant_block_synchronize block.
                        previous_was_unnecessary = false
                    end

                    array.previous_is_read .= array.is_read
                    array.previous_is_written .= array.is_written
                else
                    # If checking is inactive, set as if at 'previous' the array was
                    # always read/written so that the next set of checks don't detect a
                    # 'redundant' call which is actually only 'redundant' just after an
                    # inactive region (e.g. initialisation or writing output).
                    array.previous_is_read .= true
                    array.previous_is_written .= true
                end
            end

            array.is_read .= false
            array.is_written .= false
        end
        @debug_detect_redundant_block_synchronize begin
            if debug_detect_redundant_is_active[]
                # Check the previous call was unnecessary on all processes, not just
                # this one
                previous_was_unnecessary = MPI.Allreduce(previous_was_unnecessary,
                                                         MPI.Op(&, Bool), comm_world)

                if (previous_was_unnecessary && global_size[] > 1)
                    # The intention of this debug block is to detect when calls to
                    # _block_synchronize() are not necessary and can be removed. It's not
                    # obvious that this will always work - it might be that a call to
                    # _block_synchronize() is necessary with some options, but not
                    # necessary with others. Hopefully it will be possible to handle
                    # this by moving the _block_synchronize() call inside appropriate
                    # if-clauses. If not, it might be necessary to define something like
                    # _block_synchronize_ignore_redundant() to skip this check because
                    # the check is ambiguous.
                    #
                    # If we are running in serial (global_size[] == 1), then none of the
                    # _block_synchronize() calls are 'necessary', so this check is not
                    # useful.
                    error("Previous call to _block_synchronize() was not necessary. "
                          * "Call was from:\n"
                          * "$(previous_block_synchronize_stackstring[])")
                end

                st = stacktrace()
                stackstring = string([string(s, "\n") for s ∈ st]...)
                previous_block_synchronize_stackstring[] = stackstring
            end
        end

        # Also check 'anysv' and 'anyzv' arrays, as these are synchronized by this call.
        # `missing` passed as the call_site argument here indicates that the check of
        # call_site has already been done.
        _anysv_subblock_synchronize(missing)
        _anyzv_subblock_synchronize(missing)

        MPI.Barrier(comm_block[])
    end
end

"""
Call an MPI Barrier for all processors in an 'anysv' sub-block.

The 'anysv' region is used to parallelise the collision operator. See
[`moment_kinetics.looping.get_best_anysv_split`](@ref).

Used to synchronise processors that are working on the same shared-memory array(s)
between operations, to avoid race conditions. Should be even cheaper than
[`@_block_synchronize`](@ref) because it only requires communication on a smaller
communicator.

Note: `_anysv_subblock_synchronize()` may be called different numbers of times on different
sub-blocks, depending on how the species and spatial dimensions are split up.
`@debug_detect_redundant_block_synchronize` is not implemented (yet?) for
`_anysv_subblock_synchronize()`.
"""
macro _anysv_subblock_synchronize()
    id_hash = @debug_block_synchronize_quick_ifelse(
                   hash(string(@__FILE__, @__LINE__)),
                   nothing
                  )
    return :( _anysv_subblock_synchronize($id_hash) )
end

"""
Internal function called by `anysv` synchronization macros.
"""
function _anysv_subblock_synchronize(call_site::Union{Nothing,Missing,UInt64})
    if comm_anysv_subblock[] == MPI.COMM_NULL
        # No synchronization to do for a null communicator
        return nothing
    end

    MPI.Barrier(comm_anysv_subblock[])

    @debug_block_synchronize_backtrace begin
        st = stacktrace()
        stackstring = string([string(s, "\n") for s ∈ st]...)

        # Only include file and line number in the string that we hash so that
        # function calls with different signatures are not seen as different
        # (e.g. time_advance!() with I/O arguments on rank-0 but not on other
        # ranks).
        signaturestring = string([string(s.file, s.line) for s ∈ st]...)

        hash = sha256(signaturestring)
        all_hashes = reshape(MPI.Allgather(hash, comm_anysv_subblock[]), length(hash),
                             MPI.Comm_size(comm_anysv_subblock[]))
        for i ∈ 1:anysv_subblock_size[]
            h = all_hashes[:, i]
            if h != hash
                error("_anysv_subblock_synchronize() called inconsistently\n",
                      "rank $(block_rank[]) called from:\n",
                      stackstring)
            end
        end
    end

    @debug_block_synchronize_quick begin
        if call_site === nothing
            error("Got call_site=nothing. This should not happen when debugging with "
                  * "@debug_block_synchronize_quick.")
        end
        # If call_site===missing, then this function was called from inside
        # _block_synchronize(), and the call site was already checked there.
        if call_site !== missing
            all_hashes = MPI.Allgather(call_site, comm_anysv_subblock[])
            if !all(h -> h == all_hashes[1], all_hashes)
                error("_anysv_subblock_synchronize() called inconsistently")
            end
        end
    end

    @debug_shared_array begin
        # Check for potential race conditions:
        # * Between _anysv_subblock_synchronize() any element of an array should be
        #   written to by at most one rank.
        # * If an element is not written to, any rank can read it.
        # * If an element is written to, only the rank that writes to it should read it.
        #
        # Check each array that was shared across comm_anysv_subblock[] as its
        # communicator (global_anysv_debugmpisharedarray_store). Also check the slice of
        # each array shared across comm_block[] that may be accessed inside an anysv
        # region (i.e. the slice including only the r- and z-indices that are accessed
        # inside the local anysv region). Assume that arrays allocated with other subblock
        # communicators (e.g. comm_anyzv_subblock[]) are incompatible with an anysv
        # subblock, so will never be accessed and do not need to be checked.
        #
        # Note that slicing of arrays from `global_debugmpisharedarray_store` is likely to
        # cause errors if it is done before 'looping' is set up, because
        # `loop_ranges_store` will be uninitialised or contain values from a previous run.
        # Therefore `@_anysv_subblock_synchronize()` should not be called before
        # `setup_loop_ranges!()` has been.
        local_r_inds = loop_ranges_store[(:anysv,)].r
        local_z_inds = loop_ranges_store[(:anysv,)].z
        function get_local_slice(array)
            if !(:r ∈ array.dim_names && :z ∈ array.dim_names)
                # Array does not have both r- and z-dimensions, so there is no slice that
                # is allowed to be accessed inside an anysv region.
                return nothing
            end

            r_dims = findall(array.dim_names .== :r)
            for dim_ind ∈ reverse(r_dims)
                array = selectdim(array, dim_ind, local_r_inds)
            end

            z_dims = findall(array.dim_names .== :z)
            for dim_ind ∈ reverse(z_dims)
                array = selectdim(array, dim_ind, local_z_inds)
            end
            return array
        end
        @debug_detect_redundant_block_synchronize previous_was_unnecessary = true
        arrays_to_check = global_anysv_debugmpisharedarray_store
        if call_site !== missing
            # This is an actual anysv subblock synchronization, not just a call within
            # _block_synchronize() to check anysv arrays.
            arrays_to_check = (arrays_to_check..., (get_local_slice(a) for a ∈
                                                    global_debugmpisharedarray_store)...)
        end
        for array ∈ arrays_to_check
            if array === nothing
                # `nothing` is returned by get_local_slice() if the array did not have
                # both r- and z-dimensions, so is not allowed to be accessed inside an
                # anysv region.
                continue
            end

            debug_check_shared_array(array; comm=comm_anysv_subblock[])

            @debug_detect_redundant_block_synchronize begin
                # debug_detect_redundant_is_active[] is set to true at the beginning of
                # time_advance!() so that we do not do these checks during
                # initialization: they cause problems with @debug_initialize_NaN during
                # array allocation; generally it does not matter if there are a few
                # extra _block_synchronize() calls during initialization, so it is not
                # worth the effort to trim them down to the absolute minimum.
                if debug_detect_redundant_is_active[]

                    if !debug_check_shared_array(array; check_redundant=true,
                                                 comm_anysv_subblock)
                        # If there was a failure for at least one array, the previous
                        # _block_synchronize was necessary - if the previous call was not
                        # there, for this array array.is_read and array.is_written would
                        # have the values of combined_is_read and combined_is_written,
                        # and would fail the debug_check_shared_array() above this
                        # @debug_detect_redundant_block_synchronize block.
                        previous_was_unnecessary = false
                    end

                    array.previous_is_read .= array.is_read
                    array.previous_is_written .= array.is_written
                else
                    # If checking is inactive, set as if at 'previous' the array was
                    # always read/written so that the next set of checks don't detect a
                    # 'redundant' call which is actually only 'redundant' just after an
                    # inactive region (e.g. initialisation or writing output).
                    array.previous_is_read .= true
                    array.previous_is_written .= true
                end
            end

            array.is_read .= false
            array.is_written .= false
        end
        @debug_detect_redundant_block_synchronize begin
            if debug_detect_redundant_is_active[]
                # Check the previous call was unnecessary on all processes, not just
                # this one
                previous_was_unnecessary = MPI.Allreduce(previous_was_unnecessary,
                                                         MPI.Op(&, Bool), comm_anysv_subblock[])

                if (previous_was_unnecessary && global_size[] > 1)
                    # The intention of this debug block is to detect when calls to
                    # _block_synchronize() are not necessary and can be removed. It's not
                    # obvious that this will always work - it might be that a call to
                    # _block_synchronize() is necessary with some options, but not
                    # necessary with others. Hopefully it will be possible to handle
                    # this by moving the _block_synchronize() call inside appropriate
                    # if-clauses. If not, it might be necessary to define something like
                    # _block_synchronize_ignore_redundant() to skip this check because
                    # the check is ambiguous.
                    #
                    # If we are running in serial (global_size[] == 1), then none of the
                    # _block_synchronize() calls are 'necessary', so this check is not
                    # useful.
                    error("Previous call to _block_synchronize() was not necessary. "
                          * "Call was from:\n"
                          * "$(previous_block_synchronize_stackstring[])")
                end

                st = stacktrace()
                stackstring = string([string(s, "\n") for s ∈ st]...)
                previous_block_synchronize_stackstring[] = stackstring
            end
        end

        MPI.Barrier(comm_anysv_subblock[])
    end
end

"""
Call an MPI Barrier for all processors in an 'anyzv' sub-block.

The 'anyzv' region is used to parallelise the kinetic electron implicit solve. See
[`moment_kinetics.looping.get_best_anyzv_split`](@ref).

Used to synchronise processors that are working on the same shared-memory array(s)
between operations, to avoid race conditions. Should be even cheaper than
[`@_block_synchronize`](@ref) because it only requires communication on a smaller
communicator.

Note: `_anyzv_subblock_synchronize()` may be called different numbers of times on different
sub-blocks, depending on iteration counts and how the r-dimension is split up.
`@debug_detect_redundant_block_synchronize` is not implemented (yet?) for
`_anyzv_subblock_synchronize()`.
"""
macro _anyzv_subblock_synchronize()
    id_hash = @debug_block_synchronize_quick_ifelse(
                   hash(string(@__FILE__, @__LINE__)),
                   nothing
                  )
    return :( _anyzv_subblock_synchronize($id_hash) )
end

"""
Internal function called by `anyzv` synchronization macros.
"""
function _anyzv_subblock_synchronize(call_site::Union{Nothing,Missing,UInt64})
    if comm_anyzv_subblock[] == MPI.COMM_NULL
        # No synchronization to do for a null communicator
        return nothing
    end

    MPI.Barrier(comm_anyzv_subblock[])

    @debug_block_synchronize_backtrace begin
        st = stacktrace()
        stackstring = string([string(s, "\n") for s ∈ st]...)

        # Only include file and line number in the string that we hash so that
        # function calls with different signatures are not seen as different
        # (e.g. time_advance!() with I/O arguments on rank-0 but not on other
        # ranks).
        signaturestring = string([string(s.file, s.line) for s ∈ st]...)

        hash = sha256(signaturestring)
        all_hashes = reshape(MPI.Allgather(hash, comm_anyzv_subblock[]), length(hash),
                             MPI.Comm_size(comm_anyzv_subblock[]))
        for i ∈ 1:anyzv_subblock_size[]
            h = all_hashes[:, i]
            if h != hash
                error("_anyzv_subblock_synchronize() called inconsistently\n",
                      "rank $(block_rank[]) called from:\n",
                      stackstring)
            end
        end
    end

    @debug_block_synchronize_quick begin
        if call_site === nothing
            error("Got call_site=nothing. This should not happen when debugging with "
                  * "@debug_block_synchronize_quick.")
        end
        # If call_site===missing, then this function was called from inside
        # _block_synchronize(), and the call site was already checked there.
        if call_site !== missing
            all_hashes = MPI.Allgather(call_site, comm_anyzv_subblock[])
            if !all(h -> h == all_hashes[1], all_hashes)
                error("_anyzv_subblock_synchronize() called inconsistently")
            end
        end
    end

    @debug_shared_array begin
        # Check for potential race conditions:
        # * Between _anyzv_subblock_synchronize() any element of an array should be
        #   written to by at most one rank.
        # * If an element is not written to, any rank can read it.
        # * If an element is written to, only the rank that writes to it should read it.
        #
        # Check each array that was shared across comm_anyzv_subblock[] as its
        # communicator (global_anyzv_debugmpisharedarray_store). Also check the slice of
        # each array shared across comm_block[] that may be accessed inside an anyzv
        # region (i.e. the slice including only the r-indices that are accessed inside the
        # local anyzv region). Assume that arrays allocated with other subblock
        # communicators (e.g. comm_anysv_subblock[]) are incompatible with an anyzv
        # subblock, so will never be accessed and do not need to be checked.
        #
        # Note that slicing of arrays from `global_debugmpisharedarray_store` is likely to
        # cause errors if it is done before 'looping' is set up, because
        # `loop_ranges_store` will be uninitialised or contain values from a previous run.
        # Therefore `@_anyzv_subblock_synchronize()` should not be called before
        # `setup_loop_ranges!()` has been.
        local_r_inds = loop_ranges_store[(:anyzv,)].r
        function get_local_slice(array)
            if :r ∉ array.dim_names
                # Array does not have an r-dimension, so there is no slice that is allowed
                # to be accessed inside an anyzv region.
                return nothing
            end

            r_dims = findall(array.dim_names .== :r)
            for dim_ind ∈ reverse(r_dims)
                array = selectdim(array, dim_ind, local_r_inds)
            end
            return array
        end
        @debug_detect_redundant_block_synchronize previous_was_unnecessary = true
        arrays_to_check = global_anyzv_debugmpisharedarray_store
        if call_site !== missing
            # This is an actual anyzv subblock synchronization, not just a call within
            # _block_synchronize() to check anyzv arrays.
            arrays_to_check = (arrays_to_check..., (get_local_slice(a) for a ∈
                                                    global_debugmpisharedarray_store)...)
        end
        for array ∈ arrays_to_check
            if array === nothing
                # `nothing` is returned by get_local_slice() if the array did not have an
                # r-dimension, so is not allowed to be accessed inside an anyzv region.
                continue
            end

            debug_check_shared_array(array; comm=comm_anyzv_subblock[])

            @debug_detect_redundant_block_synchronize begin
                # debug_detect_redundant_is_active[] is set to true at the beginning of
                # time_advance!() so that we do not do these checks during
                # initialization: they cause problems with @debug_initialize_NaN during
                # array allocation; generally it does not matter if there are a few
                # extra _block_synchronize() calls during initialization, so it is not
                # worth the effort to trim them down to the absolute minimum.
                if debug_detect_redundant_is_active[]

                    if !debug_check_shared_array(array; check_redundant=true,
                                                 comm_anyzv_subblock)
                        # If there was a failure for at least one array, the previous
                        # _block_synchronize was necessary - if the previous call was not
                        # there, for this array array.is_read and array.is_written would
                        # have the values of combined_is_read and combined_is_written,
                        # and would fail the debug_check_shared_array() above this
                        # @debug_detect_redundant_block_synchronize block.
                        previous_was_unnecessary = false
                    end

                    array.previous_is_read .= array.is_read
                    array.previous_is_written .= array.is_written
                else
                    # If checking is inactive, set as if at 'previous' the array was
                    # always read/written so that the next set of checks don't detect a
                    # 'redundant' call which is actually only 'redundant' just after an
                    # inactive region (e.g. initialisation or writing output).
                    array.previous_is_read .= true
                    array.previous_is_written .= true
                end
            end

            array.is_read .= false
            array.is_written .= false
        end
        @debug_detect_redundant_block_synchronize begin
            if debug_detect_redundant_is_active[]
                # Check the previous call was unnecessary on all processes, not just
                # this one
                previous_was_unnecessary = MPI.Allreduce(previous_was_unnecessary,
                                                         MPI.Op(&, Bool), comm_anyzv_subblock[])

                if (previous_was_unnecessary && global_size[] > 1)
                    # The intention of this debug block is to detect when calls to
                    # _block_synchronize() are not necessary and can be removed. It's not
                    # obvious that this will always work - it might be that a call to
                    # _block_synchronize() is necessary with some options, but not
                    # necessary with others. Hopefully it will be possible to handle
                    # this by moving the _block_synchronize() call inside appropriate
                    # if-clauses. If not, it might be necessary to define something like
                    # _block_synchronize_ignore_redundant() to skip this check because
                    # the check is ambiguous.
                    #
                    # If we are running in serial (global_size[] == 1), then none of the
                    # _block_synchronize() calls are 'necessary', so this check is not
                    # useful.
                    error("Previous call to _block_synchronize() was not necessary. "
                          * "Call was from:\n"
                          * "$(previous_block_synchronize_stackstring[])")
                end

                st = stacktrace()
                stackstring = string([string(s, "\n") for s ∈ st]...)
                previous_block_synchronize_stackstring[] = stackstring
            end
        end

        MPI.Barrier(comm_anyzv_subblock[])
    end
end

"""
    halo_swap!(x::AbstractArray, r, z)

Enforce consistency of 'halo cells' - i.e. the grid points on block boundaries (in the
\$r\$ and \$z\$ directions) that are shared by the grids owned by two processes (or more
on block corners).

For consistency when adding random noise, just chooses the value from the upper/outer
process rather than averaging.
"""
function halo_swap! end

# If we want a different operation than just taking the upper/outer value, could add an
# optional `op` argument, and if it is passed do an `MPI.Allreduce!()` with that operation
# instead of the Isend/Irecv.

# Version for fields or electron moments
function halo_swap!(x::AbstractArray{T,2} where T, r, z)
    if block_rank[] == 0
        # Send r-boundary inward.
        req_r1 = MPI.Irecv!(@view(x[:,end]), r.comm; source=r.nextrank)
        req_r2 = MPI.Isend(@view(x[:,1]), r.comm; dest=r.prevrank)

        # Need to complete the r-communication before starting the z-communication to
        # ensure block-corner points are consistently communicated.
        MPI.Waitall([req_r1, req_r2])

        # Send z-boundary downward.
        req_z1 = MPI.Irecv!(@view(x[end,:]), z.comm; source=z.nextrank)
        req_z2 = MPI.Isend(@view(x[1,:]), z.comm; dest=z.prevrank)

        MPI.Waitall([req_z1, req_z2])
    end
end

# Version for ion or neutral moments
function halo_swap!(x::AbstractArray{T,3} where T, r, z)
    if block_rank[] == 0
        # Send r-boundary inward.
        req_r1 = MPI.Irecv!(@view(x[:,end,:]), r.comm; source=r.nextrank)
        req_r2 = MPI.Isend(@view(x[:,1,:]), r.comm; dest=r.prevrank)

        # Need to complete the r-communication before starting the z-communication to
        # ensure block-corner points are consistently communicated.
        MPI.Waitall([req_r1, req_r2])

        # Send z-boundary downward.
        req_z1 = MPI.Irecv!(@view(x[end,:,:]), z.comm; source=z.nextrank)
        req_z2 = MPI.Isend(@view(x[1,:,:]), z.comm; dest=z.prevrank)

        MPI.Waitall([req_z1, req_z2])
    end
end

# Version for electron distribution function
function halo_swap!(x::AbstractArray{T,4} where T, r, z)
    if block_rank[] == 0
        # Send r-boundary inward.
        req_r1 = MPI.Irecv!(@view(x[:,:,:,end]), r.comm; source=r.nextrank)
        req_r2 = MPI.Isend(@view(x[:,:,:,1]), r.comm; dest=r.prevrank)

        # Need to complete the r-communication before starting the z-communication to
        # ensure block-corner points are consistently communicated.
        MPI.Waitall([req_r1, req_r2])

        # Send z-boundary downward.
        req_z1 = MPI.Irecv!(@view(x[:,:,end,:]), z.comm; source=z.nextrank)
        req_z2 = MPI.Isend(@view(x[:,:,1,:]), z.comm; dest=z.prevrank)

        MPI.Waitall([req_z1, req_z2])
    end
end

# Version for ion distribution function
function halo_swap!(x::AbstractArray{T,5} where T, r, z)
    if block_rank[] == 0
        # Send r-boundary inward.
        req_r1 = MPI.Irecv!(@view(x[:,:,:,end,:]), r.comm; source=r.nextrank)
        req_r2 = MPI.Isend(@view(x[:,:,:,1,:]), r.comm; dest=r.prevrank)

        # Need to complete the r-communication before starting the z-communication to
        # ensure block-corner points are consistently communicated.
        MPI.Waitall([req_r1, req_r2])

        # Send z-boundary downward.
        req_z1 = MPI.Irecv!(@view(x[:,:,end,:,:]), z.comm; source=z.nextrank)
        req_z2 = MPI.Isend(@view(x[:,:,1,:,:]), z.comm; dest=z.prevrank)

        MPI.Waitall([req_z1, req_z2])
    end
end

# Version for neutral distribution function
function halo_swap!(x::AbstractArray{T,6} where T, r, z)
    if block_rank[] == 0
        # Send r-boundary outward.
        req_r1 = MPI.Irecv!(@view(x[:,:,:,:,end,:]), r.comm; source=r.nextrank)
        req_r2 = MPI.Isend(@view(x[:,:,:,:,1,:]), r.comm; dest=r.prevrank)

        # Need to complete the r-communication before starting the z-communication to
        # ensure block-corner points are consistently communicated.
        MPI.Waitall([req_r1, req_r2])

        # Send z-boundary downward.
        req_z1 = MPI.Irecv!(@view(x[:,:,:,end,:,:]), z.comm; source=z.nextrank)
        req_z2 = MPI.Isend(@view(x[:,:,:,1,:,:]), z.comm; dest=z.prevrank)

        MPI.Waitall([req_z1, req_z2])
    end
end

"""
Set up communications

Check that global variables are in the correct state (i.e. caches were emptied
correctly if they were used before).

Also does some set up for debugging routines, if they are active.
"""
function initialize_comms!()
    if length(global_Win_store) > 0
        free_shared_arrays()
    end

    @debug_detect_redundant_block_synchronize begin
        debug_detect_redundant_is_active[] = false
    end

    return nothing
end

"""
Clean up from communications

Do any needed clean-up for MPI, etc. Does not call `MPI.Finalize()` - this is called
anyway when Julia exits, and we do not want to call it explicitly so that multiple runs
can be done in a single Julia session.

Frees any shared-memory arrays.
"""
function finalize_comms!()
    free_shared_arrays()

    return nothing
end

"""
"""
function free_shared_arrays()
    @debug_shared_array resize!(global_debugmpisharedarray_store, 0)
    @debug_shared_array resize!(global_anysv_debugmpisharedarray_store, 0)
    @debug_shared_array resize!(global_anyzv_debugmpisharedarray_store, 0)

    for w ∈ global_Win_store
        MPI.free(w)
    end
    resize!(global_Win_store, 0)

    return nothing
end

end # communication
