export judiPhoto, judiInitialStateProjection

# Sizes extras
JUDI.space_src(N::NTuple{2, Integer}, nsrc::Integer) = AbstractSize((:src, :x, :z), (nsrc, N...))
JUDI.space_src(N::NTuple{3, Integer}, nsrc::Integer) = AbstractSize((:src, :x, :y, :z), (nsrc, N...))

struct judiInitialStateProjection{D} <: judiNoopOperator{D}
    m::AbstractSize
    n::AbstractSize
end

"""
    judiInitialStateProjection(model)

Construct the projection operator that sets the initial state into the wavefield for propagation.
This operator is a No-op operation that will propagate a [`judiInitialState`](@ref) if combined with a JUDI
propagator.
"""
judiInitialStateProjection(model, nsrc=1) = judiInitialStateProjection{eltype(model.m)}(space_src(model.n, nsrc), time_space(model.n))

struct judiPhoto{D, O} <: judiComposedPropagator{D, O}
    m::AbstractSize
    n::AbstractSize
    F::judiModeling
    rInterpolation::Projection{D}
    Init::jAdjoint{<:judiInitialStateProjection{D}}
end

"""
    judiPhoto(F::judiPropagator, geometry::Geometry;)

Constructs a photoacoustic linear operator solving the wave equation associated with F.model. The parametrizations
currently supported through JUDI are isotropic acoustic (with or without density), acoustic anisotropic (TTI/VTI) and 
visco-acoustic.

Arguments
============
`F`: The base JUDI propagator (judiModeling)
`geometry`: the receiver interpolation (judiProjection) for data measurment
"""
function judiPhoto(F::judiPropagator{D, O}, geometry::Geometry; nsrc=1) where {D, O}
    initState = adjoint(judiInitialStateProjection{D}(space_src(F.model.n, nsrc), time_space(F.model.n)))
    return judiPhoto{D, :forward}(rec_space(geometry), space_src(F.model.n, nsrc), F, judiProjection(geometry), initState)
end

judiPhoto(model::JUDI.AbstractModel, geometry::Geometry; options=Options(), nsrc=1) = judiPhoto(judiModeling(model; options=options), geometry; nsrc=nsrc)
*(F::judiDataModeling{D, O}, I::jAdjoint{<:judiInitialStateProjection{D}}) where {D, O} = judiPhoto{D, :forward}(F.m, space(F.model.n), F.F, F.rInterpolation, I)

adjoint(J::judiPhoto{D, O}) where {D, O} = judiPhoto{D, adjoint(O)}(J.n, J.m, J.F, J.rInterpolation, J.Init)
getindex(J::judiPhoto{D, O}, i) where {D, O} = judiPhoto{D, O}(J.m[i], J.n[i], J.F, J.rInterpolation[i], J.Init)

make_input(J::judiPhoto{D, O}, q) where {D<:Number, O} = Geometry(J.rInterpolation.geometry), make_input(q, J.model, J.options)
make_input(J::judiPhoto{D, O}, ::Nothing) where {D<:Number, O} = Geometry(J.rInterpolation.geometry), nothing

*(J::judiPhoto{T, :forward}, q::Array{T, 3}) where {T} = J*vec(q)
*(J::judiPhoto{T, :forward}, q::Array{T, 4}) where {T} = J*vec(q)

process_input_data(::judiPhoto{D, :forward}, q::judiInitialState{D}) where {D<:Number} = q

############################################################

function _forward_prop(J::judiPhoto{T, O}, q::AbstractArray{T}, op::Py; dm=nothing) where {T, O}
    # Get necessary inputs
    recGeometry, init_dist = make_input(J, q)

    # Compute illumination ?
    opname = isnothing(dm) ? (:forward) : (:born)
    illum = compute_illum(J.F.model, opname)

    # Check if need to skip compute
    if (~isnothing(dm) && norm(dm) == 0) || (norm(init_dist) == 0)
        dsim = zeros(Float32, recGeometry.nt[1], length(recGeometry.xloc[1]))
        return judiVector{Float32, Matrix{Float32}}(1, recGeometry, [dsim])
    end

    # Set up Python model structure
    modelPy = devito_model(J.F.model, J.F.options, dm)
    dtComp  = pyconvert(Float32, modelPy.critical_dt)
    nt = length(0:dtComp:recGeometry.t[1])

    # Pad initial state to the computational (PML-padded) grid
    init_dist = pad_array(init_dist, modelPy.padsizes; mode=:zeros)

    # Set up coordinates
    rec_coords = setup_grid(recGeometry, J.F.model.n)

    # Devito interface — wrapcall_data returns a tuple when illum is requested, otherwise a single array
    argout = wrapcall_data(op, modelPy, rec_coords, init_dist, nt,
                           ic=J.F.options.IC, illum=illum)
    dsim, I = illum ? (argout[1], argout[2]) : (argout, nothing)

    dsim = judiVector{Float32, Matrix{Float32}}(1, recGeometry, [time_resample(dsim, dtComp, recGeometry)])
    !illum && (return dsim)

    I = remove_padding(I, pyconvert(Tuple, modelPy.padsizes))
    return dsim, PhysicalParameter(I, pyconvert(Tuple, modelPy.spacing), pyconvert(Tuple, modelPy.origin))
end


function _reverse_propagate(J::judiPhoto{T, O}, q::AbstractArray{T}, op::Py; init=nothing) where {T, O}
    options = J.options
    opname = isnothing(init) ? (:adjoint) : (:adjoint_born)

    # Get input data from source and operator
    recGeometry, init_dist = make_input(J, init)
    srcData = make_input(q)

    # Set up Python model structure
    modelPy = devito_model(J.F.model, J.F.options)
    dtComp  = pyconvert(Float32, modelPy.critical_dt)

    # Compute illumination ?
    illum = compute_illum(J.F.model, opname)

    # Set up coordinates
    rec_coords = setup_grid(recGeometry, J.F.model.n)

    # Extrapolate input data to computational grid
    qIn = time_resample(srcData, recGeometry, dtComp)

    # Pad initial-state to computational (PML-padded) grid when needed (adjoint_born path)
    if !isnothing(init)
        init_dist = pad_array(init_dist, modelPy.padsizes; mode=:zeros)
    end

    args = isnothing(init) ? (modelPy, qIn, rec_coords) : (modelPy, qIn, rec_coords, init_dist)

    # Gradient options
    length(options.frequencies) == 0 ? freqs = nothing : freqs = options.frequencies
    argout = wrapcall_data(op, args..., freq_list=freqs,
                           checkpointing=options.optimal_checkpointing, ic=options.IC, illum=illum,
                           dft_sub=options.dft_subsampling_factor[1], t_sub=options.subsampling_factor)

    # When illum is false, wrapcall_data returns a single array — wrap so downstream indexing works
    argout = argout isa Tuple ? argout : (argout,)

    # Actual adjoint
    g = remove_padding(argout[1], pyconvert(Tuple, modelPy.padsizes); true_adjoint=(J.options.sum_padding && ~isnothing(init)))
    !illum && (return g)

    # Illums
    Is = JUDI.post_process(argout[2:end], modelPy, Val(:adjoint_born), nothing, nothing, Options(;sum_padding=false))
    return g, Is...
end

# JUDI interface for single source wave operator. JUDI 4 dispatches as `propagate(F, q, illum)`;
# `illum` is honored inside `_forward_prop` / `_reverse_propagate` via `compute_illum`.
propagate(J::judiPhoto{T, :forward}, q::AbstractArray{T}, ::Bool) where {T} = _forward_prop(J, q, impl.forwardis_data)
propagate(J::judiJacobian{D, :born, FT}, q::AbstractArray{T}, ::Bool) where {T, D, FT<:judiPhoto} = _forward_prop(J.F, J.q, impl.bornis_data; dm=q)
propagate(J::judiPhoto{T, :adjoint}, q::AbstractArray{T}, ::Bool) where {T} = judiInitialState(_reverse_propagate(J, q, impl.adjointis))
propagate(J::judiJacobian{D, :adjoint_born, FT}, q::AbstractArray{T}, ::Bool) where {T, D, FT<:judiPhoto} = PhysicalParameter(_reverse_propagate(J.F, q, impl.adjointbornis; init=J.q), J.model.d, J.model.o)
