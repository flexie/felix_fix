module PhotoAcoustic


using LinearAlgebra, Reexport
@reexport using JUDI

PhotoAcoustic_path = dirname(pathof(PhotoAcoustic))

using DSP, PythonCall, FFTW, JOLI
using FourierTools

import Base: getindex, *, copy!, copyto!, similar, getproperty, display
import JUDI: judiMultiSourceVector, judiComposedPropagator, judiPropagator, judiNoopOperator
import JUDI: judiDataModeling, judiModeling, judiJacobian, jAdjoint, Projection, judiVector, Geometry
import JUDI: judiProjection, JUDIOptions, Options, PhysicalParameter, AbstractSize, AbstractModel
import JUDI: RangeOrVec, make_input, propagate, zero, process_input_data, setup_grid
import JUDI: wrapcall_data, compute_illum, devito_model, remove_padding, pad_array
import JUDI: time_resample, make_src, get_nsrc, n_samples, calculate_dt, post_process
import PythonCall: Py, pyconvert, pyimport
import JUDI: space_src, time_space, time_space_src, rec_space, space
import LinearAlgebra: adjoint

const impl = PythonCall.pynew()

function __init__()
    pyimport("sys").path.insert(0, PhotoAcoustic_path)
    PythonCall.pycopy!(impl, pyimport("implementation"))
end

# utility for data loading 

PhotoAcoustic_data = joinpath(PhotoAcoustic_path, "../data")

# Utilities
include("utils.jl")
# Sources
include("judiInitialState.jl")
# Operators
include("judiPhoto.jl")
# Transducer
include("transducer.jl")

end # module

