# Changelog

## v0.5.0

Breaking-dependency update so that the package builds on current Julia.

- Bump `JUDI` compat to `^4.1`. JUDI 4 switched its Devito interop from PyCall to PythonCall; PhotoAcoustic now imports `PythonCall` (`Py`, `pyimport`, `pyconvert`) instead of `PyCall` (`PyObject`, `PyNULL`).
- `src/PhotoAcoustic.jl` initialises `impl` via `PythonCall.pynew()` / `pycopy!` instead of `PyNULL()` / `copy!`.
- `src/judiPhoto.jl`:
  - `op::PyObject` → `op::Py` on `_forward_prop` / `_reverse_propagate`.
  - Drop `space_order=` kwarg from `wrapcall_data` calls (JUDI 4 reads it from the Python model).
  - Coalesce `wrapcall_function` / `wrapcall_weights` into the single `wrapcall_data` (the only Python wrapper still exported by JUDI 4).
  - Pad `init_dist` to the PML grid after `devito_model` (was a separate `pad_sizes(model, options; so=0)` call in JUDI 3 that no longer exists).
  - `propagate(F, q)` methods become 3-arg `propagate(F, q, illum::Bool)` to match the new generic JUDI dispatch.
- `src/judiInitialState.jl`: `make_input(::judiInitialState, model::Model, options)` becomes `make_input(::judiInitialState, ::JUDI.AbstractModel, ::JUDIOptions)` and no longer pads — padding is now performed inside `_forward_prop`.
- `src/transducer.jl`: `post_process(...)` signature picks up the extra `srcGeometry` slot that JUDI 4 passes; `modelPy::PyObject` → `modelPy::Py`.
- `src/implementation.py`:
  - Fetch the time-stepped wavefield from `kw[name]` (where `name` is `"u"` for forward, `"v"` for adjoint) instead of from the value JUDI 4's `forward` now returns as `uout` (which can be a DFT-mode tuple or a time-subsampled save buffer).
  - Build the IBP correction term against `as_tuple(uout)[0]` so the t_sub > 1 path doesn't try to sympify a Python list.
  - Use `np.array(init_dist)` instead of `init_dist[:]` when copying into the Devito wavefield (PythonCall surface).
- Add explicit dependencies for `DSP`, `FFTW`, `JOLI`, `PythonCall` — JUDI 4 no longer re-exports these by name and the package was already using them.
- Replace `PyPlot` with `PythonPlot` in `examples/*.jl` and `examples/notebooks/*.ipynb` to match the new PythonCall-based stack.
- Drop Julia < 1.10. CI matrix is now `{1.10, 1}` on Ubuntu, `1` on macOS.
- Bump CI actions: `actions/checkout@v4`, `julia-actions/setup-julia@v2`.

### Known issues

- `test/test_sensitivities.jl`'s Jacobian Taylor-rate test currently lands at ~1.2 instead of the expected 1.5625 (within `stol=0.1`). Forward/adjoint pairing and the option-matrix tests pass; the linearised Jacobian needs a closer look against the JUDI 4 `born`/`gradient` interface change.
