from devito import Operator, Function, Eq, Inc, norm
from devito.tools import as_tuple

import numpy as np

import interface
from propagators import forward, adjoint, gradient, born


def _wf_name(kwargs):
    return "v" if kwargs.get('fw', True) is False else "u"


def forwardis(model, rcv_coords, init_dist, nt, **kwargs):
    """
    Forward photoacoustic propagator. Propagates and intitial state init_dist
    consisting of the first two time steps (u(t=0)=f and u.dt(t=0)=0)
    """
    return_op = kwargs.get('return_op', False)
    kwargs['return_op'] = True
    op, uout, rcv, kw = forward(model, None, rcv_coords, np.zeros((nt,)), **kwargs)

    # JUDI 4's forward returns `uout`, which may be the raw wavefield, a time-subsampled
    # save buffer, or a tuple of DFT modes. The actual time-stepped wavefield is in `kw`
    # under its name ("u" forward, "v" adjoint) — use that to set the initial conditions.
    name = _wf_name(kwargs)
    u = kw[name]
    u.data[0, :] = np.array(init_dist)
    u.data[1, :] = np.array(init_dist)

    if return_op:
        return op, uout, rcv, kw

    summary = op(**kw)

    # Illumination
    I = kw.get('I' + name, None)

    # Reset initial condition in case we have a buffered `u` that needs to be reused
    us = kw['us_' + name] if kwargs.get('t_sub', 0) > 1 else u
    us.data[0] = np.array(init_dist)
    return rcv, uout, I, summary


def forwardis_data(*args, **kwargs):
    rcv, u, I, summary = forwardis(*args, **kwargs)
    return rcv.data, getattr(I, "data", None)


def bornis(model, rcv_coords, init_dist, nt, **kwargs):
    """
    Linearized forward photoacoustic propagator. Propagates and intitial state init_dist
    conssisting of the first two time steps (u(t=0)=f and u.dt(t=0)=0)
    """
    return_op = kwargs.get('return_op', False)
    kwargs['return_op'] = True
    op, uout, rcv, kw = born(model, None, rcv_coords, np.zeros((nt,)), **kwargs)

    name = _wf_name(kwargs)
    u = kw[name]
    u.data[0, :] = np.array(init_dist)
    u.data[1, :] = np.array(init_dist)

    if return_op:
        return op, uout, rcv, kw

    op(**kw)

    # Illumination
    I = kw.get('I' + name, None)

    return rcv, uout, I


def bornis_data(*args, **kwargs):
    rcv, u, I = bornis(*args, **kwargs)
    return rcv.data, getattr(I, "data", None)


def adjointis(model, y, rcv_coords, **kwargs):
    """
    Adjoint photoacoustic propagator.
    """
    kwargs.pop('checkpointing', None)
    kwargs.pop('t_sub', None)
    kwargs.pop('ic', None)

    # Run adjoint simulation with JUDI. Force return_op so we get (op, uout, rout, kw)
    # — JUDI 4's forward otherwise returns (rout, uout, I, summary) at the same arity,
    # which would bind `summary` into `kw` and break the kw['v'] lookup below.
    kwargs['fw'] = False
    kwargs['return_op'] = True
    op, _uout, _rcv, kw = forward(model, rcv_coords, None, -y, **kwargs)
    op(**kw)

    # The time-stepped wavefield is stored in kw under its name ("v" for fw=False).
    v = kw['v']

    # Extract time derivative at 0.
    init = Function(name="ini", grid=model.grid, space_order=0)

    # Correct for default scaling in injection
    mrm = model.m * model.irho
    op0 = Operator(Eq(init, mrm * v.dt))
    op0(dt=model.critical_dt, time_m=0, time_M=0)

    I = kw.get('Iv', None)
    return init.data, getattr(I, "data", None)


def adjointbornis(model, y, rcv_coords, init_dist, checkpointing=None, freq_list=None,
                  t_sub=1, **kwargs):
    """
    Adjoint photoacoustic propagator.
    """
    nt = y.shape[0]
    born_fwd = kwargs.get('born_fwd', False)
    rec, uout, Iu, _ = op_fwd_JIS[born_fwd](model, rcv_coords, init_dist, nt,
                                             save=freq_list is None, freq_list=freq_list,
                                             t_sub=t_sub, **kwargs)

    # Get operator
    kwargs['return_op'] = True
    op, g, kwg = gradient(model, y, rcv_coords, uout, save=freq_list is None, freq=freq_list,
                          **kwargs)
    op(**kwg)
    Iv = kwg.get('Iv', None)
    # Need the intergation by part correction since we compute the gradient on
    # u * v.dt (see Documentation). `uout` from JUDI 4's `forward` may be a list
    # (time-subsampled save buffer) or a single TimeFunction.
    if freq_list is None:
        w = model.irho if kwargs.get('ic', "as") == "as" else model.irho * model.m
        u_for_ibp = as_tuple(uout)[0]
        op0 = Operator(Eq(g, g -  w * kwg['v'].dt * u_for_ibp))
        op0(dt=model.critical_dt, time_m=0, time_M=0)

    return g.data, getattr(Iu, "data", None), getattr(Iv, "data", None)

op_fwd_JIS = {False: forwardis, True: bornis}
