import logging
import time
import numpy as np

from scipy.sparse.linalg import lsqr
from pylops import LinearOperator
from pylops.basicoperators import Diagonal, Identity
from pylops.optimization.leastsquares import NormalEquationsInversion, \
    RegularizedInversion
from pylops.optimization.eigs import power_iteration
from pylops.utils.backend import get_array_module, get_module_name, to_numpy

from multilayer_scintillator import tv

def _softthreshold(x, thresh):
    r"""Soft thresholding.
    Applies soft thresholding to vector ``x`` (equal to the proximity
    operator for :math:`||\mathbf{x}||_1`) as shown in [1]_.
    .. [1] Chen, Y., Chen, K., Shi, P., Wang, Y., “Irregular seismic
       data reconstruction using a percentile-half-thresholding algorithm”,
       Journal of Geophysics and Engineering, vol. 11. 2014.
    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Vector
    thresh : :obj:`float`
        Threshold
    Returns
    -------
    x1 : :obj:`numpy.ndarray`
        Tresholded vector
    """
    if np.iscomplexobj(x):
        # https://stats.stackexchange.com/questions/357339/soft-thresholding-
        # for-the-lasso-with-complex-valued-data
        x1 = np.maximum(np.abs(x) - thresh, 0.) * np.exp(1j * np.angle(x))
    else:
        x1 = np.maximum(np.abs(x)-thresh, 0.) * np.sign(x)
    return x1

def FISTA(Op, data, niter, eps=0.1, beta=0., alpha=None, eigsiter=None, eigstol=0,
          tol=1e-10, returninfo=False, show=False, threshkind='soft',
          perc=None, callback=None, decay=None, SOp=None, restart_interval=-1, xstart=None, dims=None, C=2):

    # choose thresholding function
    if threshkind == 'soft':
        threshf = _softthreshold
        sparsef = lambda x: x 
    elif threshkind == 'tvapprox':
        threshf = lambda x, thresh: tv.tv_proximal_approx(x, dims, thresh)
        tvop = tv.tv(dims)
        sparsef = lambda x: tvop * x
    
    # identify backend to use
    ncp = get_array_module(data)

    # prepare decay (if not passed)
    if perc is None and decay is None:
        decay = ncp.ones(niter)

    if show:
        tstart = time.time()
        print('FISTA optimization (%s thresholding)\n'
              '-----------------------------------------------------------\n'
              'The Operator Op has %d rows and %d cols\n'
              'eps = %10e\ttol = %10e\tniter = %d' % (threshkind,
                                                      Op.shape[0],
                                                      Op.shape[1],
                                                      eps, tol, niter))
    # step size
    t_ = time.time()
    if alpha is None:
        if not isinstance(Op, LinearOperator):
            Op = LinearOperator(Op, explicit=False)
        # compute largest eigenvalues of Op^H * Op
        Op1 = LinearOperator(Op.H * Op, explicit=False)
        if get_module_name(ncp) == 'numpy':
            maxeig = np.abs(Op1.eigs(neigs=1, symmetric=True, niter=eigsiter,
                                     **dict(tol=eigstol, which='LM')))[0]
        else:
            maxeig = np.abs(power_iteration(Op1, niter=eigsiter,
                                            tol=eigstol, dtype=Op1.dtype,
                                            backend='cupy')[0])
        L = maxeig + 2 * beta
        alpha0 = 1 / L
        alpha = alpha0 / C
    time_eig = time.time() - t_

    # define threshold
    thresh = eps * alpha * 0.5

    if show:
        if perc is None:
            print('alpha = %10e\tthresh = %10e' % (alpha, thresh))
        else:
            print('alpha = %10e\tperc = %.1f' % (alpha, perc))
        print('-----------------------------------------------------------\n')
        head1 = '   Itn       x[0]        r2norm     r12norm     xupdate     scale      t         restart   suppsz (pred)'
        print(head1)

    # initialize model and cost function
    xinv = ncp.zeros(int(Op.shape[1]), dtype=Op.dtype)
    #xinv = ncp.random.uniform(size=(int(Op.shape[1]),))

    if xstart is None:
        pass
        #xinv = ncp.random.randn(int(Op.shape[1])) # start with random x
    else:
        if show:
            print("using xstart")
        xinv = xstart
    
    zinv = xinv.copy()
    t = 1
    if returninfo:
        cost = np.zeros(niter + 1)

    time_matvec = 0
    time_matTvec = 0
    time_ptwise = 0
    time_prox = 0

    # iterate
    for iiter in range(niter):
        #alpha = alpha0/(iiter+1)
        xinvold = xinv.copy()

        # compute residual
        t_ = time.time()
        Opz = Op._matvec(zinv.flatten())
        time_matvec += time.time() - t_

        t_ = time.time()
        resz = data - Opz
        time_ptwise += time.time() - t_

        # compute gradient
        t_ = time.time()
        Oprz = Op._rmatvec(resz)
        time_matTvec += time.time() - t_

        t_ = time.time()
        Oprz -= beta * zinv
        grad = alpha * Oprz
        
        time_ptwise += time.time() - t_
        
        t_ = time.time()

        # update inverted model
        xinv_unthesh = zinv + grad
        if SOp is not None:
            xinv_unthesh = SOp.rmatvec(xinv_unthesh)
        if perc is None:
            xinv = threshf(xinv_unthesh, decay[iiter] * thresh)
        else:
            xinv = threshf(xinv_unthesh, 100 - perc)
        if SOp is not None:
            xinv = SOp.matvec(xinv)
            
        time_prox += time.time() - t_
        
        t_ = time.time()

        # model update
        if iiter > 0:
            xupdateold = xupdate
        xupdate = np.linalg.norm(xinv - xinvold)

        restart = np.sum((zinv - xinv) * (xinv - xinvold)) 

        #update auxiliary coefficients
        if restart > 0 or (restart_interval != -1 and iiter > 0 and iiter % restart_interval == 0):
            t = 1
            if show:
                pass
                print("restarting momentum!")
            pass 
            t = 1

        told = t
        t = (1. + np.sqrt(1. + 4. * t ** 2)) / 2.
        zinv = xinv + ((told - 1.) / t) * (xinv - xinvold)


        if returninfo or show:
            costdata_approx = 0.5 * np.linalg.norm(data - Opz) ** 2 # should use x
            costreg = eps * np.linalg.norm(sparsef(xinv), ord=1)
        if returninfo:
            cost[iiter] = costdata_approx + costreg

        time_ptwise += time.time() - t_

        # run callback
        if callback is not None:
            callback(xinv)

        if show:
            if iiter < 10 or iiter % 100 == 0:
                msg = '%6g  %12.5e  %10.3e   %9.3e  %10.3e %10.3e %10.3e %10.3e %6g (%6g)' % \
                      (iiter + 1, to_numpy(xinv[:2])[0], costdata_approx,
                       costdata_approx + costreg, xupdate, ((told-1.)/t), t,
                       restart, np.nonzero(sparsef(xinv))[0].size, np.nonzero(sparsef(xinv) > 0.00001)[0].size)
                print(msg)

        # check tolerance
        if xupdate < tol:# and iiter > 10:
            niter = iiter
            break

    # get values pre-threshold  at locations where xinv is different from zero
    # xinv = np.where(xinv != 0, xinv_unthesh, xinv)

    if show:
        print('\nIterations = %d        Total time (s) = %.2f'
              % (niter, time.time() - tstart))
        print('---------------------------------------------------------\n')

    if returninfo:
        return xinv, niter, cost[:niter]
    else:
        return xinv, [niter, xupdate, [time_matvec, time_matTvec, time_prox, time_ptwise, time_eig]]
