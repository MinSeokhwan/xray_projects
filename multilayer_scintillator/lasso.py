from time import time_ns
import autograd.numpy as np
import pylops
from autograd.extend import primitive, defvjp
from autograd import grad

from multilayer_scintillator import pylops_solvers
from multilayer_scintillator import tv
solver = None
supp_thresh = 0.001

@primitive
def solve(params, make_pyLop, make_matvec, make_matTvec, make_noise, u, make_alpha, make_beta, 
        max_iters, tol, verbosity=0, restart=-1, help=False, reg='l1', dims=None):
    
    import time
    t0 = time.time()

    op = make_pyLop(params)
    n,p = op.shape
    noise = make_noise(params)
    if type(noise) == np.numpy_boxes.ArrayBox:
        noise = noise._value#[float(n) for n in noise] # autograd arraybox -> np array
    
    if not help:
        # print("avg:", np.average(np.abs(op.todense())))
        # X = np.random.randn(n, p) * np.average(np.abs(op.todense()))
        # op = pylops.MatrixMult(X)
        y = op.matvec(u)
        y += noise
        alpha = make_alpha(params)
        beta = make_beta(params)

        if reg == 'l1':
            threshkind = 'soft'
        elif reg == 'tv':
            threshkind = 'tvapprox'

        uest, [iters, xupdate, [time_matvec, time_matTvec, time_prox, time_ptwise, t_eig]] = \
            pylops_solvers.FISTA(op, y, max_iters, beta=beta*n, eps=alpha*n*2, tol=tol, 
            show=(verbosity > 2), callback=None, restart_interval=-1, 
            threshkind=threshkind, dims=dims, C=(100 if reg == 'tv' else 2)) # disable warm start
        
        fake_uest = fake_solve(uest, params, make_pyLop, make_matvec, make_matTvec, make_noise, u, make_alpha, make_beta, 
                max_iters, tol, verbosity=0, restart=-1, help=False, reg=reg, dims=dims)
        
        # from sklearn import linear_model as lm
        # # global solver
        # # if solver is None:
        # solver = lm.Lasso(alpha=alpha, fit_intercept=False, max_iter=max_iters, tol=tol, selection='random')
        # solver.fit(op.todense(), y)
        # uest = solver.coef_
        # iters = solver.n_iter_
    else:
        uest = fake_solve(u, params, make_pyLop, make_matvec, make_matTvec, make_noise, u, make_alpha, make_beta, 
                max_iters, tol, verbosity=0, restart=-1, help=False, reg=reg, dims=dims)
    #print("Compare uest and fake_uest", np.all(np.isclose(uest, fake_uest)), support(uest) == support(fake_uest), np.linalg.norm(uest-fake_uest)/np.linalg.norm(uest))

    t1 = time.time()
    if verbosity > 0:
        toprint = ""
        if not help:
            toprint += "Solve took {:.3f} ({:.3f} / {:.3f} / {:.3f} / {:.3f} / {:.3f}) seconds, {} iterations, {} final xupdate.\n"\
                   .format(t1-t0, time_matvec, time_matTvec, time_prox, time_ptwise, t_eig, iters, xupdate)
            toprint += "uest v.s. fake_uest: {}".format(np.linalg.norm(uest - fake_uest) / np.linalg.norm(uest))
        if reg == 'l1':
            regop = pylops.Identity(p)
        elif reg == 'tv':
            regop = tv.tv(dims)
        regu = regop * u
        reguest = regop * uest
        suppu = np.abs(regu) > supp_thresh
        suppuest = np.abs(reguest) > supp_thresh
        suppboth = suppu & suppuest
        suppuonly = suppu & np.logical_not(suppuest)
        suppuestonly = suppuest & np.logical_not(suppu)
        mag = lambda v: np.sum(v**2)
        toprint += "\nEstimated support: uest = {} ({:.2g})".format(np.nonzero(suppuest)[0].size, mag(reguest[suppuest])/mag(reguest))
        toprint += "\nSupport sizes: u and uest = {} ({:.2g}, {:.2g}, {:.2g}), u - uest = {} ({:.2g}), uest - u = {} ({:.2g})"\
                    .format(np.nonzero(suppboth)[0].size, mag(regu[suppboth])/mag(regu), mag(reguest[suppboth])/mag(regu), mag(regu[suppboth] - reguest[suppboth])/mag(regu),
                    np.nonzero(suppuonly)[0].size, mag(regu[suppuonly])/mag(regu),
                    np.nonzero(suppuestonly)[0].size, mag(reguest[suppuestonly])/mag(regu))
        # print("u:", u)
        # print("uest:", uest)
#        if verbosity > 1:
#            toprint += ("\nSupport of u is {}\n"
#                "u on support(u) is {}\n"
#                "Support of uest is {}\n"
#                "uest on support(uest) is {}").format(support(u), u[support(u)],
#                                                     support(uest), uest[support(uest)])
        print(toprint, flush=True)
        print('', flush=True)
    return uest

def fake_solve(uest, params, make_pyLop, make_matvec, make_matTvec, make_noise, u, make_alpha, make_beta, 
        max_iters, tol, verbosity=0, restart=-1, help=False, reg='l1', dims=None):
    op = make_pyLop(params)
    n,p = op.shape

    if reg == 'l1':
        regop = pylops.Identity(p)
    elif reg == 'tv':
        regop = tv.tv(dims)

    supp = np.abs(regop * uest) > supp_thresh
    sign = np.sign(regop * uest)

    if reg == 'l1':
        P = pylops.Restriction(p, np.nonzero(supp)[0]) # projects onto support of uest
    elif reg == 'tv':
        P = tv.tv_proj(dims, supp)     

    # print("number of regions:", P.shape[0])
    # print("proj:", P.T * P * uest)
    # print("dims:", dims)
    # print("region vals:", P.region_vals(uest))

    alpha = make_alpha(params)
    beta = make_beta(params)
    noise = make_noise(params) 

    leftOp = P * op.T * op * P.T + n * beta * pylops.Identity(P.shape[0]) # left operator for KKT equation restricted to support
    rhs = P * op.T * (op * u + noise) - alpha * n * P * regop.T * sign

    import time
    t0 = time.time()
    fake_uest_proj, iters, _ = pylops.optimization.basic.cg(leftOp, rhs, x0=np.zeros(P.shape[0]), tol=tol, niter=max_iters, show=(verbosity>2))
    t1 = time.time()
    if verbosity > 0:
        print("Fake solve took {} seconds, {} iterations".format(t1 - t0, iters), flush=True)
    # print("leftOp shape: {}, rhs shape: {}".format(leftOp.shape, rhs.shape))
    # uest_proj = P.T * P * uest
    # print("projection error", np.linalg.norm(uest_proj - uest) / np.linalg.norm(uest))
    # fake_rvals = P.region_vals(P.T * fake_uest_proj)
    # rvals = P.region_vals(uest_proj)
    # print("system error:", fake_rvals, rvals, np.linalg.norm(fake_rvals - rvals))
    fake_uest = P.T * fake_uest_proj
    return fake_uest

def solve_grad(uest, params, make_pyLop, make_matvec, make_matTvec, make_noise, u, make_alpha, make_beta, 
                max_iters, tol, verbosity=0, restart=-1, test=False, help=False, 
                reg='l1', dims=None):
    op = make_pyLop(params)
    n,p = op.shape

    if reg == 'l1':
        regop = pylops.Identity(p)
    elif reg == 'tv':
        regop = tv.tv(dims)

    supp = np.abs(regop * uest) > supp_thresh
    sign = np.sign(regop * uest)

    if reg == 'l1':
        P = pylops.Restriction(p, np.nonzero(supp)[0]) # projects onto support of uest
    elif reg == 'tv':
        P = tv.tv_proj(dims, supp)

    left_op = P * op.T * op * P.T + n * make_beta(params) * pylops.Identity(P.shape[0])
    # the derivative of rvec w.r.t. params is db/dp - dA/dp * uest, which we need to dot with the adjoint var
    def restriction_matvec(x):
        x = np.reshape(x, p)
        y = x[np.nonzero(supp)[0]]
        y = y.ravel()
        return y
    def make_rvec(params):
        matvec = make_matvec(params)
        matTvec = make_matTvec(params)
        alpha = make_alpha(params)
        beta = make_beta(params)
        # b = matTvec(matvec(u) + make_noise(params))[supp] - alpha * n * sign[supp]
        # Auest = matTvec(matvec(uest))[supp] + n * beta * uest[supp]
        if reg == 'l1':
            b1 = restriction_matvec(matTvec(matvec(u) + make_noise(params)))
            b2 = restriction_matvec(regop.T._matvec(sign))
            b = b1 - alpha * n* b2
            A1 = restriction_matvec(matTvec(matvec(uest)))
            A2 = restriction_matvec(uest)
            Auest = A1 + n * beta * A2
        elif reg == 'tv':
            b = P._matvec(matTvec(matvec(u) + make_noise(params))) - alpha * n * P._matvec(regop.T._matvec(sign))
            Auest = P._matvec(matTvec(matvec(uest))) + n * beta * P._matvec(uest)
        return b - Auest
    def vecjacprod(vec):
        vec_supp = P * vec
        import time
        t0 = time.time()
        #import scipy
        #print(left_op.todense())
       # lmbda, istop, iters = scipy.sparse.linalg.lsqr(left_op, vec_supp)[0:3]
        #lmbda, iters = scipy.sparse.linalg.cg(left_op, vec_supp)
        #print("info:", info)
        lmbda, iters, _ = pylops.optimization.basic.cg(left_op, vec_supp, x0=np.zeros(P.shape[0]), tol=tol, niter=max_iters, show=(verbosity>2)) # solve for adjoint var 
        t1 = time.time()
        if verbosity > 0:
            print("Adjoint problem took {} seconds, {} iterations".format(t1 - t0, iters), flush=True)
        # leftvec = P.T * lmbda
        # gdat = find_vinneru_grad(params, make_matvec, make_matTvec, leftvec, u - uest)
        func = lambda params: np.dot(lmbda, make_rvec(params))
        gdat = grad(func)(params)
        return gdat
    return vecjacprod
        
defvjp(solve, solve_grad, argnums=[0])