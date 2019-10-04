from numpy.random import multivariate_normal
from statsmodels.sandbox.distributions.extras import mvnormcdf
from scipy.stats import norm
from scipy.stats import truncnorm
from scipy.stats import binom as p_binom
from scipy.special import binom
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm, tqdm_notebook
from copy import deepcopy
from itertools import product
from . import poly_blend as pb

def one_gen_fam_matrix(nn):
    result = np.ones((nn+2, nn+2))
    for ii in range(nn):
        for jj in range(ii+1, nn):
            result[ii, jj] = 0.5
            result[jj, ii] = 0.5
    for ii in range(nn):
        result[ii, nn] = 0.5
        result[nn, ii] = 0.5
        result[ii, nn+1] = 0.5
        result[nn+1, ii] = 0.5
    return result

def gen_one_gen_probs(nn, h2, TT, FF, log_out=True, abs_err=1e-3, maxpts_mult=5e6):
    result = [np.zeros(nn+1), np.zeros(nn+1), np.zeros(nn+1)]
    for ii in range(3):
        for jj in range(nn+1):
            YY = np.array([1] + [1]*jj + [0]*(nn-jj) + [1]*ii + [0]*(2-ii))
            abseps = abs_err / (binom(2, ii)*binom(nn, jj)) / (3*(nn+1))
            result[ii][jj] = pb.threshold_prob(YY=YY,
                                               index=[0],
                                               GG=np.zeros(nn+3),
                                               beta=0,
                                               mu=0,
                                               Vp=1,
                                               h2=h2,
                                               FF=FF,
                                               TT=TT,
                                               log_out=log_out,
                                               genz=True,
                                               maxpts_mult=maxpts_mult,
                                               abseps=abseps)
    return result

def one_gen_ex(nn, h2, TT, abs_err=1e-4, maxpts_mult=5e6):
    if nn == 0:
        return None
    FF = one_gen_fam_matrix(nn+1)
    probs = gen_one_gen_probs(nn, h2, TT, FF, abs_err=abs_err, maxpts_mult=maxpts_mult)
    null_heights = []
    null_lambdas = []
    null_aff_off = []
    for ii in range(3):
        for jj in range(nn+1):
            null_heights.append(np.exp(probs[ii][jj])*binom(2,ii)*binom(nn,jj))
            null_aff_off.append(jj)
            if ii == 2:
                null_lambdas.append(0)
            else:
                null_lambdas.append(np.exp(-probs[ii][jj]))
    mend_heights = []
    mend_lambdas = []
    mend_aff_off = []
    for jj in range(nn+1):
        mend_heights.append(p_binom.pmf(jj, nn, 0.5))
        mend_lambdas.append(np.exp(-probs[1][jj]))
        mend_aff_off.append(jj)
    return (np.array(null_heights)/np.sum(null_heights), np.array(null_lambdas),
            np.array(mend_heights)/np.sum(mend_heights), np.array(mend_lambdas),
            np.array(null_aff_off), np.array(mend_aff_off))

def plot_one_gen_ex(nh, nl, mh, ml, ax):
    zero_nl = np.array([nl[ii] == 0 for ii in range(len(nl))])
    zero_prob = np.sum(nh[zero_nl])

    ax.set(xlim=[-0.5, np.max(nl)*2], ylim=[np.min(nh)/2, np.max(nh)*2],
           yscale="log", ylabel=r"$p_{null}(\lambda)$", xlabel=r"$\lambda$")
    ax.set_xscale("symlog", linthreshx=1)
    ax.plot([0,0], [np.min(nh)/2, zero_prob], "--", c="blue", alpha=0.5)

    for ii in range(len(nl[~zero_nl])):
        ax.plot(np.ones(2)*nl[~zero_nl][ii],
                [np.min(nh)/2, nh[~zero_nl][ii]], "--", c="blue", alpha=0.5)
    for ii in range(len(ml)):
        ax.plot(np.ones(2)*ml[ii],
                [np.min(nh)/2, mh[ii]], "--", c="red", alpha=0.5)

    ax.scatter(0, zero_prob, c="blue")
    ax.scatter(nl[~zero_nl], nh[~zero_nl], c="blue")
    ax.scatter(ml, mh, c="red")
    return zero_nl, zero_prob

def cumm_dist(nh, nl):
    return lambda ll: np.sum(nh[nl <= ll])

def power(nh, nl, mh, ml, alpha):
    F_dist = cumm_dist(nh, nl)
    result = 0
    foo = np.ones(ml.size)
    for ii, ll in enumerate(ml):
        result += mh[ii]*(F_dist(ll) > (1-alpha))
        foo[ii] = F_dist(ll) > (1-alpha)
    return result, foo
