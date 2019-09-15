from numpy.random import multivariate_normal
from statsmodels.sandbox.distributions.extras import mvnormcdf
from scipy.stats import norm
from scipy.stats import truncnorm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm, tqdm_notebook
from copy import deepcopy
from itertools import product

def subset_matrix(YY, FF, cond):
    """Return FF with rows and columns removed where YY doesnt match cond."""
    if np.sum(YY == cond) == len(YY):
        return FF
    result = np.zeros((np.sum(YY==cond), np.sum(YY==cond)))
    ii = 0
    for i_p, Y_i in enumerate(YY):
        if Y_i == cond:
            jj = 0
            for j_p, Y_j in enumerate(YY):
                if Y_j == cond:
                    result[ii, jj] = FF[i_p, j_p]
                    jj += 1
            ii += 1
    return result

def threshold_prob(YY, index, GG, beta, mu, Vp, h2, FF, TT):
    """Calculate the probability of binary phenotypes in a pedigree, conditional on index individuals.

    Keyword arguments:
    YY    -- Binary phenotype array, numpy array with 0 for below thresh, 1 for above
    index -- List of index patient indexes
    GG    -- Genotypes, numpy array of 0,1,2 giving the number of alleles
    beta  -- Effect size of the Mendelian locus
    mu    -- Mean population trait value
    Vp    -- Population trait variance
    h2    -- Trait heritability
    FF    -- Kinship matrix
    TT    -- Trait threshold for exhibiting the phenotype
    """
    n_above = np.sum(YY)
    n_below = np.size(YY) - n_above
    below_FF = subset_matrix(YY, FF, 0)
    above_FF = subset_matrix(YY, FF, 1)
    below_GG = np.array([GG_i for ii, GG_i in enumerate(GG) if YY[ii] == 0])
    above_GG = np.array([GG_i for ii, GG_i in enumerate(GG) if YY[ii] == 1])

    YY_index = np.array([1*(ii in index) for ii, _ in enumerate(YY)])

    index_FF = subset_matrix(YY_index, FF, 1)

    GG_index = [GG_i for ii, GG_i in enumerate(GG) if ii in index]

    lower_lims = [xx if xx==xx else TT
                  for xx in -np.inf*(1-np.array(YY))]
    upper_lims = [xx if xx==xx else TT
                  for xx in np.inf*np.array(YY)]
    means = np.ones(np.size(YY))*mu + np.array(GG)*beta
    cov = Vp*h2*FF + Vp*(1-h2)*np.identity(np.size(YY))

    P1 = mvnormcdf(lower=lower_lims,
                   upper=upper_lims,
                   mu=means,
                   cov=cov, maxpts=np.size(YY)*20000)

    lower_lims_index = [xx for ii, xx in enumerate(lower_lims)
                        if ii in index]
    upper_lims_index = [xx for ii, xx in enumerate(upper_lims)
                        if ii in index]
    YY_index_only = np.array([xx for ii, xx in enumerate(YY)
                              if ii in index])
    means_index = (np.ones(np.size(YY_index_only))*mu +
                   np.array(GG_index)*beta)
    cov_index = (Vp*h2*index_FF +
                        Vp*(1-h2)*np.identity(np.size(YY_index_only)))

    if np.size(lower_lims_index) > 1:
        P2 = mvnormcdf(lower=lower_lims_index,
                       upper=upper_lims_index,
                       mu=np.array(means_index),
                       cov=cov_index)
    else:
        if lower_lims_index[0] == -np.inf:
            P2 = norm.cdf(upper_lims_index[0],
                          loc=means_index[0],
                          scale=np.sqrt(cov_index[0,0]))
        else:
            P2 = 1 - norm.cdf(lower_lims_index[0],
                              loc=means_index[0],
                              scale=np.sqrt(cov_index[0,0]))

    return np.log(P1) - np.log(P2) #(P1/P2)

def conditional_covariance(CC, index, not_index):
    """Calculate the covariance matrix of non-index individuals conditional on index."""
    new_order = np.concatenate((not_index, index))
    rCC = CC[:,new_order][new_order,:]
    n_ind = len(index)
    n_not_ind = len(not_index)
    nn = n_ind + n_not_ind
    CC11 = rCC[:,0:n_not_ind][0:n_not_ind,:]
    CC12 = rCC[0:n_not_ind,:][:,n_not_ind:nn]
    CC21 = rCC[n_not_ind:nn,:][:,0:n_not_ind]
    CC22 = rCC[n_not_ind:nn,:][:,n_not_ind:nn]

    return (CC11 + np.matmul(np.matmul(CC12, np.linalg.inv(CC22)), CC21),
            CC11, CC12, CC21, CC22)

def simulate_polygenic(FF, index, Y_index, Vp, h2, TT, mu, GG, beta, size=1):
    """Simulate trait values conditional on binary phenotype status in index patients.

    Keyword arguments:
    FF      -- Kinship matrix
    index   -- List of index patient indexes
    Y_index -- Binary phenotype values in index patients
    Vp      -- Population trait variance
    h2      -- Trait heritability
    TT      -- Trait threshold for exhibiting the phenotype
    mu      -- Trait mean in population
    GG      -- Genotypes, numpy array of 0,1,2 giving the number of alleles
    beta    -- Effect size of the Mendelian locus
    size    -- Number of simulations to run (default 1)
    """
    cov = Vp*h2*FF + Vp*(1-h2)*np.identity(FF.shape[0])
    not_index = [ii for ii in range(FF.shape[0])
                 if ii not in index]
    cd_cov, CC11, CC12, CC21, CC22 = conditional_covariance(cov,
                                                            index,
                                                            not_index)
    lower_index = [xx if xx==xx else TT for xx in -np.inf*(1-Y_index)]
    upper_index = [xx if xx==xx else TT for xx in np.inf*Y_index]
    rv_size = [size, len(index)]
    index_vals = truncnorm.rvs(a=lower_index,
                               b=upper_index,
                               loc=GG[index]*beta + np.ones(len(index))*mu,
                               scale=np.ones(len(index))*np.sqrt(Vp),
                               size=rv_size)
    means = mu + np.matmul(np.matmul(CC12, np.linalg.inv(CC22)),
                           index_vals[:,:,np.newaxis]-mu)
    result = np.zeros((size, FF.shape[0]-len(index)))
    # Create a separate array that stores the index trait values as well
    result_full = np.zeros((size, FF.shape[0]))
    for ii, ind in enumerate(index):
        result_full[:,ind] = index_vals[:,ii]
    for ii in range(size):
        result[ii,:] = multivariate_normal(mean=means[ii][:,0], cov=cd_cov)
        for jj, ind in enumerate(not_index):
            result_full[ii,ind] = result[ii,jj]
    return result, result_full

def calc_all_lambda(fam, YY_set, index, GG, beta, mu, Vp, h2, FF, TT, missing=None):
    """Calculate a set of phenotype probabilities conditional on index phenotypes.
    WARNING: if there is missing data you still need to pass the FULL family
    and set of phenotypes (YY_set)
    """
    if missing is not None:
        if np.sum(missing) == (YY_set.shape[1]-len(index)):
            # If every non-focal individual is missing, the conditional probability is
            # not undefined, so None is the correct reponse.
            print("All non-focal individuals missing, returning None")
            return None, None

    all_probs = np.zeros(YY_set.shape[0])
    prob_set = {}
    for ii in range(YY_set.shape[0]):
        if missing is None:
            key = repr(YY_set[ii,:])
        else:
            key = repr(YY_set[ii,~missing])
        if key not in prob_set.keys():
            dom_compatible = possible_dominant_missing(fam, YY_set[ii,:], missing=missing)
            if not dom_compatible:
                prob_set[key] = 0
            else:
                if missing is None:
                    prob_set[key] = threshold_prob(YY=YY_set[ii,:],
                                                   index=index,
                                                   GG=GG,
                                                   beta=beta,
                                                   mu=mu,
                                                   Vp=Vp,
                                                   h2=h2,
                                                   FF=FF,
                                                   TT=TT)
                else:
                    prob_set[key] = threshold_prob(YY=YY_set[ii,~missing],
                                                   index=index,
                                                   GG=GG,
                                                   beta=beta,
                                                   mu=mu,
                                                   Vp=Vp,
                                                   h2=h2,
                                                   FF=FF,
                                                   TT=TT)
        all_probs[ii] = prob_set[key]
    return all_probs, prob_set

def get_proband(fam):
    result = []
    for ii, ind in enumerate(fam['inds']):
        if ind.proband == 1:
            result.append(ii)
    return result

def get_status(fam):
    result = np.zeros(len(fam['inds']))
    for ii, ind in enumerate(fam['inds']):
        if ind.affected == 2:
            result[ii] = 1
    return result

def get_ancestors(ii, inds, level):
    """Return a list of all ancestors of an individual as tuples of id and level"""
    if inds[ii].father == 0 and inds[ii].mother == 0:
        return [(ii, level)]
    if inds[ii].father == 0:
        return get_ancestors(inds[ii].mother-1, inds, level+1) + [(ii, level)]
    if inds[ii].mother == 0:
        return get_ancestors(inds[ii].father-1, inds, level+1) + [(ii, level)]
    else:
        return (get_ancestors(inds[ii].father-1, inds, level+1) +
                get_ancestors(inds[ii].mother-1, inds, level+1) + [(ii,level)])

def get_lowest_anc(ancset_1, ancset_2):
    """Return one of the lowest ancestors shared by two individuals."""
    lowest_anc = None
    for anc_1 in ancset_1:
        for anc_2 in ancset_2:
            if anc_1[0] == anc_2[0]:
                if lowest_anc is None:
                    lowest_anc = (anc_1[0], anc_1[1]+anc_2[1])
                elif lowest_anc[1] > (anc_1[1]+anc_2[1]):
                    lowest_anc = (anc_1[0], anc_1[1]+anc_2[1])
    return lowest_anc

def calc_kinship(id_1, id_2, all_ancestors):
    """Calculate the kinship between two individuals.
    WARNING: individual ids must be in order from 1 to n

    Keyword arguments:
    id_1           -- ID of first individual (id field of fam file)
    id_2           -- ID of second individual
    all_ancestors  -- List of ancestor lists for each individual in the pedigree
    """
    ancset_1 = all_ancestors[id_1-1]
    ancset_2 = all_ancestors[id_2-1]
    kinship = 0
    while get_lowest_anc(ancset_1, ancset_2) is not None:
        lowest_anc = get_lowest_anc(ancset_1, ancset_2)
        kinship += 2**(-lowest_anc[1])
        ancset_1 = [anc for anc in ancset_1 if anc[0] not in
                    [aa[0] for aa in all_ancestors[lowest_anc[0]]]]
        ancset_2 = [anc for anc in ancset_2 if anc[0] not in
                    [aa[0] for aa in all_ancestors[lowest_anc[0]]]]
    return kinship

def fam_to_cov(inds):
    """Calculate kinship (covariance) matrix given list of individuals and parents."""
    try:
        ancestor_set = [get_ancestors(ii, inds, 0) for ii in range(len(inds))]
    except IndexError:
        print('Pedigree incomplete')
        return None
    result = np.zeros((len(inds), len(inds)))
    for ii, ind in enumerate(inds):
        for jj, ind in enumerate(inds):
            if inds[ii].monozygotic == inds[jj].id:
                result[ii,jj] = 1
            else:
                result[ii,jj] = calc_kinship(ii+1, jj+1, ancestor_set)
    return result

def remove_missing(fam, treat_dec_missing=False, keep_affected_dec=True):
    """Remove individuals with missing data from family and covariance matrix.

    Keyword arguments:
    fam                -- dictionary for family
    treat_dec_missing  -- whether to treat deceased individuals as potentially missing
    keep_affected_dec  -- whether to keep deceased individuals marked as affected
    """
    missing = is_missing(fam, treat_dec_missing, keep_affected_dec)
    new_fam = {'ped': fam['ped'], 'inds': []}
    for ii, miss in enumerate(missing):
        if not miss:
            new_fam['inds'] += [fam['inds'][ii]]
    new_fam['cov'] = subset_matrix(missing, fam['cov'], False)
    return new_fam

def is_missing(fam, treat_dec_missing=False, keep_affected_dec=True):
    """Calculate list of which individual in the family are missing."""
    missing = np.array([False]*len(fam['inds']))
    for ii, ind in enumerate(fam['inds']):
        if ind.missing:
            missing[ii] = True
        elif ind.status:
            if treat_dec_missing:
                if ind.affected == 1:
                    missing[ii] = True
                elif not keep_affected_dec:
                    missing[ii] = True
    return missing

def read_pedigree(fname, sep=","):
    """Read pedigrees from csv and calculate kinship matrix for each."""
    ped_csv = pd.read_csv(fname, sep=sep)
    fams = []
    curr_fam = None
    ii = -1
    for row in ped_csv.itertuples():
        if row.ped != curr_fam:
            curr_fam = row.ped
            fams += [{}]
            ii += 1
            fams[ii]['ped'] = row.ped
            fams[ii]['inds'] = []
        fams[ii]['inds'] += [row]
    passing = np.ones(len(fams))
    for ii, fam in enumerate(fams):
        fam['cov'] = fam_to_cov(fam['inds'])
        if fam['cov'] is None:
            passing[ii] = 0
            print('problem in {}'.format(ii))
    return fams, passing

def get_parents(anc_set):
    """Get indicies of parents in an ancestor set."""
    return [anc[0] for anc in anc_set if anc[1] == 1]

def possible_dominant(fam, YY):
    """Calculate whether compatible with dominant Mendelian transmission of a single allele.
    given a family dictionary and binary phenotype vector.
    """
    inds = fam['inds']
    ancestor_sets = [get_ancestors(ii, inds, 0) for ii, _ in enumerate(inds)]
    originator_set = []
    for ii, anc_set in enumerate(ancestor_sets):
        if YY[ii] == 1:
            if len(anc_set) == 0:
                originator_set.append(ii)
            else:
                originator = True
                for anc in anc_set:
                    if (anc[1] == 1) and (YY[anc[0]] == 1):
                        originator = False
                if originator:
                    originator_set.append(ii)
    if len(originator_set) == 0:
        return False
    if len(originator_set) == 1:
        return True
    shared_parents = []
    for ii, originator in enumerate(originator_set):
        current_parents = get_parents(ancestor_sets[originator])
        if ii == 0:
            shared_parents += current_parents
        else:
            for parent in shared_parents:
                if parent not in current_parents:
                    shared_parents.remove(parent)

    if len(shared_parents) == 0:
        return False
    else:
        return True

def possible_dominant_missing(fam, YY, missing=None):
    """Ask whether there is a possible dominant transmission pattern
    when some data are missing.
    """
    if missing is None:
        return possible_dominant(fam, YY)

    if possible_dominant(fam, YY):
        return True
    num_missing = np.sum(missing)
    for poss_phen in product([0,1], repeat=num_missing):
        YY[missing] = np.array(poss_phen)
        if possible_dominant(fam, YY):
            return True
    return False

def calc_all_comps(fam, YY_set):
    """Calculate dominant compatibilities within a family for a set of pheotypes."""
    all_comps = [False]*YY_set.shape[0]
    comp_set = {}
    for ii in range(YY_set.shape[0]):
        key = repr(YY_set[ii,:])
        if key not in comp_set.keys():
            comp_set[key] = possible_dominant(fam, YY_set[ii,:])
        all_comps[ii] = comp_set[key]
    return np.array(all_comps)

def get_fam(fams, fam_name):
    """Return the family with a particular name from a list of family data."""
    for fam in fams:
        if fam['ped'] == fam_name:
            return fam
    return None

def get_fam_index(fams, fam_name):
    for ii, fam in enumerate(fams):
        if fam['ped'] == fam_name:
            return ii
    return None
