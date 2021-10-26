"""Functions for single arcs with multivariate output."""
import numpy as np
import numpy.polynomial.polynomial
from scipy.linalg import block_diag
from scipy.stats import multivariate_normal

from my_decorators import single_or_list


@single_or_list(kw='priors')
def make_mean_vect(priors):
    """Return an array of the prior means of the model parameters."""
    result = []
    for prior in priors:
        result.extend([prior['aMean'], prior['bMean'], prior['cMean']])
    return np.array(result)


@single_or_list(kw='priors')
def make_var_vect(priors):
    """Return an array of the prior variance of the model parameters."""
    result = []
    for prior in priors:
        result.extend([prior['aStd'], prior['bStd'], prior['cStd']])
    return np.square(np.array(result))


def make_design_matrix(xarray, output_dims=1):
    """Build the design matrix for the problem at hand."""
    return block_diag(*[np.array([[x**2, x, 1] for x in xarray]) for i in range(output_dims)])


@single_or_list(kw='priors')
def make_noise_cov(priors, input_vector):
    """Build the gaussian noise's covariance matrix."""
    return block_diag(*[(prior['noiseStd']**2)*np.identity(len(input_vector)) for prior in priors])


def is_static_prior(prior):
    """Tell if a prior is static in time."""
    return all(prior[param] == 0 for param in ['aMean', 'bMean', 'aStd', 'bStd'])


@single_or_list(kw='priors')
def arc_likelihood(priors, data, *, disable_opti=False):
    """Take a prior and a set of input/output values and return the log-likelihood of the data."""
    # 1 input, variable number of outputs
    (input_vector, output_vectors) = zip(*data)
    output_dim = len(priors)

    if len(priors) == 1 and is_static_prior(priors[0]) and not disable_opti:
        # Optimized version for static priors
        return _arc_likelihood_static_prior(priors[0], np.array(output_vectors))

    # Capital Phi in the doc
    design_matrix = make_design_matrix(input_vector, output_dim)
    # Bold mu in the doc
    mean_vect_prior = make_mean_vect(priors)
    # Means vector for the data
    mean_vect_data = np.matmul(design_matrix, mean_vect_prior)
    # Bold sigma^2 in the doc
    var_vect = make_var_vect(priors)
    # Noise component of covariance matrix
    noise_cov = make_noise_cov(priors, input_vector)
    # Covariance matrix for the data
    cov_mat_data = noise_cov + \
        (design_matrix @ np.diag(var_vect) @ np.transpose(design_matrix))
    # Bold t in the doc
    target_values = np.array(output_vectors).flatten('F')  # Flatten column first

    return multivariate_normal.logpdf(target_values, mean=mean_vect_data, cov=cov_mat_data)


def _arc_likelihood_static_prior(prior, data):
    """Take a static prior and a set of input/output values and return the log-likelihood of the data."""
    d = len(data)
    # Centered data w.r.t the mean mean
    cdata = data - prior['cMean']
    # The covariance matrix is xId + y
    x = prior['noiseStd'] ** 2
    y = prior['cStd'] ** 2
    # Its inverse is vId - w
    v = 1/x
    w = y/x/(x+d*y)
    # Its determinant has a closed formula
    # det = (x**(d-1)*(x+d*y))
    logdet = (d-1) * np.log(x) + np.log(x+d*y)
    # This is the exponent in the multivariate Gaussian density formula
    exponent = v * (cdata @ cdata) - w * sum(cdata) ** 2

    loglik = -(exponent + d * np.log(2 * np.pi) + logdet)/2
    return loglik


def arc_max_a_posteriori(prior, data):
    """Take a prior and a set of input/output values and return the most likely arc with its loglikelihood."""
    # NYI
    return {"LL": None, "Arc": None}


def arc_max_likelihood(data, return_estimates=False):
    """Return the maximum likelihood arc for a set of input/output values."""
    (input_vector, output_vector) = zip(*data)
    polyfit = np.polynomial.polynomial.polyfit(input_vector, output_vector, 2)
    if return_estimates:
        return list(reversed(polyfit)), np.polynomial.polynomial.polyval(input_vector, polyfit)
    else:
        return list(reversed(polyfit))


def normalize_x(data_slice, linear_sampling=True):
    """Normalize input variable (or generate it if needed) to range from 0 to 1."""
    # TODO: Automatically detect if linear sampling (beat-wise) or not (note-wise)
    if linear_sampling:
        if len(data_slice) == 1:
            return [(0, data_slice[0])]
        else:
            return [(float(i)/(len(data_slice)-1), data_point) for (i, data_point) in enumerate(data_slice)]
    else:
        max_x, _ = data_slice[-1]
        return [(float(x)/max_x, y) for x, y in data_slice]


def known_segmentation_max_likelihood(data, segmentation):
    """Perform a ML estimation with known boundaries."""
    y = []  # ML estimation of the denoised data
    models = []  # ML coefficient estimates
    lengths = []
    for (bound_curr, bound_next) in zip(segmentation, segmentation[1:]):
        data_pairs = normalize_x(data[(bound_curr+1):bound_next+1])
        model, values = arc_max_likelihood(data_pairs, return_estimates=True)
        models.append(model)
        y.extend(values)
        lengths.append(bound_next-bound_curr)
    return y, models, lengths
