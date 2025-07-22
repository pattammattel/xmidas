""" Helper Functions (make a class later)"""


import h5py
import logging
import numpy as np
import pandas as pd
import os
import scipy.optimize as opt
import scipy.stats as stats

import sklearn.decomposition as sd
import sklearn.cluster as sc
from scipy.signal import savgol_filter
from skimage.transform import resize
from skimage import filters
from sklearn import linear_model

# from larch.xafs import pre_edge, preedge, mback
# from larch.io import read_ascii, read_athena
# from larch import Group
import xraydb
from pystackreg import StackReg

from xmidas.utils.larch_norm import pre_edge_simple


logger = logging.getLogger()

def get_xrf_data(h='h5file'):
    """
    get xrf stack from h5 data generated at NSLS-II beamlines

     Arguments:
        h5/hdf5 file

     Returns:
         norm_xrf_stack -  xrf stack image normalized with Io
         mono_e  - excitation enegy used for xrf
         beamline - identity of the beamline
         Io_avg - an average Io value, used before taking log

    """

    f = h5py.File(h, "r")

    if list(f.keys())[0] == "xrfmap":
        logger.info("Data from HXN/TES/SRX")
        beamline = f["xrfmap/scan_metadata"].attrs["scan_instrument_id"]

        try:

            beamline_scalar = {"HXN": 2, "SRX": 0, "TES": 0}

            if beamline in beamline_scalar.keys():

                Io = np.array(f["xrfmap/scalers/val"])[:, :, beamline_scalar[beamline]]
                raw_xrf_stack = np.array(f["xrfmap/detsum/counts"])
                norm_xrf_stack = raw_xrf_stack
                Io_avg = int(remove_nan_inf(Io).mean())
            else:
                logger.error("Unknown Beamline Scalar")
        except Exception:
            logger.warning("Unknown Scalar: Raw Detector count in use")
            norm_xrf_stack = np.array(f["xrfmap/detsum/counts"])

    elif list(f.keys())[0] == "xrmmap":
        logger.info("Data from XFM")
        beamline = "XFM"
        raw_xrf_stack = np.array(f["xrmmap/mcasum/counts"])
        Io = np.array(f["xrmmap/scalars/I0"])
        norm_xrf_stack = raw_xrf_stack
        Io_avg = int(remove_nan_inf(Io).mean())

    elif list(f.keys())[0] == "MAPS":
        logger.info("MAPS file")
        beamline = "APS"
        raw_xrf_stack = np.array(f["MAPS/Spectra/mca_arr"])
        Io = 1 #have to find the name of the scalar
        norm_xrf_stack = raw_xrf_stack.transpose((1, 2, 0))
        Io_avg = int(remove_nan_inf(Io).mean())

    else:
        logger.error("Unknown Data Format")

    try:
        mono_e = int(f["xrfmap/scan_metadata"].attrs["instrument_mono_incident_energy"] * 1000)
        logger.info("Excitation energy was taken from the h5 data")

    except Exception:
        mono_e = 12000
        logger.info(f"Unable to get Excitation energy from the h5 data; using default value {mono_e} KeV")

    return remove_nan_inf(norm_xrf_stack.transpose((2, 0, 1))), mono_e + 1500, beamline, Io_avg


def remove_nan_inf(im):
    im = np.array(im, dtype=np.float32)
    im[np.isnan(im)] = 0
    im[np.isinf(im)] = 0
    return im


def rebin_image(im, bin_factor):
    arrx, arry = np.shape(im)
    if arrx / bin_factor != int or arrx / bin_factor != int:
        logger.error("Invalid Binning")

    else:
        shape = (arrx / bin_factor, arry / bin_factor)
        return im.reshape(shape).mean(-1).mean(1)


def remove_hot_pixels(image_array, NSigma=5):
    image_array = remove_nan_inf(image_array)
    a, b, c = np.shape(image_array)
    img_stack2 = np.zeros((a, b, c))
    for i in range(a):
        im = image_array[i, :, :]
        im[abs(im) > np.std(im) * NSigma] = im.mean()
        img_stack2[i, :, :] = im
    return img_stack2


def smoothen(image_array, w_size=5):
    a, b, c = np.shape(image_array)
    spec2D_Matrix = np.reshape(image_array, (a, (b * c)))
    smooth2D_Matrix = savgol_filter(spec2D_Matrix, w_size, w_size - 2, axis=0)
    return remove_nan_inf(np.reshape(smooth2D_Matrix, (a, b, c)))


def resize_stack(image_array, upscaling=False, scaling_factor=2):
    en, im1, im2 = np.shape(image_array)

    if upscaling:
        im1_ = im1 * scaling_factor
        im2_ = im2 * scaling_factor
        img_stack_resized = resize(image_array, (en, im1_, im2_))

    else:
        im1_ = int(im1 / scaling_factor)
        im2_ = int(im2 / scaling_factor)
        img_stack_resized = resize(image_array, (en, im1_, im2_))

    return img_stack_resized


def normalize(image_array, norm_point=-1):
    norm_stack = image_array / image_array[norm_point]
    return remove_nan_inf(norm_stack)


def remove_edges(image_array):
    # z, x, y = np.shape(image_array)
    return image_array[:, 1:-1, 1:-1]


def background_value(image_array):
    img = image_array.mean(0)
    img_h = img.mean(0)
    img_v = img.mean(1)
    h = np.gradient(img_h)
    v = np.gradient(img_v)
    bg = np.min([img_h[h == h.max()], img_v[v == v.max()]])
    return bg


def background_subtraction(img_stack, bg_percentage=10):
    img_stack = remove_nan_inf(img_stack)
    a, b, c = np.shape(img_stack)
    ref_image = np.reshape(img_stack.mean(0), (b * c))
    bg_ratio = int((b * c) * 0.01 * bg_percentage)
    bg_ = np.max(sorted(ref_image)[0:bg_ratio])
    bged_img_stack = img_stack - bg_[:, np.newaxis, np.newaxis]
    return bged_img_stack


def background_subtraction2(img_stack, bg_percentage=10):
    img_stack = remove_nan_inf(img_stack)
    a, b, c = np.shape(img_stack)
    bg_ratio = int((b * c) * 0.01 * bg_percentage)
    bged_img_stack = img_stack.copy()

    for n, img in enumerate(img_stack):
        bg_ = np.max(sorted(img.flatten())[0:bg_ratio])
        print(bg_)
        bged_img_stack[n] = img - bg_

    return remove_nan_inf(bged_img_stack)


def background1(img_stack):
    img = img_stack.sum(0)
    img_h = img.mean(0)
    img_v = img.mean(1)
    h = np.gradient(img_h)
    v = np.gradient(img_v)
    bg = np.min([img_h[h == h.max()], img_v[v == v.max()]])
    return bg


def get_sum_spectra(image_array):
    spec = np.nansum(image_array, axis=(1, 2))
    return spec


def get_mean_spectra(image_array):
    spec = np.nanmean(image_array, axis=(1, 2))
    return spec


def flatten_(image_array):
    z, x, y = np.shape(image_array)
    flat_array = np.reshape(image_array, (x * y, z))
    return flat_array


def image_to_pandas(image_array):
    a, b, c = np.shape(image_array)
    im_array = np.reshape(image_array, ((b * c), a))
    a, b = im_array.shape
    df = pd.DataFrame(
        data=im_array[:, :], columns=["e" + str(i) for i in range(b)], 
        index=["s" + str(i) for i in range(a)]
    )
    return df


def image_to_pandas2(image_array):
    a, b, c = np.shape(image_array)
    im_array = np.reshape(image_array, (a, (b * c)))
    a, b = im_array.shape
    df = pd.DataFrame(
        data=im_array[:, :], index=["e" + str(i) for i in range(a)], 
        columns=["s" + str(i) for i in range(b)]
    )
    return df


def neg_log(image_array):
    absorb = -1 * np.log(image_array)
    return remove_nan_inf(absorb)


def clean_stack(img_stack, auto_bg=False, bg_percentage=5):
    a, b, c = np.shape(img_stack)

    if auto_bg is True:
        bg_ = background1(img_stack)

    else:
        sum_spec = (img_stack.sum(1)).sum(1)
        ref_stk_num = np.where(sum_spec == sum_spec.max())[-1]

        ref_image = np.reshape(img_stack[ref_stk_num], (b * c))
        bg_ratio = int((b * c) * 0.01 * bg_percentage)
        bg_ = np.max(sorted(ref_image)[0:bg_ratio])

    bg = np.where(img_stack[ref_stk_num] > bg_, img_stack[ref_stk_num], 0)
    bg2 = np.where(bg < bg_, bg, 1)

    bged_img_stack = img_stack * bg2

    return remove_nan_inf(bged_img_stack)


def subtractBackground(im_stack, bg_region):
    if bg_region.ndim == 3:
        bg_region_ = np.mean(bg_region, axis=(1, 2))

    elif bg_region.ndim == 2:
        bg_region_ = np.mean(bg_region, axis=1)

    else:
        bg_region_ = bg_region

    return im_stack - bg_region_[:, np.newaxis, np.newaxis]


def classify(img_stack, correlation="Pearson"):
    img_stack_ = img_stack
    a, b, c = np.shape(img_stack_)
    norm_img_stack = normalize(img_stack_)
    f = np.reshape(norm_img_stack, (a, (b * c)))

    max_x, max_y = np.where(norm_img_stack.sum(0) == (norm_img_stack.sum(0)).max())
    ref = norm_img_stack[:, int(max_x), int(max_y)]
    corr = np.zeros(len(f.T))
    for s in range(len(f.T)):
        if correlation == "Kendall":
            r, p = stats.kendalltau(ref, f.T[s])
        elif correlation == "Pearson":
            r, p = stats.pearsonr(ref, f.T[s])

        corr[s] = r

    cluster_image = np.reshape(corr, (b, c))
    return (cluster_image ** 3), img_stack_


def correlation_kmeans(img_stack, n_clusters, correlation="Pearson"):
    img, bg_image = classify(img_stack, correlation)
    img[np.isnan(img)] = -99999
    X = img.reshape((-1, 1))
    k_means = sc.KMeans(n_clusters)
    k_means.fit(X)

    X_cluster = k_means.labels_
    X_cluster = X_cluster.reshape(img.shape) + 1

    return X_cluster


def cluster_stack(
    im_array, method="KMeans", n_clusters_=4, decomposed=False, 
    decompose_method="PCA", decompose_comp=2
    ):
    a, b, c = im_array.shape

    if method == "Correlation-Kmeans":

        X_cluster = correlation_kmeans(im_array, n_clusters_, correlation="Pearson")

    else:

        methods = {
            "MiniBatchKMeans": sc.MiniBatchKMeans,
            "KMeans": sc.KMeans,
            "MeanShift": sc.MeanShift,
            "Spectral Clustering": sc.SpectralClustering,
            "Affinity Propagation": sc.AffinityPropagation,
        }

        if decomposed:
            im_array = denoise_with_decomposition(im_array, method_=decompose_method, 
                                                  n_components=decompose_comp)

        flat_array = np.reshape(im_array, (a, (b * c)))
        init_cluster = methods[method](n_clusters=n_clusters_)
        init_cluster.fit(np.transpose(flat_array))
        X_cluster = init_cluster.labels_.reshape(b, c) + 1

    decon_spectra = np.zeros((a, n_clusters_))
    decon_images = np.zeros((n_clusters_, b, c))

    for i in range(n_clusters_):
        mask_i = np.where(X_cluster == (i + 1), X_cluster, 0)
        spec_i = get_sum_spectra(im_array * mask_i)
        decon_spectra[:, i] = spec_i
        decon_images[i] = im_array.sum(0) * mask_i

    return decon_images, X_cluster, decon_spectra

def kmeans_variance(im_array):
    a, b, c = im_array.shape
    flat_array = np.reshape(im_array, (a, (b * c)))
    var = np.arange(24)
    clust_n = np.arange(24) + 2

    for clust in var:
        init_cluster = sc.KMeans(n_clusters=int(clust + 2))
        init_cluster.fit(np.transpose(flat_array))
        var_ = init_cluster.inertia_
        var[clust] = np.float64(var_)

    return clust_n, var


def pca_scree(im_stack):
    new_image = im_stack.transpose(2, 1, 0)
    x, y, z = np.shape(new_image)
    img_ = np.reshape(new_image, (x * y, z))
    # pca = sd.PCA(z)
    # pca.fit(img_)
    pca = sd.TruncatedSVD(z - 1)
    pca.fit(img_)
    var = pca.singular_values_
    # var = pca.singular_values_
    return var


def decompose_stack(im_stack, decompose_method="PCA", 
                    n_components_=3, generate_label_img = True):
    
    new_image = im_stack.transpose(2, 1, 0)
    new_image[new_image<0] = 0
	
    x, y, z = np.shape(new_image)
    img_ = np.reshape(new_image, (x * y, z))
    methods_dict = {
        "PCA": sd.PCA,
        "IncrementalPCA": sd.IncrementalPCA,
        "NMF": sd.NMF,
        "FastICA": sd.FastICA,
        "DictionaryLearning": sd.MiniBatchDictionaryLearning,
        "FactorAnalysis": sd.FactorAnalysis,
        "TruncatedSVD": sd.TruncatedSVD,
    }

    _mdl = methods_dict[decompose_method](n_components=n_components_)

    ims = (_mdl.fit_transform(img_).reshape(x, y, n_components_)).transpose(2, 1, 0)
    spcs = _mdl.components_.transpose()
    decon_spetra = np.zeros((z, n_components_))
    decom_map = np.zeros((ims.shape))

    if generate_label_img:
        for i in range(n_components_):
            f = ims.copy()[i]
            f[f < 0] = 0
            f = np.where(f > 0 * np.std(f), f, 0)
            spec_i = ((new_image.T * f).sum(1)).sum(1)
            decon_spetra[:, i] = spec_i

            f[f > 0] = i + 1
            decom_map[i] = f
        decom_map = decom_map.sum(0)

    return np.float32(ims), spcs, decon_spetra, decom_map



def denoise_with_decomposition(img_stack, method_="PCA", n_components=4):
    new_image = img_stack.transpose(2, 1, 0)
    x, y, z = np.shape(new_image)
    img_ = np.reshape(new_image, (x * y, z))

    methods_dict = {
        "PCA": sd.PCA,
        "IncrementalPCA": sd.IncrementalPCA,
        "NMF": sd.NMF,
        "FastICA": sd.FastICA,
        "DictionaryLearning": sd.DictionaryLearning,
        "FactorAnalysis": sd.FactorAnalysis,
        "TruncatedSVD": sd.TruncatedSVD,
    }

    decomposed = methods_dict[method_](n_components=n_components)

    ims = (decomposed.fit_transform(img_).reshape(x, y, n_components)).transpose(2, 1, 0)
    ims[ims < 0] = 0
    ims[ims > 0] = 1
    mask = ims.sum(0)
    mask[mask > 1] = 1
    # mask = uniform_filter(mask)
    filtered = img_stack * mask
    # plt.figure()
    # plt.imshow(filtered.sum(0))
    # plt.title('background removed')
    # plt.show()
    return remove_nan_inf(filtered)


def interploate_E(refs, e):
    n = np.shape(refs)[1]
    refs = np.array(refs)
    ref_e = refs[:, 0]
    ref = refs[:, 1:n]
    all_ref = []
    for i in range(n - 1):
        ref_i = np.interp(e, ref_e, ref[:, i])
        all_ref.append(ref_i)
    return np.array(all_ref)

def rfactor(spectrum_experimental, spectrum_fit):
    r"""
    Computes R-factor based on two spectra

    Parameters
    ----------
    spectrum_experimental : ndarray
        spectrum data on which fitting is performed (N elements)

    spectrum_fit : ndarray
        fitted spectrum (weighted sum of spectrum components, N elements)

    Returns
    -------
        float, the value of R-factor
    """

    # Compute R-factor
    dif = spectrum_experimental - spectrum_fit
    dif_sum = np.sum(np.abs(dif), axis=0)
    data_sum = np.sum(np.abs(spectrum_experimental), axis=0)

    # Avoid accidental division by zero (or a very small number)
    data_sum = np.clip(data_sum, a_min=1e-30, a_max=None)

    return dif_sum / data_sum

def rfactor_compute(spectrum, fit_results, ref_spectra):
    r"""
    Computes R-factor for the fitting results

    Parameters
    ----------
    spectrum : ndarray
        spectrum data on which fitting is performed (N elements)

    fit_results : ndarray
        results of fitting (coefficients, K elements)

    ref_spectra : 2D ndarray
        reference spectra used for fitting (NxK element array)

    Returns
    -------
        float, the value of R-factor
    """

    # Check if input parameters are valid
    assert (
        spectrum.ndim == 1 or spectrum.ndim == 2
    ), "Parameter 'spectrum' must be 1D or 2D array, ({spectrum.ndim})"
    assert spectrum.ndim == fit_results.ndim, (
        f"Spectrum data (ndim = {spectrum.ndim}) and fitting results "
        f"(ndim = {fit_results.ndim}) must have the same number of dimensions"
    )
    assert ref_spectra.ndim == 2, "Parameter 'ref_spectra' must be 2D array, ({ref_spectra.ndim})"
    assert spectrum.shape[0] == ref_spectra.shape[0], (
        f"Arrays 'spectrum' ({spectrum.shape}) and 'ref_spectra' ({ref_spectra.shape}) "
        "must have the same number of data points"
    )
    assert fit_results.shape[0] == ref_spectra.shape[1], (
        f"Arrays 'fit_results' ({fit_results.shape}) and 'ref_spectra' ({ref_spectra.shape}) "
        "must have the same number of spectrum points"
    )
    if spectrum.ndim == 2:  # Only if multiple spectra are processed
        assert spectrum.shape[1] == fit_results.shape[1], (
            f"Arrays 'spectrum' {spectrum.shape} and 'fit_results' {fit_results.shape}"
            "must have the same number of columns"
        )

    spectrum_fit = np.matmul(ref_spectra, fit_results)
    return rfactor(spectrum, spectrum_fit)


def fitting_admm(data, ref_spectra, *, rate=0.2, maxiter=100, epsilon=1e-30, 
                 non_negative=True, weight_to_whiteline = False):
    r"""
    Fitting of multiple spectra using ADMM method.

    Parameters
    ----------

    data : ndarray(float), 2D
        array holding multiple observed spectra, shape (K, N), where K is the number of energy points,
        and N is the number of spectra

    absorption_refs : ndarray(float), 2D
        array of references, shape (K, Q), where Q is the number of references.

    maxiter : int
        maximum number of iterations. Optimization may stop prematurely if convergence criteria are met.

    rate : float
        descent rate for optimization algorithm. Currently is used only for ADMM fitting (1/lambda).

    epsilon : float
        small value used in stopping criterion of ADMM optimization algorithm.

    non_negative : bool
        if True, then the solution is guaranteed to be non-negative

    Returns
    -------

    map_data_fitted : ndarray(float), 2D
        fitting results, shape (Q, N), where Q is the number of references and N is the number of spectra.

    map_rfactor : ndarray(float), 2D
        map that represents R-factor for the fitting, shape (M,N).

    convergence : ndarray(float), 1D
        convergence data returned by ADMM algorithm

    feasibility : ndarray(float), 1D
        feasibility data returned by ADMM algorithm

    The prototype for the ADMM fitting function was implemented by Hanfei Yan in Matlab.
    """
    #print(type(data))
    if data.ndim == 1:
        data = np.expand_dims(data, axis=1) #"Data array 'data' must have 2 dimensions"
    #print(type(data))
    assert ref_spectra.ndim == 2, "Data array 'ref_spectra' must have 2 dimensions"

    n_pts = data.shape[0]
    n_pixels = data.shape[1]
    n_pts_2 = ref_spectra.shape[0]
    n_refs = ref_spectra.shape[1]

    assert (
        n_pts == n_pts_2
    ), f"ADMM fitting: number of spectrum points in data ({n_pts}) and references ({n_pts_2}) do not match."

    assert rate > 0.0, f"ADMM fitting: parameter 'rate' is zero or negative ({rate:.6g})"

    assert maxiter > 0, f"ADMM fitting: parameter 'maxiter' is zero or negative ({rate})"

    assert epsilon > 0.0, f"ADMM fitting: parameter 'epsilon' is zero or negative ({rate:.6g})"

    wgt = np.ones(len(data))

    #print(wgt)

    if weight_to_whiteline:
        wgt[0:-15] = 0.4
        wgt[-25:-1]=0.5

        #print(wgt)

    y = data
    # Calculate some quantity to be used in the iteration
    A = ref_spectra
    At = np.transpose(A)
    #print(np.shape(At), np.shape(y), np.shape(np.diag(wgt)))
    #print(np.diag(wgt))

    z = A.T @ np.diag(wgt) @ y
    c = A.T @ np.diag(wgt) @ A

    # Initialize variables
    w = np.ones(shape=[n_refs, n_pixels])
    u = np.zeros(shape=[n_refs, n_pixels])

    # Feasibility test: x == w
    convergence = np.zeros(shape=[maxiter])
    feasibility = np.zeros(shape=[maxiter])

    dg = np.eye(n_refs, dtype=float) * rate
    m1 = np.linalg.inv((c + dg))

    n_iter = 0
    for i in range(maxiter):
        m2 = z + (w - u) * rate
        x = np.matmul(m1, m2)
        w_updated = x + u
        if non_negative:
            w_updated = w_updated.clip(min=0)
        u = u + x - w_updated

        conv = np.linalg.norm(w_updated - w) / np.linalg.norm(w_updated)
        convergence[i] = conv
        feasibility[i] = np.linalg.norm(x - w_updated)

        w = w_updated

        if conv < epsilon:
            n_iter = i + 1
            break

    # Compute R-factor
    rfactor = rfactor_compute(data, w, ref_spectra)
    #rfactor = 1

    convergence = convergence[:n_iter]
    feasibility = feasibility[:n_iter]

    return w, rfactor, convergence, feasibility


def getStats(spec, fit, num_refs=2):
    stats = {}
    
    SS_tot = np.sum((spec - np.mean(spec))**2)
    SS_res = np.sum((spec - fit) ** 2)
    #r_factor = (np.sum(spec - fit) ** 2) / np.sum(spec ** 2) 
    r_factor = 1 - (SS_res / SS_tot) #temp to get r2 array for paper
    stats["R_Factor"] = np.around(r_factor, 5)

    y_mean = np.sum(spec) / len(spec)
    #SS_tot = np.sum((spec - y_mean) ** 2)

    r_square = 1 - (SS_res / SS_tot)
    stats["R_Square"] = np.around(r_square, 4)

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html
    # https://en.wikipedia.org/wiki/Chi-squared_distribution
    # https://ned.ipac.caltech.edu/level5/Leo/Stats2_4.html
    chisq = np.sum(((spec - fit) ** 2) / np.var(spec))
    # chisq = np.sum((spec - fit) ** 2)
    stats["Chi_Square"] = np.around(chisq, 5)

    red_chisq = chisq / (len(spec) - num_refs)
    stats["Reduced Chi_Square"] = np.around(red_chisq, 5)

    return stats


def xanes_fitting_1D(spec, e_list, refs, 
                     method="NNLS", alphaForLM=0.01):
    
    """Linear combination fit of image data with reference standards"""

    #check for energy unit in e_list vs ref
    #print(f"at 1D fit function{e_list[0]}")
    

    # #energy list should be corrected for shifts before passing here
    # if len(str(int(e_list[0]))) <3:
    #     e_list *=1000
    # #print(e_list)
    
    # #print(refs[0][0])
    # if len(str(int(refs[0][0]))) <3:
    #     refs[:,0] =refs[:,0]*1000
    # #print(refs[0])

    spec = np.nan_to_num(spec)
    refs = np.nan_to_num(refs)
    int_refs = interploate_E(refs, e_list)

       
    if method == "NNLS":
        coeffs, r = opt.nnls(int_refs.T, spec)
        #print(f"{coeffs}: nnls")

    elif method == "LASSO":
        lasso = linear_model.Lasso(positive=True, alpha=alphaForLM)  # lowering alpha helps with 1D fits
        fit_results = lasso.fit(int_refs.T, spec)
        coeffs = fit_results.coef_

    elif method == "RIDGE":
        ridge = linear_model.Ridge(alpha=alphaForLM)
        fit_results = ridge.fit(int_refs.T, spec)
        coeffs = fit_results.coef_

    elif method == "ADMM":
        coeffs,r_factor,convergence, feasibility = fitting_admm(spec, 
                                                                int_refs.T, 
                                                                maxiter=100, 
                                                                rate=alphaForLM, 
                                                                epsilon=1e-30)
        coeffs = np.squeeze(coeffs.T)
        # print(f"{coeffs}; ADMM")

    fit = np.dot(coeffs , int_refs)
    stats = getStats(spec, fit, num_refs=np.min(np.shape(int_refs.T)))

    return stats, coeffs


def xanes_fitting(im_stack, e_list, refs, method="NNLS", alphaForLM=0.1, binStack=False):
    """Linear combination fit of image data with reference standards"""

    if binStack:
        im_stack = resize_stack(im_stack, scaling_factor=4)

    en, im1, im2 = np.shape(im_stack)
    im_array = im_stack.reshape(en, im1 * im2)
    coeffs_arr = []
    r_factor_arr = []
    lasso = linear_model.Lasso(positive=True, alpha=alphaForLM)
    if not method=="ADMM":
        for n, i in enumerate(range(im1 * im2)):
            stats, coeffs = xanes_fitting_1D(im_array[:, i], 
                                             e_list, 
                                             refs, 
                                             method=method, 
                                             alphaForLM=alphaForLM)
            
            coeffs_arr.append(coeffs)
            r_factor_arr.append(stats["R_Factor"])

        abundance_map = np.reshape(coeffs_arr, (im1, im2, -1))
        r_factor_im = np.reshape(r_factor_arr, (im1, im2))

    elif method=="ADMM":
        int_refs = interploate_E(refs,e_list)
        coeffs_arr,r_factor_im,convergence, feasibility = fitting_admm(im_array, 
                                                                       int_refs.T, 
                                                                       maxiter=100, 
                                                                       rate=alphaForLM, 
                                                                       epsilon=1e-30)
        # print(coeffs_arr.shape)

        #plt.imshow(coeffs_arr[1].reshape(im1, im2))

        abundance_map = np.reshape((coeffs_arr.T), (im1, im2, -1))

    return abundance_map, r_factor_im, np.mean(coeffs_arr, axis=0)


def xanes_fitting_Line(im_stack, e_list, refs, method="NNLS", alphaForLM=0.05):
    """Linear combination fit of image data with reference standards"""
    en, im1, im2 = np.shape(im_stack)
    im_array = np.mean(im_stack, 2)
    coeffs_arr = []
    meanStats = {"R_Factor": 0, "R_Square": 0, "Chi_Square": 0, "Reduced Chi_Square": 0}

    for i in range(im1):
        stats, coeffs = xanes_fitting_1D(im_array[:, i], e_list, refs, 
                                         method=method, alphaForLM=alphaForLM)
        coeffs_arr.append(coeffs)
        for key in stats.keys():
            meanStats[key] += stats[key]

    for key, vals in meanStats.items():
        meanStats[key] = np.around((vals / im1), 5)

    return meanStats, np.mean(coeffs_arr, axis=0)

#TODO ADMM maybe faster here
def xanes_fitting_Binned(im_stack, e_list, refs, method="NNLS", alphaForLM=0.05):
    """Linear combination fit of image data with reference standards"""

    im_stack = resize_stack(im_stack, scaling_factor=10)
    # use a simple filter to find threshold value
    val = filters.threshold_otsu(im_stack[-1])
    en, im1, im2 = np.shape(im_stack)
    im_array = im_stack.reshape(en, im1 * im2)
    coeffs_arr = []
    meanStats = {"R_Factor": 0, "R_Square": 0, "Chi_Square": 0, "Reduced Chi_Square": 0}

    specs_fitted = 0
    total_spec = im1 * im2
    for i in range(total_spec):
        spec = im_array[:, i]
        # do not fit low intensity/background regions
        if spec[-1] > val:
            specs_fitted += 1
            stats, coeffs = xanes_fitting_1D(spec / spec[-1], e_list, refs, 
                                             method=method, alphaForLM=alphaForLM)
            coeffs_arr.append(coeffs)
            for key in stats.keys():
                meanStats[key] += stats[key]
        else:
            pass

    for key, vals in meanStats.items():
        meanStats[key] = np.around((vals / specs_fitted), 6)
    # print(f"{specs_fitted}/{total_spec}")
    return meanStats, np.mean(coeffs_arr, axis=0)


def create_df_from_nor(athenafile="fe_refs.nor"):
    """create pandas dataframe from athena nor file, first column
    is energy and headers are sample names"""

    refs = np.loadtxt(athenafile)
    n_refs = refs.shape[-1]
    skip_raw_n = n_refs + 6

    df = pd.read_table(
        athenafile, delim_whitespace=True, skiprows=skip_raw_n, header=None, usecols=np.arange(0, n_refs)
    )
    df2 = pd.read_table(
        athenafile, delim_whitespace=True, skiprows=skip_raw_n - 1, usecols=np.arange(0, n_refs + 1)
    )
    new_col = df2.columns.drop("#")
    df.columns = new_col
    return df, list(new_col)


def create_df_from_nor_try2(athenafile="fe_refs.nor"):
    """create pandas dataframe from athena nor file, first column
    is energy and headers are sample names"""

    refs = np.loadtxt(athenafile)
    n_refs = refs.shape[-1]
    df_refs = pd.DataFrame(refs)

    df = pd.read_csv(athenafile, header=None)
    new_col = list((str(df.iloc[n_refs + 5].values)).split(" ")[2::2])
    df_refs.columns = new_col

    return df_refs, list(new_col)


def energy_from_logfile(logfile="maps_log_tiff.txt"):
    df = pd.read_csv(logfile, header=None, delim_whitespace=True, skiprows=9)
    return df[9][df[7] == "energy"].values.astype(float)


# def xanesNormalization(e, mu, e0=7125, step=None,
#                        nnorm=2, nvict=0, pre1=None, pre2=-50,
#                        norm1=100, norm2=None, method="pre_edge",
#                        useFlattened=False,  Elemline = "Fe_K"):

#     elem, line = Elemline.split('_')
#     elemZ = xraydb.atomic_number(elem)
#     dat = Group(name='larchgroup', col1=e, col2=mu)


#     if method == "guess":
#         result = preedge(e, mu, e0, step=step, nnorm=nnorm, nvict=nvict)

#         return result["pre1"], result["pre2"], result["norm1"], result["norm2"]

#     elif method == "mback":
#         mback(e,mu, group=dat, z=elemZ, edge=line, e0=e0,fit_erfc=False)
#         return dat.f2, dat.fpp

#     else:
#         pre_edge(e, mu,group=dat,e0=e0, step=step, nnorm=nnorm,nvict=nvict, pre1=pre1,
#                  pre2=pre2, norm1=norm1,norm2=norm2, make_flat = True)

#         if useFlattened:
#             normSpec = dat.flat
#         else:
#             normSpec = dat.norm

#         return dat.pre_edge, dat.post_edge, normSpec

def xanesNormalization(e, mu, e0=7125, nnorm=2, nvict=0,
                       pre1=None, pre2=-50, norm1=None, norm2=None,
                       useFlattened=False, Elemline="Fe_K"):
    """
    e, mu       : energy & absorption arrays
    e0          : edge energy (float)
    nnorm       : degree of post-edge polynomial
    nvict       : exponent for pre-edge fit weighting
    pre1, pre2  : relative pre-edge fitting bounds
    norm1, norm2: relative post-edge fitting bounds
    useFlattened: ignored in this simple version
    Elemline    : placeholder, no longer used
    """
    # perform pre-edge subtraction + normalization
    res = pre_edge_simple(
        e, mu,
        e0=e0, nnorm=nnorm, nvict=nvict,
        pre1=pre1, pre2=pre2,
        norm1=norm1, norm2=norm2
    )

    pre_edge_arr  = res["pre_edge"]
    post_edge_arr = res["post_edge"]
    norm_arr      = res["norm"]

    return pre_edge_arr, post_edge_arr, norm_arr

# def xanesNormStack(e_list, im_stack, e0=7125, step=None,
#                    nnorm=2, nvict=0, pre1=None, pre2=-50,
#                    norm1=100, norm2=None, useFlattened=False, ignorePostEdgeNorm=False):
#     en, im1, im2 = np.shape(im_stack)
#     im_array = im_stack.reshape(en, im1 * im2)
#     normedStackArray = np.zeros_like(im_array)
#     dat = Group(name='larchgroup', col1=e_list, col2=get_mean_spectra(im_stack))


#     for i in range(im1 * im2):
#         pre_edge(e_list, im_array[:, i], e0=e0, group=dat, step=step, nnorm=nnorm,
#                  nvict=nvict, pre1=pre1, pre2=pre2, norm1=norm1, norm2=norm2,make_flat = True)

#         if useFlattened:
#             normSpec = dat.flat
#         else:
#             normSpec = dat.norm

#         if ignorePostEdgeNorm:
#             normedStackArray[:, i] = normSpec * dat.post_edge
#         else:
#             normedStackArray[:, i] = normSpec

#     return remove_nan_inf(np.reshape(normedStackArray, (en, im1, im2)))



def xanesNormStack(e_list, im_stack,
                   e0=7125, nnorm=2, nvict=0,
                   pre1=None, pre2=-50,
                   norm1=None, norm2=None,
                   ignorePostEdgeNorm=False):
    """
    Apply XANES normalization to every spectrum in a 3D stack.

    Parameters
    ----------
    e_list : 1D array
        Energy grid.
    im_stack : 3D array, shape (nE, nX, nY)
        Raw absorption stack.
    e0, nnorm, nvict, pre1, pre2, norm1, norm2 : fit params
        Passed to pre_edge_simple.
    ignorePostEdgeNorm : bool
        If True, multiply normalized spec by post-edge baseline.

    Returns
    -------
    normed_stack : 3D array, same shape as im_stack
    """
    nE, nX, nY = im_stack.shape
    flat_in = im_stack.reshape(nE, -1)
    flat_out = np.zeros_like(flat_in)

    for i in range(flat_in.shape[1]):
        spec = flat_in[:, i]
        res = pre_edge_simple(
            e_list, spec,
            e0=e0, nnorm=nnorm, nvict=nvict,
            pre1=pre1, pre2=pre2,
            norm1=norm1, norm2=norm2
        )
        norm_spec  = res["norm"]
        post_edge  = res["post_edge"]

        if ignorePostEdgeNorm:
            flat_out[:, i] = norm_spec * post_edge
        else:
            flat_out[:, i] = norm_spec

    # reshape back and remove any NaN/inf
    return remove_nan_inf(flat_out.reshape(nE, nX, nY))

def getDeconvolutedXANESSpectrum(xanesStack, chemMapStack, energy, clusterSigma=1):
    compXanesSpetraAll = pd.DataFrame()
    compXanesSpetraAll['Energy'] = energy

    for n, compImage in enumerate(chemMapStack):
        mask = np.where(compImage > clusterSigma * np.std(compImage), compImage, 0)
        compXanesSpetraAll[f'Component_{n + 1}'] = get_mean_spectra(xanesStack * mask)
    return compXanesSpetraAll


def align_stack(
    stack_img, ref_image_void=True, ref_stack=None, 
    transformation=StackReg.TRANSLATION, reference="previous"
):

    """Image registration flow using pystack reg"""

    # all the options are in one function

    sr = StackReg(transformation)

    if ref_image_void:
        tmats_ = sr.register_stack(stack_img, reference=reference)

    else:
        tmats_ = sr.register_stack(ref_stack, reference=reference)
        out_ref = sr.transform_stack(ref_stack)

    out_stk = sr.transform_stack(stack_img, tmats=tmats_)
    return np.float32(out_stk), tmats_


def align_simple(stack_img, transformation=StackReg.TRANSLATION, reference="previous"):

    sr = StackReg(transformation)
    tmats_ = sr.register_stack(stack_img, reference="previous")
    for i in range(10):
        out_stk = sr.transform_stack(stack_img, tmats=tmats_)
    return np.float32(out_stk)


def align_with_tmat(stack_img, tmat_file, transformation=StackReg.TRANSLATION):

    sr = StackReg(transformation)
    out_stk = sr.transform_stack(stack_img, tmats=tmat_file)
    return np.float32(out_stk)


def align_stack_iter(
    stack,
    ref_stack_void=True,
    ref_stack=None,
    transformation=StackReg.TRANSLATION,
    method=("previous", "first"),
    max_iter=2,
):
    if ref_stack_void:
        ref_stack = stack

    for i in range(max_iter):
        sr = StackReg(transformation)
        for ii in range(len(method)):
            print(ii, method[ii])
            tmats = sr.register_stack(ref_stack, reference=method[ii])
            ref_stack = sr.transform_stack(ref_stack)
            stack = sr.transform_stack(stack, tmats=tmats)

    return np.float32(stack)

def normalize_and_scale(stack):
    return (stack**2/stack.sum(0)[np.newaxis,:,:])/stack.max()


def applyMaskGetMeanSpectrum(im_stack, mask):
    """A 2d mask to multiply with the 3d xanes stack and returns mean spectrum"""

    masked_stack = im_stack * mask
    return get_mean_spectra(masked_stack)


def modifyStack(
    raw_stack,
    normalizeStack=False,
    normToPoint=-1,
    applySmooth=False,
    smoothWindowSize=3,
    applyThreshold=False,
    thresholdValue=0,
    removeOutliers=False,
    nSigmaOutlier=3,
    applyTranspose=False,
    transposeVals=(0, 1, 2),
    applyCrop=False,
    cropVals=(0, 1, 2),
    removeEdges=False,
    resizeStack=False,
    upScaling=False,
    binFactor=2,
):

    """A giant function to modify the stack with many possible operations.
    all the changes can be saved to a jason file as a config file. Enabling and
    distabling the sliders is a problem"""

    """
    normStack = normalize(raw_stack, norm_point=normToPoint)
    smoothStack = smoothen(raw_stack, w_size= smoothWindowSize)
    thresholdStack = clean_stack(raw_stack, auto_bg=False, bg_percentage = thresholdValue)
    outlierStack = remove_hot_pixels(raw_stack, NSigma=nSigmaOutlier)
    transposeStack = np.transpose(raw_stack, transposeVals)
    croppedStack = raw_stack[cropVals]
    edgeStack = remove_edges(raw_stack)
    binnedStack = resize_stack(raw_stack,upscaling=upScaling,scaling_factor=binFactor)

    """

    if removeOutliers:
        modStack = remove_hot_pixels(raw_stack, NSigma=nSigmaOutlier)

    else:
        modStack = raw_stack

    if applyThreshold:
        modStack = clean_stack(modStack, auto_bg=False, bg_percentage=thresholdValue)

    else:
        pass

    if applySmooth:
        modStack = smoothen(modStack, w_size=smoothWindowSize)

    else:
        pass

    if applyTranspose:
        modStack = np.transpose(modStack, transposeVals)

    else:
        pass

    if applyCrop:
        modStack = modStack[cropVals]

    else:
        pass

    if normalizeStack:
        modStack = normalize(raw_stack, norm_point=normToPoint)
    else:
        pass
