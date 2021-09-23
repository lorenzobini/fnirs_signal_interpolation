from mne.io import Raw
from utils import channels_indices
from scipy.spatial.distance import pdist, squareform
from scipy import interpolate
import numpy as np
import warnings


def interpolate_channels(inst: Raw, bad_chs, method='nearest', exclude=()):
    """
    # Authors: Denis Engemann <denis.engemann@gmail.com>, Lorenzo Bini <lrnbini@gmail.com>
    # The original function was implemented as part of the MNE-Nirs library
    # The nearest-neighbor interpolation has been adapted from the original source code
    # The quadratic, cubic and bicubic interpolation have been implemented in the current version
    :param inst: the fNIRS recording
    :param method: the distance method that will be used
    :param bad_chs: the list of channels to interpolate
    :param exclude: the list of channels to exclude
    :return: the fNIRS recording with the bad channels interpolated
    """
    if len(bad_chs) == 0:
        ValueError("List of bad channels can not be empty.")

    # Retrieving the channel names and indices of the bad channels
    nirs_ch_names = inst.info['ch_names']
    nirs_ch_names = [ch for ch in nirs_ch_names if ch not in exclude]

    picks_nirs = channels_indices(inst, nirs_ch_names)
    picks_bad = channels_indices(inst, bad_chs)

    # Retrieving the 2D location of the channels on the template
    chs = [inst.info['chs'][i] for i in picks_nirs]
    locs2d = np.array([ch['loc'][:2] for ch in chs])

    # Initializing the distance function
    dist = pdist(locs2d)
    dist = squareform(dist)

    for i, bad in enumerate(picks_bad):
        dists_to_bad = dist[bad]
        # Ignore distances to self
        dists_to_bad[dists_to_bad == 0] = np.inf
        # Ignore distances to other bad channels
        dists_to_bad[picks_bad] = np.inf

        if method == 'nearest':
            inst = nearest_neighbour(inst, dists_to_bad, bad, nirs_ch_names, bad_chs[i])

        if method == 'cubic' or method == 'quadratic':
            inst = univariate_interpolation(inst, bad, dists_to_bad, nirs_ch_names, bad_chs[i], method)

        if method == 'bicubic':
            inst = bicubic_interpolation(inst, picks_nirs, picks_bad, bad_chs[i])

        inst.info['bads'] = [ch for ch in inst.info['bads'] if ch in exclude]

    return inst


def nearest_neighbour(inst, dists_to_bad, bad, nirs_ch_names, bad_ch):
    """

    :return:
    """
    # Find closest remaining channels for same frequency
    closest_idx = np.argmin(dists_to_bad) + (bad % 2)
    inst.apply_function(lambda a1: np.squeeze(inst.get_data(nirs_ch_names[closest_idx])),
                        bad_ch,
                        channel_wise=True)

    return inst


def univariate_interpolation(inst, bad, dists_to_bad, nirs_ch_names, bad_ch, method):
    """

    :return:
    """
    # Find closest remaining channels for same frequency
    closest_idx_1, _, closest_idx_2, _ = np.argpartition(dists_to_bad, 1)[0:4]
    closest_idx_1 += bad % 2
    closest_idx_2 += bad % 2

    # Sampling
    upper_limit = inst.n_times

    x = np.arange(0, upper_limit, np.floor(upper_limit * 0.02), dtype=int)
    x_new = np.arange(0, upper_limit, dtype=int)

    y = np.mean([np.squeeze(inst.get_data(nirs_ch_names[closest_idx_1]))[x],
                 np.squeeze(inst.get_data(nirs_ch_names[closest_idx_2]))[x]], axis=0)

    # interpolation step
    f1d = interpolate.interp1d(x, y, kind=method, fill_value="extrapolate")

    # y array that contains the interpolated data points
    y_interp = f1d(x_new)
    inst.apply_function(lambda a1: y_interp, bad_ch, channel_wise=True)

    return inst


def bicubic_interpolation(inst, picks_nirs, picks_bad, bad_ch):
    """

    :return:
    """
    # Retrieving the list of good channels
    picks_good = [pick for pick in picks_nirs if pick not in picks_bad]
    chs_good = [inst.info['chs'][k] for k in picks_good]

    # Retrieving 2D location for good and bad channels
    locs2d_good = np.array([ch['loc'][:2] for ch in chs_good])
    loc2d_bad = inst.info['chs'][i]['loc'][:2]

    # Setting up the available the 2D coordinates for interpolation
    x = [x1 for x1, _ in locs2d_good]
    y = [y1 for _, y1 in locs2d_good]
    data = np.squeeze(inst.get_data(picks_good))

    # Setting up the complete list of coordinates with the channels to interpolate
    x_new = x.copy()
    x_new.append(loc2d_bad[0])
    y_new = y.copy()
    y_new.append(loc2d_bad[1])

    z_interp = []

    indices_to_correct = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k in range(len(data[1])):  # Length of signal
            z = data[:, k]
            # Performing interpolation
            f2d = interpolate.interp2d(x, y, z, kind='cubic')
            interp_value = f2d(loc2d_bad[0], loc2d_bad[1])[0]

            if interp_value > 1.0 or interp_value < -1.0:
                # Saving the points that produce sharp peaks
                indices_to_correct.append(len(z_interp))

            z_interp.append(interp_value)

    # Correcting the peaks
    for index in indices_to_correct:
        try:
            # The peak is corrected by computing the mean of the two adjacent values
            z_interp[index] = np.mean(z_interp[index - 1], z_interp[index + 1])
        except:
            # For peaks located at the extremities, the value is copy-pasted from the nearest
            try:
                z_interp[index] = z_interp[index - 1]
            except:
                z_interp[index] = z_interp[index + 1]

        if z_interp[index] > 1.0:
            z_interp[index] = 1.0
        elif z_interp[index] < -1.0:
            z_interp[index] = -1.0

    # Overwriting the signal with the interpolated channels
    inst.apply_function(lambda a1: np.array(z_interp), bad_ch, channel_wise=True)

    return inst

