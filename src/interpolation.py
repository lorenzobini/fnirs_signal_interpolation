from mne.io import Raw
from utils import channels_indices
from scipy.spatial.distance import pdist, squareform
from scipy import interpolate
import numpy as np
import warnings


def interpolate_channels(signal: Raw, bad_chs, method='nearest', exclude=()):
    """
    # Authors: Denis Engemann <denis.engemann@gmail.com>, Lorenzo Bini <lrnbini@gmail.com>
    # The original function was implemented as part of the MNE-Nirs library
    # The nearest-neighbor interpolation has been adapted from the original source code
    # The quadratic, cubic and bicubic interpolation have been implemented in the current version
    :param signal: the fNIRS recording
    :param method: the distance method that will be used
    :param bad_chs: the list of channels to interpolate
    :param exclude: the list of channels to exclude
    :return: the fNIRS recording with the bad channels interpolated
    """
    if len(bad_chs) == 0:
        ValueError("List of bad channels can not be empty.")

    # Retrieving the channel names and indices of the bad channels
    channel_names = signal.info['ch_names']
    channel_names = [ch for ch in channel_names if ch not in exclude]

    good_channels = channels_indices(signal, channel_names)
    bad_channels = channels_indices(signal, bad_chs)

    # Retrieving the 2D location of the channels on the template
    chs = [signal.info['chs'][i] for i in good_channels]
    locs2d_good = np.array([ch['loc'][:2] for ch in chs])

    # Initializing the distance function
    dist = pdist(locs2d_good)
    dist = squareform(dist)

    for i, bad in enumerate(bad_channels):
        dists_to_bad = dist[bad]
        # Ignore distances to self
        dists_to_bad[dists_to_bad == 0] = np.inf
        # Ignore distances to other bad channels
        dists_to_bad[bad_channels] = np.inf

        if method == 'nearest':
            signal = nearest_neighbour(signal, dists_to_bad, bad, channel_names, bad_chs[i])

        if method in ['linear', 'cubic', 'quadratic']:
            signal = univariate_interpolation(signal, bad_chs[i], bad, dists_to_bad, channel_names, method)

        if method in ['bilinear', 'bicubic', 'quintic']:
            locs2d_bad = signal.info['chs'][i]['loc'][:2]
            signal = bivariate_interpolation(signal, good_channels, bad_channels, locs2d_good, locs2d_bad, bad_chs[i])

        signal.info['bads'] = [ch for ch in signal.info['bads'] if ch in exclude]

    return signal


def nearest_neighbour(inst, dists_to_bad, bad, nirs_ch_names, bad_ch):
    """

    :param inst:
    :param dists_to_bad:
    :param bad:
    :param nirs_ch_names:
    :param bad_ch:
    :return:
    """
    # Find closest remaining channels for same frequency
    closest_idx = np.argmin(dists_to_bad) + (bad % 2)
    inst.apply_function(lambda a1: np.squeeze(inst.get_data(nirs_ch_names[closest_idx])),
                        bad_ch,
                        channel_wise=True)

    return inst


def univariate_interpolation(signal, bad_channel, bad_channel_id, dists_to_bad, channel_names,  method):
    """

    :param signal:
    :param bad_channel:
    :param bad_channel_id:
    :param dists_to_bad:
    :param channel_names:
    :param method:
    :return:
    """
    # Find closest remaining channels for same frequency
    closest_idx_1, _, closest_idx_2, _ = np.argpartition(dists_to_bad, 1)[0:4]
    closest_idx_1 += bad_channel_id % 2
    closest_idx_2 += bad_channel_id % 2

    # Sampling
    upper_limit = signal.n_times

    x = np.arange(0, upper_limit, np.floor(upper_limit * 0.02), dtype=int)
    x_new = np.arange(0, upper_limit, dtype=int)

    y = np.mean([np.squeeze(signal.get_data(channel_names[closest_idx_1]))[x],
                 np.squeeze(signal.get_data(channel_names[closest_idx_2]))[x]], axis=0)

    # interpolation step
    f1d = interpolate.interp1d(x, y, kind=method, fill_value="extrapolate")

    # y array that contains the interpolated data points
    y_interp = f1d(x_new)
    signal.apply_function(lambda a1: y_interp, bad_channel, channel_wise=True)

    return signal


def bivariate_interpolation(signal, good_channels, locs2d_good, locs2d_bad, bad_ch, method):
    """
    :param signal:
    :param good_channels:
    :param locs2d_good:
    :param locs2d_bad:
    :param bad_ch:
    :param linear:
    :return:
    """

    # Setting up the available the 2D coordinates for interpolation
    x = [x1 for x1, _ in locs2d_good]
    y = [y1 for _, y1 in locs2d_good]

    # Extracting the signal
    data = np.squeeze(signal.get_data(good_channels))

    # Setting up the complete list of coordinates with the channels to interpolate
    x_new = x.copy()
    x_new.append(locs2d_bad[0])
    y_new = y.copy()
    y_new.append(locs2d_bad[1])

    z_interp = []

    indices_to_correct = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k in range(len(data[1])):  # Length of signal
            z = data[:, k]
            # Performing interpolation
            f2d = interpolate.interp2d(x, y, z, kind=method)
            interp_value = f2d(locs2d_bad[0], locs2d_bad[1])[0]

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
    signal.apply_function(lambda a1: np.array(z_interp), bad_ch, channel_wise=True)

    return signal

