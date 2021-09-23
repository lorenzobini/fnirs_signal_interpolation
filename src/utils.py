from mne.io import Raw


def channels_indices(rec: Raw, chs: [str]):
    """
    Determines the indices of the given channels within the recording object
    :param rec:
    :param chs:
    :return:
    """
    chs_indices = []
    rec_df = rec.copy().to_data_frame()
    for ch in chs:
        if ch not in rec.ch_names:
            RuntimeWarning(f"Could not determine the index of channel {ch} in recording.")
        # Determining the patch channels' index
        ch_index = rec_df.columns.get_loc(ch) - 1
        chs_indices.append(ch_index)

    return chs_indices