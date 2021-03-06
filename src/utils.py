from mne.io import Raw, read_raw_fif
import os
from tkinter import *
from tkinter.filedialog import askopenfilenames
import numpy as np
import re
import itertools
import random
from collections import defaultdict


def pick_files():
    """
    It opens a file picker window and lets the user choose a .FIF file to import.

    :return dat: loaded .FIF file

    :raises FileNotFoundError
    :raises IOError
    """
    root = Tk()
    filenames = askopenfilenames(initialdir=os.path.join(os.getcwd(), 'data') + os.sep,
                                 filetypes=(("FIF files", "*.fif"),
                                            ("All Files", "*.*")),
                                 title="Choose a file to import.")
    root.destroy()

    return filenames


def load_recording(filename, path: str, subfolder: str = None, load_data: bool = False):
    """
    Loads the MNE raw signal object from a specified .fif file without loading the data, unless specified.
    To load the data at a later time make use of the load_data() built-in function from MNE.

    :param filename: the name of the .fif file to load
    :param path: the directory where the file is located
    :param subfolder: the subfolder, can be also inputted via the path
    :param load_data: if True loads the whole recording in memory, requires more RAM

    :return: the loaded recording in Raw format
    """

    rec = None

    if subfolder is None:
        filepath = f"{path}\\{filename}"
    else:
        filepath = f"{os.path.join(path, subfolder)}\\{filename}"

    try:
        rec = read_raw_fif(fname=filepath, verbose=False)
    except FileNotFoundError:
        ImportWarning(f"Could not import file {filename} because file does not exists in the specified directory.")
    except NotADirectoryError:
        ImportWarning(f"Could not import file {filename} because the directory does not exist.")
    except:
        ImportWarning(f"Could not import file {filename}. Unknown error.")

    if load_data:
        rec.load_data()

    return rec


def save_recording(rec: Raw, filename: str, path: str, subfolder: str = None):
    """
    Save a given Raw file to disk.

    :param rec: the fNIRS recording
    :param filename: the name to assign to the saved file
    :param path: the directory in which to save the recording
    :param subfolder: subfolder to append to the path (optional)
    """
    if subfolder is None:
        rec.save(fname=f"{path}{filename}_raw.fif", overwrite=True, verbose=False)
    else:
        os.makedirs(os.path.join(path, subfolder), exist_ok=True)
        rec.save(fname=f"{os.path.join(path, subfolder)}\\{filename}_raw.fif", overwrite=True, verbose=False)


def read_channels_list(path, filename="channels_to_interpolate.txt"):
    """
    Reads list of channels to interpolate from file.

    :param path: the path leading to the file
    :param filename: the file name

    :return: the list of channels to interpolate
    """
    channels = []
    if os.path.isfile(path + filename):
        with open(path + filename) as f:
            lines = f.readlines()
            for line in lines:
                if re.match(r'(S\d+[ab-d]?_D\d+[ab-d]?)', line):
                    ch = re.match(r'(S\d+[ab-d]?_D\d+[ab-d]?)\w+', line).group(0)
                    channels.append(ch + " 762")
                    channels.append(ch + " 845")
                else:
                    UserWarning(f"Channel {line} ignored due to incorrect formatting.")
    else:
        ImportWarning(f"File {filename} not found in {path}.")

    return channels


def normalize(signal):
    """
    Normalizes the signal to a normal distribution of mean 0 and standard deviation 1

    :param signal: the time series of a channel

    :return: the normalised fNIRS recording
    """
    if max(signal) == min(signal):
        normalized = 2 * (signal - min(signal)) - 1
    else:
        normalized = 2 * (signal - min(signal)) / (max(signal) - min(signal)) - 1

    if np.isnan(normalized).any():
        ValueError("Signal value cannot be NaN.")

    return normalized


def append_hbohbr(chs, values=False):
    """
    Creates a list of channels with the HBO/HBR notation from a list of channels without HBO/HBR notation.

    :param chs: the list of channels without notation

    :return: the list of channels with HBO/HBR notation
    """
    ch_names_hbohbr = []
    for ch_name in chs:
        if values:
            # Re-introducing hbo/hbr notations in values
            ch_names_hbohbr.append(ch_name + " 762")
            ch_names_hbohbr.append(ch_name + " 845")
        else:
            # Re-introducing hbo/hbr notations
            ch_names_hbohbr.append(ch_name + " hbo")
            ch_names_hbohbr.append(ch_name + " hbr")

    return ch_names_hbohbr


def channels_indices(rec: Raw, chs: [str]):
    """
    Determines the indices of the given channels within the recording object

    :param rec: the fNIRS recording
    :param chs: the list of channels to locate

    :return: the list of indices associated to the provided channels
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


def discard_channels(rec: Raw, exc_chs: [str]):
    """
    Discards the received channels from the set of recordings.

    :param rec: the fNIRS recording
    :param exc_chs: the list of channels to exclude

    :return: the recording minus the discarded channels
    """
    # Checking if all received channels are actually present in the recording
    channels = rec.info.ch_names
    mismatches = np.setdiff1d(exc_chs, channels)
    # Removing possible channels that are not present in the recording
    for mismatch in mismatches: exc_chs.remove(mismatch)

    # Removing channels from recordings
    try:
        rec.drop_channels(exc_chs)
    except:
        UserWarning("Function drop_channels was unsuccessful.")

    return rec


def sample_channels_to_interpolate(rec: Raw):
    """
    Given a high-resolution recording, returns a list of random channels to remove. The function applies the
    rule of one channel per transmitter.

    :param rec: the high resolution recording

    :return: the list of channels to remove
    """
    channels = rec.info.ch_names
    bad_tr, bad_chs, bad_reps = isolate_bads(rec)

    chs = defaultdict(lambda: None)
    for ch in channels:
        try:
            # Obtaining the channel's transmitter
            tr = re.search(r"(S\d+[ab-d]?)", ch).group(0)
            # Obtaining channels associated with each transmitter
            ch_name = re.search(r"(S\d+[ab-d]?_D\d+[ab-d]?)", ch).group(0)

            if tr in bad_tr:
                continue
            if chs[tr] is None:
                # Creating first element in the list of channels linked to the transmitter
                chs[tr] = [ch_name]
            elif ch_name not in chs[tr]:
                # Populating the list of channels linked to the transmitter
                chs[tr].append(ch_name)
        except AttributeError:
            UserWarning(f"Channel name {ch} is not formatted correctly.")

    # Converting dict to list
    ch_list = []
    for _, value in chs.items():
        ch_list.append(list(value))

    # Randomly removing as many transmitters as the number of extra-bad-channels-per-transmitter has been found
    # This prevents the removal of too many channels when having several bad channels in the recording.
    for i in range(0, bad_reps):
        random_item_from_list = random.choice(ch_list)
        ch_list.remove(random_item_from_list)

    # Selecting all possible combinations of channels
    ch_list = list(itertools.product(*ch_list))

    # Including the bad channels as channels to remove for every LR recording
    bad_chs = tuple(bad_chs)
    ch_list = [i + bad_chs for i in ch_list]

    ch_list = list(random.choices(ch_list, k=1))
    ch_list = [list(i) for i in ch_list][0]

    return ch_list


def isolate_bads(rec):
    """
    Isolates the list of bad channels and respective transmitters
    :param rec: the fNIRS recording
    :return: the list of bad channels
    :return: the list of transmitters associated to bad channels
    :return: the number of bad channels that fall outside the 1-channel-per-transmitter rule, which indicates how many
             "good" transmitters to ignore from the computation to produce a 26-channels LR recording
    """
    channels = rec.info['bads']

    chs = defaultdict(lambda: None)
    for ch in channels:
        try:
            # Obtaining the bad channel's transmitter
            tr = re.search(r"(S\d+[ab-d]?)", ch).group(0)
            # Obtaining bad channels associated with each transmitter
            ch_name = re.search(r"(S\d+[ab-d]?_D\d+[ab-d]?)", ch).group(0)

            if chs[tr] is None:
                # Creating first element in the list of bad channels linked to the transmitter
                chs[tr] = [ch_name]
            elif ch_name not in chs[tr]:
                # Populating the list of bad channels linked to the transmitter
                chs[tr].append(ch_name)
        except AttributeError:
            UserWarning(f"Channel name {ch} is not formatted correctly.")

    ch_list = []  # The list of bad channels
    tr_list = []  # The list of transmitters associated to bad channels
    reps = 0      # The number of bad channels that fall outside the 1-channel-per-transmitter rule
    for transmitter, channels in chs.items():
        tr_list.append(transmitter)
        ch_list.extend(channels)
        reps += len(channels) - 1

    return tr_list, ch_list, reps
