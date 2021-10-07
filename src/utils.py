from mne.io import Raw, read_raw_fif
import os
from tkinter import *
from tkinter.filedialog import askopenfilenames
import numpy as np


def pick_files():
    """
    It opens a file picker window and lets the user choose a .MAT, .OXY3 or .OXY4 file to import.
    :return dat: loaded .mat file
    :raises FileNotFoundError
    :raises IOError
    """
    root = Tk()
    filenames = askopenfilenames(initialdir=os.path.join(os.getcwd(), 'data') + os.sep,
                                 filetypes=(("MAT files", "*.mat"),
                                            ("OXY3 files", "*.oxy3"),
                                            ("OXY4 files", "*.oxy4"),
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
    :param rec:
    :param filename
    :param path:
    :param subfolder:
    """
    if subfolder is None:
        rec.save(fname=f"{path}{filename}_raw.fif", overwrite=True, verbose=False)
    else:
        os.makedirs(os.path.join(path, subfolder), exist_ok=True)
        rec.save(fname=f"{os.path.join(path, subfolder)}\\{filename}_raw.fif", overwrite=True, verbose=False)


def read_channels_to_interpolate(path=c.CONFIG_PATH, filename="channels_to_interpolate.txt"):
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
