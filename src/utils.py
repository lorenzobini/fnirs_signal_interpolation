from mne.io import Raw, read_raw_fif
import os
from tkinter import *
from tkinter.filedialog import askopenfilenames


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

def save_mne_recording(rec: Raw, filename: str, path: str, subfolder: str = None):
    """
    Save a given Tensor into an image file.
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