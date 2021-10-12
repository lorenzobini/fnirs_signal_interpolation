from utils import *
import mne
import mne_nirs
from interpolation import interpolate_channels


def main():
    global filepath
    files = pick_files()
    for file in files:
        recording = load_recording(file)

        # ## RETRIEVING LIST OF CHANNELS TO EXCLUDE
        chs_to_exclude = read_channels_list(filepath, filename="channels_to_exclude.txt")

        if chs_to_exclude:
            chs_to_exclude = append_hbohbr(chs_to_exclude)
            recording = discard_channels(recording, chs_to_exclude)

        # ## NORMALIZING DATA
        recording = recording.copy().apply_function(fun=normalize,
                                                    picks=recording.info.ch_names,
                                                    channel_wise=True)

        # ## RETRIEVING LIST OF CHANNELS TO INTERPOLATE
        chs_to_interpolate = read_channels_list(filepath, filename="channels_to_interpolate.txt")
        chs_to_interpolate = append_hbohbr(chs_to_interpolate)


        # ## INTERPOLATING
        methods = ["nearest", "linear", "quadratic", "cubic", "bilinear", "bicubic", "quintic"]
        for method in methods:
            interp_recording = interpolate_channels(recording.copy(), chs_to_interpolate, method=method)


if __name__ == "__main__":
    global filepath
    os.makedirs("data", exist_ok=True)
    filepath = os.path.join(os.getcwd(), 'data') + os.sep

    if os.path.isfile(filepath + "channels_to_interpolate.txt"):
        if os.stat("file").st_size == 0:
            ImportError("The file channels_to_interpolate.txt is empty. Please specify a list of channels to interpolate.")
    else:
        f = open(os.path.isfile(filepath + "channels_to_interpolate"), "w+")
        f.close()
        ImportError("The file channels_to_interpolate.txt did not exist. An empty file has been created. Please specify a list of channels to interpolate.")

    if os.path.isfile(filepath + "channels_to_exclude.txt"):
        if os.stat("file").st_size == 0:
            ImportWarning("The file channels_to_exclude.txt is empty. Please specify eventual channels to ignore to avoid errors.")
    else:
        f = open(os.path.isfile(filepath + "channels_to_exclude"), "w+")
        f.close()
        ImportError("The file channels_to_exclude.txt did not exist. An empty file has been created. Please specify eventual channels to ignore to avoid errors.")