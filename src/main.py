from utils import *
import mne
import mne_nirs
from interpolation import interpolate_channels


files = pick_files()
recording = load_recording(files[0])

# ## NORMALIZING DATA
recording = recording.copy().apply_function(fun=normalize,
                                            picks=recording.info.ch_names,
                                            channel_wise=True)

# ## RETRIEVING LIST OF CHANNELS TO INTERPOLATE
chs_to_interpolate = read_channels_to_interpolate()
chs_to_interpolate = append_hbohbr(chs_to_interpolate)

# ## INTERPOLATING
methods = ["nearest", "linear", "quadratic", "cubic", "bilinear", "bicubic", "quintic"]
for method in methods:
    interp_recording = interpolate_channels(recording.copy(), chs_to_interpolate, method=method)