import xarray as xr
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from utils import create_folder

# Add the main_folder to sys.path no matter where the script is run from to be able to get to dataset_utils.py
current_file = os.path.abspath(__file__)  # Absolute path to this script
main_folder = os.path.dirname(os.path.dirname(current_file))  # Go up two levels
sys.path.insert(0, main_folder)  # Insert main_folder at the start of sys.path

from src.dataset_utils import get_channel_by_stripe_ru, channel_to_numpycomplex64

if __name__ == '__main__':
    new_dataset_location = "/home/user/6GTandem_RT_server/dataset_5681_ue_locations_reduced_polarization"
    
    # set output subTHz path
    sub_thzchannel_output_path = os.path.join(new_dataset_location, 'sub_thz_channels')
    create_folder(sub_thzchannel_output_path)

    # set output sub10 path
    sub_10channel_output_path = os.path.join(new_dataset_location, 'sub_10ghz_channels')
    create_folder(sub_10channel_output_path)

    # path to the dataset directory
    nr_ue_locations = 5681
    basepath = f"/home/user/6GTandem_RT_server/dataset_{nr_ue_locations}_ue_locations"

    for ue_id in range(nr_ue_locations):
        sub_thz_out_file = os.path.join(sub_thzchannel_output_path, f"channels_thz_ue_{ue_id}.nc")
        sub10_out_file = os.path.join(sub_10channel_output_path, f"channels_sub10ghz_ue_{ue_id}.nc")


        # let's load the sub THz channels dataset for the matched user
        ds_sub_thz = xr.load_dataset(os.path.join(basepath, "sub_thz_channels", f"channels_thz_ue_{int(ue_id)}.nc"))
        channel_subthz = ds_sub_thz["channel"]
        channel_subthz = channel_to_numpycomplex64(channel_subthz.values)  # convert to numpy complex64
        channel_subthz_1_pol = channel_subthz[:, 0:4, 4:8, :] # keep only matched polarization

        # pack channel in a xarray dataset
        ds_sub_thz_reduced = xr.Dataset(
            data_vars={
                "channel": (
                    ("tx_pair", "rx_ant", "tx_ant", "subcarrier"),
                    channel_subthz_1_pol
                )
            },
            coords={
                "tx_pair": ds_sub_thz["tx_pair"].values,
                "stripe_idx": ("tx_pair", ds_sub_thz["stripe_idx"].values),
                "RU_idx": ("tx_pair", ds_sub_thz["RU_idx"].values),
                "rx_ant": np.arange(channel_subthz_1_pol.shape[1]),  # 0..3
                "tx_ant": np.arange(channel_subthz_1_pol.shape[2]),  # 0..3
                "subcarrier": ds_sub_thz["subcarrier"].values,
            },
            attrs=ds_sub_thz.attrs
        )
        ds_sub_thz_reduced.to_netcdf(sub_thz_out_file, format="NETCDF4", auto_complex=True)


        # let's load the sub 10 GHz channels dataset for the matched user
        ds_sub10 = xr.load_dataset(os.path.join(basepath, "sub_10ghz_channels", f"channels_sub10ghz_ue_{int(ue_id)}.nc"))
        channel_sub10 = ds_sub10["channel"]
        channel_sub10 = channel_to_numpycomplex64(channel_sub10.values)  # convert to numpy complex64
        channel_sub10_1_pol = channel_sub10[:, 0:4, 4:8, :] # keep only matched polarization

        ds_sub10_reduced = xr.Dataset(
            data_vars={
                "channel": (
                    ("ap", "rx_ant", "tx_ant", "subcarrier"),
                    channel_sub10_1_pol
                )
            },
            coords={
                "ap": ds_sub10["ap"].values,
                "rx_ant": np.arange(channel_sub10_1_pol.shape[1]),  # 0..3
                "tx_ant": np.arange(channel_sub10_1_pol.shape[2]),  # 0..3
                "subcarrier": ds_sub10["subcarrier"].values,
            },
            attrs=ds_sub10.attrs
        )

        ds_sub10_reduced.to_netcdf(sub10_out_file, format="NETCDF4", auto_complex=True)
        print(f'UE idx {ue_id} processed, {nr_ue_locations - ue_id - 1} to go!')
