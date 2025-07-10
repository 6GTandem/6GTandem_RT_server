import xarray as xr
import numpy as np
import os
import sys

# Add the main_folder to sys.path no matter where the script is run from to be able to get to dataset_utils.py
current_file = os.path.abspath(__file__)  # Absolute path to this script
main_folder = os.path.dirname(os.path.dirname(current_file))  # Go up two levels
sys.path.insert(0, main_folder)  # Insert main_folder at the start of sys.path

from src.dataset_utils import get_channel_by_stripe_ru, channel_to_numpycomplex64

if __name__ == '__main__':
    # path to the dataset directory
    nr_ue_locations = 5681
    basepath = f"/home/user/6GTandem_RT_server/dataset_{nr_ue_locations}_ue_locations"

    """------------------------------ interacting with the locations dataset example ------------------------------"""
    print(f'----------------------------- locations dataset example -----------------------------')
    # load user locations dataset
    ue_ds = xr.load_dataset(os.path.join(basepath, "ue_locations", f"ue_locations_{nr_ue_locations}.nc"))
    
    # Print dataset overview
    print(ue_ds)

    # View all available zones where there are users
    print("Available zones:", set(ue_ds['zone'].values))

    # --- Select users by zone ---
    zone1_users = ue_ds.where(ue_ds['zone'] == 'Zone 1', drop=True)
    zone2_users = ue_ds.where(ue_ds['zone'] == 'Zone 2', drop=True)
    zone3_users = ue_ds.where(ue_ds['zone'] == 'Zone 3', drop=True)
    zone4_users = ue_ds.where(ue_ds['zone'] == 'Zone 4', drop=True)
    grid_users  = ue_ds.where(ue_ds['zone'] == 'Grid', drop=True)

    # Print number of users per zone
    print(f"Zone 1 users: {zone1_users.sizes['user']}")
    print(f"Zone 2 users: {zone2_users.sizes['user']}")
    print(f"Zone 3 users: {zone3_users.sizes['user']}")
    print(f"Zone 4 users: {zone4_users.sizes['user']}")
    print(f"Grid users:  {grid_users.sizes['user']}")

    # As xarray variables
    x_coords = zone1_users['x']
    y_coords = zone1_users['y']
    z_coords = zone1_users['z']

    # Or convert to NumPy for further processing
    positions_zone1 = zone1_users[['x', 'y', 'z']].to_array().values.T  # shape: (num_users, 3)

    # select users based if they are on the grid or not (The grid contains all locations where there are RUs)
    on_grid_users = ue_ds.where(ue_ds['ue_on_stripe_grid'], drop=True)
    off_grid_users = ue_ds.where(~ue_ds['ue_on_stripe_grid'], drop=True)

    print(f"Users on grid: {on_grid_users.sizes['user']}")
    print(f"Users off grid: {off_grid_users.sizes['user']}")

    # get user under stripe idx and ru idx
    ue_stripe_idx = 2
    ue_ru_idx = 10
    matched_user = ue_ds.where(
        (ue_ds['ue_on_stripe_grid']) &
        (ue_ds['ue_stripe_idx'] == ue_stripe_idx) &
        (ue_ds['ue_ru_idx'] == ue_ru_idx),
        drop=True
    )

    print(f"User at stripe: {ue_stripe_idx} and RU: {ue_ru_idx}: x={matched_user['x'].values}, "
            f"y={matched_user['y'].values}, "
            f"z={matched_user['z'].values}")
    
    ue_id = matched_user['user_id'].values.item()  # get user id
    print(f"User ID: {ue_id}")

    """------------------------------ interacting with the sub THz channels dataset example ------------------------------"""
    print(f'----------------------------- sub THz channels dataset example -----------------------------')

    # let's load the sub THz channels dataset for the matched user
    ds_sub_thz = xr.load_dataset(os.path.join(basepath, "sub_thz_channels", f"channels_thz_ue_{int(ue_id)}.nc"))
    print(ds_sub_thz)

    # print some meta data
    print(f"User {ds_sub_thz.attrs['user_idx']} is in zone: {ds_sub_thz.attrs['zone']}")
    print(f"Location: x={ds_sub_thz.attrs['user_x']}, y={ds_sub_thz.attrs['user_y']}, z={ds_sub_thz.attrs['user_z']}")

    # access channel between the UE and a specific RU at stripe_idx and RU_idx
    stripe_idx, RU_idx = 5, 15
    channel = get_channel_by_stripe_ru(ds_sub_thz, stripe_idx=stripe_idx, ru_idx=RU_idx)
    if channel is not None:
        print(f"channels between UE {ue_id} at location: x={ds_sub_thz.attrs['user_x']}, y={ds_sub_thz.attrs['user_y']}, z={ds_sub_thz.attrs['user_z']} and RU {RU_idx} at stripe {stripe_idx}:")
        print(channel.shape)  # (rx_ant, tx_ant, subcarrier)
        print(channel.values.dtype)
        channel_np = channel_to_numpycomplex64(channel.values) # convert to numpy complex64
        print(channel_np.shape)  # (rx_ant, tx_ant, subcarrier)
        print(f'shape meaning: {channel.dims} note that the rx_ant are at the ue side and tx_ant are at the RU side')
        print(channel_np.dtype)  # complex64
        #print(channel_np)  # complex64 array
    else:
        print(f"invalide stripe or RU idx: {stripe_idx}, {RU_idx}")

    

    """------------------------------ interacting with the sub 10 GHz channels dataset example ------------------------------"""
    print(f'----------------------------- sub 10 GHz channels dataset example -----------------------------')

    # let's load the sub 10 GHz channels dataset for the matched user
    ds_sub10 = xr.load_dataset(os.path.join(basepath, "sub_10ghz_channels", f"channels_sub10ghz_ue_{int(ue_id)}.nc"))
    print(ds_sub10)
    # print all AP names
    ap_names = ds_sub10["ap"].values
    print("AP names:", ap_names)

    # select channel between the UE and a specific AP 'AP_wall_1'
    channel_sub10 = ds_sub10["channel"].sel(ap=ap_names[2])  # select 3 AP
    print(f"Channel shape for AP {ap_names[2]}: {channel_sub10.shape}")  # (rx_ant, tx_ant, subcarrier)
    print(f'shape meaning: {channel_sub10.dims} note that the rx_ant are at the ue side and tx_ant are at the AP side')
    
    channel_sub10 = channel_to_numpycomplex64(channel_sub10.values)  # convert to numpy complex64
    #print(f"Channel data for AP {ap_names[2]}:\n", channel_sub10)