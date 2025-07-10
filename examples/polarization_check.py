import xarray as xr
import numpy as np
import os
import sys
import matplotlib.pyplot as plt


# Add the main_folder to sys.path no matter where the script is run from to be able to get to dataset_utils.py
current_file = os.path.abspath(__file__)  # Absolute path to this script
main_folder = os.path.dirname(os.path.dirname(current_file))  # Go up two levels
sys.path.insert(0, main_folder)  # Insert main_folder at the start of sys.path

from src.dataset_utils import get_channel_by_stripe_ru, channel_to_numpycomplex64

if __name__ == '__main__':
    # path to the dataset directory
    nr_ue_locations = 5681
    basepath = f"/home/user/6GTandem_RT_server/dataset_{nr_ue_locations}_ue_locations"

    """------------------------------ interacting with the sub THz channels dataset example ------------------------------"""
    print(f'----------------------------- sub THz channels dataset example -----------------------------')

    # load user locations dataset
    ue_ds = xr.load_dataset(os.path.join(basepath, "ue_locations", f"ue_locations_{nr_ue_locations}.nc"))
    
    # select user under a RU
    # select users based if they are on the grid or not (The grid contains all locations where there are RUs)
    on_grid_users = ue_ds.where(ue_ds['ue_on_stripe_grid'], drop=True)
    off_grid_users = ue_ds.where(~ue_ds['ue_on_stripe_grid'], drop=True)

    print(f"Users on grid: {on_grid_users.sizes['user']}")
    print(f"Users off grid: {off_grid_users.sizes['user']}")

    # get user under stripe idx and ru idx
    ue_stripe_idx = 0
    ue_ru_idx = 0
    matched_user = ue_ds.where(
        (ue_ds['ue_on_stripe_grid']) &
        (ue_ds['ue_stripe_idx'] == ue_stripe_idx) &
        (ue_ds['ue_ru_idx'] == ue_ru_idx),
        drop=True
    )
    

    # let's load the sub THz channels dataset for the matched user
    ue_id = matched_user['user_id'].values.item()  # get user id
    ds_sub_thz = xr.load_dataset(os.path.join(basepath, "sub_thz_channels", f"channels_thz_ue_{int(ue_id)}.nc"))
    print(ds_sub_thz)



    # access channel between the UE and a specific RU at stripe_idx and RU_idx
    stripe_idx, RU_idx = ue_stripe_idx, ue_ru_idx
    channel = get_channel_by_stripe_ru(ds_sub_thz, stripe_idx=stripe_idx, ru_idx=RU_idx)
    print(channel.shape)
    channel_np = channel_to_numpycomplex64(channel.values)
    channel_q = channel_np[:, :, 0] # get channel at one subcarrier
    
    # Compute the magnitude
    magnitude = np.abs(channel_q)
    print(f'first quadrant: {magnitude[0:4, 0:4]}')
    print(f'second quadrant: {magnitude[0:4, 4:8]}')
    print(f'third quadrant: {magnitude[4:8, 0:4]}')
    print(f'fourth quadrant: {magnitude[4:8, 4:8]}')
    

    # Plot the magnitude as a heatmap
    plt.figure(figsize=(6, 5))
    plt.imshow(magnitude, cmap='viridis', interpolation='nearest')
    plt.title('Magnitude of channel_q')
    plt.colorbar(label='Magnitude')
    plt.xlabel('Receiver Antenna Index')
    plt.ylabel('Transmitter Antenna Index')
    plt.grid(False)
    # Save to file
    plt.savefig('channel_magnitude_subthz.png')  # You can open this file in VS Code or download it
    plt.close()


    """------------------------------ interacting with the sub 10 GHz channels dataset example ------------------------------"""
    print(f'----------------------------- sub 10 GHz channels dataset example -----------------------------')

    # let's load the sub 10 GHz channels dataset for the matched user
    ds_sub10 = xr.load_dataset(os.path.join(basepath, "sub_10ghz_channels", f"channels_sub10ghz_ue_{int(ue_id)}.nc"))
    print(ds_sub10)
    # print all AP names
    ap_names = ds_sub10["ap"].values
    print("AP names:", ap_names)

    # select channel between the UE and a specific AP 'AP_wall_1'
    channel_sub10 = ds_sub10["channel"].sel(ap=ap_names[1])  # select 3 AP
    print(f"Channel shape for AP {ap_names[2]}: {channel_sub10.shape}")  # (rx_ant, tx_ant, subcarrier)
    print(f'shape meaning: {channel_sub10.dims} note that the rx_ant are at the ue side and tx_ant are at the AP side')
    
    channel_sub10 = channel_to_numpycomplex64(channel_sub10.values)  # convert to numpy complex64
    magnitude_sub10 = np.abs(channel_sub10[:, :, 0])  # get channel at one subcarrier
    # Plot the magnitude as a heatmap
    plt.figure(figsize=(6, 5))
    plt.imshow(magnitude_sub10, cmap='viridis', interpolation='nearest')
    plt.title('Magnitude of channel_q')
    plt.colorbar(label='Magnitude')
    plt.xlabel('Receiver Antenna Index')
    plt.ylabel('Transmitter Antenna Index')
    plt.grid(False)
    # Save to file
    plt.savefig('channel_magnitude_sub10_ceilpannel.png')  # You can open this file in VS Code or download it
    plt.close()
    #print(f"Channel data for AP {ap_names[2]}:\n", channel_sub10)

    # select channel between the UE and a specific AP 'AP_wall_1'
    channel_sub10 = ds_sub10["channel"].sel(ap=ap_names[2])  # select 3 AP
    print(f"Channel shape for AP {ap_names[2]}: {channel_sub10.shape}")  # (rx_ant, tx_ant, subcarrier)
    print(f'shape meaning: {channel_sub10.dims} note that the rx_ant are at the ue side and tx_ant are at the AP side')
    
    channel_sub10 = channel_to_numpycomplex64(channel_sub10.values)  # convert to numpy complex64
    magnitude_sub10 = np.abs(channel_sub10[:, :, 0])  # get channel at one subcarrier
    # Plot the magnitude as a heatmap
    plt.figure(figsize=(6, 5))
    plt.imshow(magnitude_sub10, cmap='viridis', interpolation='nearest')
    plt.title('Magnitude of channel_q')
    plt.colorbar(label='Magnitude')
    plt.xlabel('Receiver Antenna Index')
    plt.ylabel('Transmitter Antenna Index')
    plt.grid(False)
    # Save to file
    plt.savefig('channel_magnitude_sub10_wallpannel.png')  # You can open this file in VS Code or download it
    plt.close()
    #print(f"Channel data for AP {ap_names[2]}:\n", channel_sub10)


    # extract copolarized subchannels VV
    H_vv = channel_sub10[0:4, 0:4, 0]
    mag_Hvv = np.abs(H_vv)
    # Plot the magnitude as a heatmap
    plt.figure(figsize=(6, 5))
    plt.imshow(mag_Hvv, cmap='viridis', interpolation='nearest')
    plt.title('Magnitude of channel_q')
    plt.colorbar(label='Magnitude')
    plt.xlabel('Receiver Antenna Index')
    plt.ylabel('Transmitter Antenna Index')
    plt.grid(False)
    # Save to file
    plt.savefig('channel_magnitude_sub10_wallpannel_Hvv.png')  # You can open this file in VS Code or download it
    plt.close()

