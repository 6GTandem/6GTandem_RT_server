import xarray as xr
import numpy as np

def get_channel_by_stripe_ru(ds, stripe_idx, ru_idx):
    match = ((ds["stripe_idx"] == stripe_idx) & (ds["RU_idx"] == ru_idx))
    if match.any():
        tx_index = match.argmax().item()  # first match
        channel = ds["channel"].isel(tx_pair=tx_index)
        return channel
    else:
        print("No matching stripe/RU combination found.")
        return None
    
def channel_to_numpycomplex64(arr_struct):
    """
    Convert xarray DataArray channel which is a structured dtype with separate real and imaginary parts$
    back to numpy complex64 array.
    """
    arr_complex = arr_struct['r'] + 1j * arr_struct['i']
    return arr_complex.astype(np.complex64)


""" sub THz example usage """
user_id = 10  # Example user index
file_path = f"/home/user/6GTandem_RT_server/dataset_5681_ue_locations/sub_thz_channels/channels_thz_ue_{user_id}.nc"

ds = xr.load_dataset(file_path)
print(ds)

# print some meta data
print(f"User {ds.attrs['user_idx']} is in zone: {ds.attrs['zone']}")
print(f"Location: x={ds.attrs['user_x']}, y={ds.attrs['user_y']}, z={ds.attrs['user_z']}")

# access channel for a specific RU at stripe_idx and RU_idx
stripe_idx, RU_idx = 5, 15
channel = get_channel_by_stripe_ru(ds, stripe_idx=stripe_idx, ru_idx=RU_idx)
if channel is not None:
    print(f"channels between UE {user_id} at location: x={ds.attrs['user_x']}, y={ds.attrs['user_y']}, z={ds.attrs['user_z']} and RU {RU_idx} at stripe {stripe_idx}:")
    print(channel.shape)  # (rx_ant, tx_ant, subcarrier)
    print(channel.values.dtype)
    channel_np = channel_to_numpycomplex64(channel.values)
    print(channel_np.shape)  # (rx_ant, tx_ant, subcarrier)
    print(channel_np.dtype)  # complex64
    print(channel_np)  # complex64 array
else:
    print(f"invalide stripe or RU idx: {stripe_idx}, {RU_idx}")

#todo add example on loading the ue_locations dataset to be able to select which UE you pick based on the zone etc


""" sub 10 GHz example usage """

print(f'----------------------------- sub 10 GHz example -----------------------------')
file_path_sub10 = f"/home/user/6GTandem_RT_server/dataset_5681_ue_locations/sub_10ghz_channels/channels_sub10ghz_ue_{user_id}.nc"
ds_sub10 = xr.load_dataset(file_path_sub10)
print(ds_sub10)

# print all AP names
ap_names = ds_sub10["ap"].values
print("AP names:", ap_names)
channel_sub10 = ds_sub10["channel"].sel(ap=ap_names[2]).values  # select 3 AP
print(f"Channel shape for AP {ap_names[2]}: {channel_sub10.shape}")  # (rx_ant, tx_ant, subcarrier)
channel_sub10 = channel_to_numpycomplex64(channel_sub10)  # convert to numpy complex64
print(f"Channel data for AP {ap_names[2]}:\n", channel_sub10)