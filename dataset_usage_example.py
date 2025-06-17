import xarray as xr

def get_channel_by_stripe_ru(ds, stripe_idx, ru_idx):
    match = ((ds["stripe_idx"] == stripe_idx) & (ds["RU_idx"] == ru_idx))
    if match.any():
        tx_index = match.argmax().item()  # first match
        channel = ds["channel"].isel(tx_pair=tx_index)
        return channel
    else:
        print("No matching stripe/RU combination found.")
        return None


user_id = 10  # Example user index
file_path = f"/home/user/6GTandem_RT_server/dataset/sub_thz_channels/channels_thz_ue_{user_id}.nc"

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
else:
    print(f"invalide stripe or RU idx: {stripe_idx}, {RU_idx}")

#todo add example on loading the ue_locations dataset to be able to select which UE you pick based on the zone etc

