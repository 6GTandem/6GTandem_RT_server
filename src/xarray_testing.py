import numpy as np 
import xarray as xr
import os

# --- generate dummy data to test ---
def generate_channel(user_idx, tx_stripe, tx_ru):
    # Replace this with actual model or ray tracing
    return (
        np.random.randn(nr_rx_antennas, nr_tx_antennas, nr_subcarriers) +
        1j * np.random.randn(nr_rx_antennas, nr_tx_antennas, nr_subcarriers)
    ).astype(np.complex64)

nr_rx_antennas = 4
nr_tx_antennas = 4
nr_subcarriers = 1024
max_tx_per_user = 500
N_stripes = 13
N_RUs = 42 #nr rus per stripe
total_N_RUs = N_stripes*N_RUs
output_file = r"/home/user/6GTandem_RT_server/dataset/channel_dataset_test.nc"

ds_users = xr.load_dataset(r"/home/user/6GTandem_RT_server/dataset/ue_locations/ue_locations_579.nc")
num_users = ds_users.sizes['user']
print(f'loaded {num_users} users')

base_output_dir = r"/home/user/6GTandem_RT_server/dataset/sub_thz_channels"#todo load from yaml

# compression for channel data
encoding = {d
    "channel": dict(zlib=True, complevel=5)
}

    # --- Main loop over users ---
for user_idx in range(num_users):
    out_file = os.path.join(base_output_dir, f"channels_thz_ue_{user_idx}.nc")
    if os.path.exists(out_file):
        print(f"User {user_idx} already processed. Skipping.")
        continue

    print(f"Processing user {user_idx}...")

    # Preallocate channel tensor and index arrays
    channel_tensor = np.empty(
        (total_N_RUs, nr_rx_antennas, nr_tx_antennas, nr_subcarriers),
        dtype=np.complex64
    )
    stripe_idx_arr = np.empty(total_N_RUs, dtype=np.int32)
    ru_idx_arr = np.empty(total_N_RUs, dtype=np.int32)

    tx_idx = 0
    # loop over all stripes
    for stripe_idx in range(N_stripes):
        # loop over all RUs in the stripe
        for RU_idx in range(N_RUs):

            # Generate channel shape: [nr_rx_antennas, nr_tx_antennas, nr_subcarriers]
            channel_tensor[tx_idx] = generate_channel(user_idx, stripe_idx, RU_idx)
            stripe_idx_arr[tx_idx] = stripe_idx
            ru_idx_arr[tx_idx] = RU_idx
            tx_idx += 1

     # Get user attributes
    user_attrs = {
        "user_idx": int(user_idx),
        "user_x": float(ds_users["x"][user_idx]),
        "user_y": float(ds_users["y"][user_idx]),
        "user_z": float(ds_users["z"][user_idx]),
        "zone": str(ds_users["zone"][user_idx].values),
        "ue_stripe_idx": (
            float(ds_users["ue_stripe_idx"][user_idx])
            if not np.isnan(ds_users["ue_stripe_idx"][user_idx])
            else "NaN"
        ),
        "ue_ru_idx": (
            float(ds_users["ue_ru_idx"][user_idx])
            if not np.isnan(ds_users["ue_ru_idx"][user_idx])
            else "NaN"
        ),
    }

    ds_user_channels = xr.Dataset(
        data_vars={
            "channel": (
                ("tx_pair", "rx_ant", "tx_ant", "subcarrier"),
                channel_tensor
            )
        },
        coords={
            "tx_pair": np.arange(total_N_RUs),
            "stripe_idx": ("tx_pair", stripe_idx_arr),
            "RU_idx": ("tx_pair", ru_idx_arr),
            "rx_ant": np.arange(nr_rx_antennas),
            "tx_ant": np.arange(nr_tx_antennas),
            "subcarrier": np.arange(nr_subcarriers),
        },
        attrs=user_attrs
    )


    ds_user_channels.to_netcdf(out_file, format="NETCDF4", auto_complex=True)
    print(f"Saved user {user_idx} to {out_file}")



print("Finished processing all users.")
