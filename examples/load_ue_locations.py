# example script on how to load UE locations and interact with the dataset
import xarray as xr

# Load the dataset
ds = xr.load_dataset(r"/home/user/6GTandem_RT_server/ue_locations/ue_locations_579.nc")

# Print dataset overview
print(ds)

# View all available zones
print("Available zones:", set(ds['zone'].values))

# --- Select users by zone ---
zone1_users = ds.where(ds['zone'] == 'Zone 1', drop=True)
zone2_users = ds.where(ds['zone'] == 'Zone 2', drop=True)
zone3_users = ds.where(ds['zone'] == 'Zone 3', drop=True)
zone4_users = ds.where(ds['zone'] == 'Zone 4', drop=True)
grid_users  = ds.where(ds['zone'] == 'Grid', drop=True)

# Print number of users per zone
print(f"Zone 1 users: {zone1_users.dims['user']}")
print(f"Zone 2 users: {zone2_users.dims['user']}")
print(f"Zone 3 users: {zone3_users.dims['user']}")
print(f"Zone 4 users: {zone4_users.dims['user']}")
print(f"Grid users:  {grid_users.dims['user']}")

# As xarray variables
x_coords = zone1_users['x']
y_coords = zone1_users['y']
z_coords = zone1_users['z']

# Or convert to NumPy for further processing
positions_zone1 = zone1_users[['x', 'y', 'z']].to_array().values.T  # shape: (num_users, 3)

# select users based if they are on the grid or not
on_grid_users = ds.where(ds['ue_on_stripe_grid'], drop=True)
off_grid_users = ds.where(~ds['ue_on_stripe_grid'], drop=True)

print(f"Users on grid: {on_grid_users.dims['user']}")
print(f"Users off grid: {off_grid_users.dims['user']}")

# get user under stripe idx and ru idx
stripe_idx = 2
ru_idx = 10
matched_user = ds.where(
    (ds['ue_on_stripe_grid']) &
    (ds['ue_stripe_idx'] == stripe_idx) &
    (ds['ue_ru_idx'] == ru_idx),
    drop=True
)

print(f"User at stripe: {stripe_idx} and RU: {ru_idx}: x={matched_user['x'].values}, "
        f"y={matched_user['y'].values}, "
        f"z={matched_user['z'].values}")




