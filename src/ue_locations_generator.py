import numpy as np 
import xarray as xr
import os
from utils import load_config

# set seed
np.random.seed(2025)

def is_point_outside_zones(x, y, z):
    """ 
    check if point is in none of the zones
    """
    # Zone 1
    if x_zone1_start <= x <= x_zone1_stop and y_zone1_start <= y <= y_zone1_stop:
        return False  # Inside Zone 1

    # Zone 2
    if x_zone2_start <= x <= x_zone2_stop and y_zone2_start <= y <= y_zone2_stop:
        return False  # Inside Zone 2

    # Zone 3
    if x_zone3_start <= x <= x_zone3_stop and y_zone3_start <= y <= y_zone3_stop:
        return False  # Inside Zone 3

    # Zone 4
    if x_zone4_start <= x <= x_zone4_stop and y_zone4_start <= y <= y_zone4_stop:
        return False  # Inside Zone 4

    # Otherwise, point is outside all zones
    return True

# coordinates of the general zone
z_height = 1.5
x_start = 1.08
x_stop = 9.34
y_start = 1.88
y_stop = 24.89

# zone 1 coordinates
x_zone1_start = 4.9
x_zone1_stop = x_stop
y_zone1_start = y_start
y_zone1_stop = y_stop
area_zone1 = (x_zone1_stop - x_zone1_start) * (y_zone1_stop - y_zone1_start)

# zone 2 coordinates
x_zone2_start = x_start
x_zone2_stop = x_zone1_start
y_zone2_start = 5.51
y_zone2_stop = 14.2
area_zone2 = (x_zone2_stop - x_zone2_start) * (y_zone2_stop - y_zone2_start)

# zone 3 coordinates
x_zone3_start = x_start
x_zone3_stop = x_zone1_start
y_zone3_start = 15.6
y_zone3_stop = 23.65
area_zone3 = (x_zone3_stop - x_zone3_start) * (y_zone3_stop - y_zone3_start)

# zone 4 coordinates
x_zone4_start = x_start
x_zone4_stop = x_zone1_start
y_zone4_start = y_start
y_zone4_stop = 4.19
area_zone4 = (x_zone4_stop - x_zone4_start) * (y_zone4_stop - y_zone4_start)

# stay 10 cm away from the walls
safety_offset = 0.1

# load config file
config = load_config()

# generate samples for zone 1
num_point_zone1 = config['ue_locations_config']['num_locations_zone1'] #todo check running time and adjust accordingly 
x_zone1 = np.random.uniform(x_zone1_start + safety_offset, x_zone1_stop - safety_offset, num_point_zone1)
y_zone1 = np.random.uniform(y_zone1_start + safety_offset, y_zone1_stop - safety_offset, num_point_zone1)
samples_zone1 = np.column_stack((x_zone1, y_zone1, z_height * np.ones(num_point_zone1)))

# generate samples for Zone 2
num_point_zone2 = int(num_point_zone1 * area_zone2 / area_zone1) # scale the number of points based on area
x_zone2 = np.random.uniform(x_zone2_start + safety_offset, x_zone2_stop - safety_offset, num_point_zone2)
y_zone2 = np.random.uniform(y_zone2_start + safety_offset, y_zone2_stop - safety_offset, num_point_zone2)
samples_zone2 = np.column_stack((x_zone2, y_zone2, z_height * np.ones(num_point_zone2)))

# generate samples for Zone 3
num_point_zone3 = int(num_point_zone1 * area_zone3 / area_zone1) # scale the number of points based on area
x_zone3 = np.random.uniform(x_zone3_start + safety_offset, x_zone3_stop - safety_offset, num_point_zone3)
y_zone3 = np.random.uniform(y_zone3_start + safety_offset, y_zone3_stop - safety_offset, num_point_zone3)
samples_zone3 = np.column_stack((x_zone3, y_zone3, z_height * np.ones(num_point_zone3)))

# generate samples for Zone 4
num_point_zone4 = int(num_point_zone1 * area_zone4 / area_zone1) # scale the number of points based on area
x_zone4 = np.random.uniform(x_zone4_start + safety_offset, x_zone4_stop - safety_offset, num_point_zone4)
y_zone4 = np.random.uniform(y_zone4_start + safety_offset, y_zone4_stop - safety_offset, num_point_zone4)
samples_zone4 = np.column_stack((x_zone4, y_zone4, z_height * np.ones(num_point_zone4)))

# grid under each RU 
stripe_start_pos = config['stripe_config']['stripe_start_pos'] 
N_RUs = config['stripe_config']['N_RUs']# adjust to size of the room (along y axis)
N_stripes = config['stripe_config']['N_stripes'] # adjust to size of the room (alang x axis)
space_between_RUs = config['stripe_config']['space_between_RUs'] # in meters
space_between_stripses = config['stripe_config']['space_between_stripes'] # in meters
samples_grid = np.zeros((N_RUs * N_stripes, 3))
stripe_labels = []
ru_labels = []
invalid_point_labels = []
for stripe_idx in range(N_stripes):
    for RU_idx in range(N_RUs):
        # compute RU position
        pos = [stripe_start_pos[0] + stripe_idx * space_between_stripses,
                  stripe_start_pos[1] + RU_idx * space_between_RUs,
                  z_height]
        
        invalid_point = is_point_outside_zones(pos[0], pos[1], pos[2])
        if invalid_point:
            print(f'stripe {stripe_idx}, ru: {RU_idx}, pos: {pos} - point outside of zone: {invalid_point}')

        samples_grid[stripe_idx * N_RUs + RU_idx, :] = pos
        stripe_labels.append(stripe_idx)
        ru_labels.append(RU_idx)
        invalid_point_labels.append(invalid_point)



print(f"Zone 1: area: {area_zone1} - {samples_zone1.shape[0]} samples")
print(f"Zone 2: area: {area_zone2} - {samples_zone2.shape[0]} samples")
print(f"Zone 3: area: {area_zone3} - {samples_zone3.shape[0]} samples")
print(f"Zone 4: area: {area_zone4} - {samples_zone4.shape[0]} samples")

# todo save results to a file
# Combine samples into a single array
all_samples = np.vstack([samples_zone1, samples_zone2, samples_zone3, samples_zone4, samples_grid])
nr_ue_locs = all_samples.shape[0]
print(f'Total samples: {nr_ue_locs}')

zone_labels = np.array(['Zone 1'] * samples_zone1.shape[0] +
               ['Zone 2'] * samples_zone2.shape[0] +
               ['Zone 3'] * samples_zone3.shape[0] +
               ['Zone 4'] * samples_zone4.shape[0] +
               ['Grid'] * samples_grid.shape[0])

ue_on_stripe_grid = np.array(
    [False] * (samples_zone1.shape[0] +
               samples_zone2.shape[0] +
               samples_zone3.shape[0] +
               samples_zone4.shape[0]) +
    [True] * samples_grid.shape[0]
) # additional boolean to check if the ue is under the stripe grid or in a zone

stripe_labels = np.array([np.nan] * samples_zone1.shape[0] +
               [np.nan] * samples_zone2.shape[0] +
               [np.nan] * samples_zone3.shape[0] +
               [np.nan] * samples_zone4.shape[0] +
               stripe_labels)

ru_labels = np.array([np.nan] * samples_zone1.shape[0] +
               [np.nan] * samples_zone2.shape[0] +
               [np.nan] * samples_zone3.shape[0] +
               [np.nan] * samples_zone4.shape[0] +
               ru_labels)

invalid_point_labels = ([False] * samples_zone1.shape[0] +
               [False] * samples_zone2.shape[0] +
               [False] * samples_zone3.shape[0] +
               [False] * samples_zone4.shape[0] +
               invalid_point_labels)

# unique id per user
user_ids = np.arange(nr_ue_locs)

# Create the Dataset
ds = xr.Dataset(
    data_vars={
        "user_id": ("user", user_ids),
        "x": ("user", all_samples[:, 0].astype(np.float32)),
        "y": ("user", all_samples[:, 1].astype(np.float32)),
        "z": ("user", all_samples[:, 2].astype(np.float32)),
        "zone": ("user", zone_labels),
        "ue_stripe_idx": ("user", stripe_labels),
        "ue_ru_idx": ("user", ru_labels),
        "ue_on_stripe_grid": ("user", ue_on_stripe_grid),
        "invalid_point": ("user", invalid_point_labels)
    }
)


# Save 
basepath = config['paths']['basepath']
ue_path = os.path.join(basepath, 'dataset','ue_locations')
file_name = os.path.join(ue_path, f"ue_locations_{nr_ue_locs}.nc")
ds.to_netcdf(file_name)
print(f"Saved samples to {file_name}")


# todo some grid locations are in a cabinet => can be filtered out later 
# based on coordinates of the grid locations and the cabinets