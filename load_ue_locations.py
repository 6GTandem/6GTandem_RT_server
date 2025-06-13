# example script on how to load UE locations and interact with the dataset
import xarray as xr

# Load the dataset
ds = xr.load_dataset("ue_locations.nc")

# Print dataset overview
print(ds)

# Unique zone names
print("Zones in dataset:", ds.zone.values.tolist())

# Access all Zone 1 samples
zone1 = ds.sel(sample=(ds.zone == "Zone 1"))
print("Zone 1 samples:", zone1)

# Get samples that belong to the Grid
grid_samples = ds.sel(sample=(ds.zone == "Grid"))
print("Grid sample count:", grid_samples.dims["sample"])

# Example: Samples at stripe 3
specific_ru = ds.where((ds.stripe_idx == 3), drop=True)
print("Stripe 3:", specific_ru.dims["sample"])

# Example: get coordinates of UE location under stripe 4 and RU 10
specific_ru= ds.where((ds.stripe_idx == 4) & (ds.ru_idx == 10), drop=True)
print("Stripe 3, RU 10:", specific_ru)

x_coords = specific_ru.x.values
y_coords = specific_ru.y.values
z_coords = specific_ru.z.values

print("X:", x_coords)
print("Y:", y_coords)
print("Z:", z_coords)
