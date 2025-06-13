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

# Example: All samples in stripe 5
stripe_5 = ds.sel(sample=(ds.stripe_idx == 5))
print("Stripe 5 samples:", stripe_5)

# Example: All samples for RU index 20 on different stripes
ru_20 = ds.sel(sample=(ds.ru_idx == 20))
print("RU index 20 samples:", ru_20)

# Example: Sample at stripe 3, RU 10
specific_ru = ds.sel(sample=(ds.stripe_idx == 3) & (ds.ru_idx == 10))
print("Stripe 3, RU 10:", specific_ru)