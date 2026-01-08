import numpy as np
import xarray as xr
import os
from utils import create_folder

# set seed
np.random.seed(2025)


def is_point_invalid(objects, x, y, z):
    """Check that the point is not inside an object."""
    # Check if it is not inside any of the objects.
    for object in objects.values():
        if object[0] <= x <= object[1] and object[2] <= y <= object[3]:
            return True

    # Otherwise, point is outside all objects.
    return False


def create_user_location_dataset(config, logger, dataset_name, x_pos, y_pos):
    # check if dataset already exists
    basepath = config["paths"]["basepath"]
    dataset_path = os.path.join(basepath, dataset_name)
    ue_path = os.path.join(dataset_path, "ue_locations")
    create_folder(ue_path)
    file_name = os.path.join(ue_path, f"ue_locations.nc")
    if os.path.exists(file_name):
        logger.info(f"Loading existing dataset from {file_name}")
        ds_users = xr.load_dataset(file_name)
    else:
        logger.info(f"Dataset not found. Creating new dataset at {file_name}")

        # grid under each RU
        stripe_direction = config["stripe_config"]["stripe_direction"]
        N_RUs = config["stripe_config"]["N_RUs"]  # adjust to size of the room (along y axis)
        N_stripes = config["stripe_config"]["N_stripes"]  # adjust to size of the room (alang x axis)
        space_between_stripes = config["stripe_config"]["space_between_stripes"]  # in meters
        ue_config = config["ue_locations_config"]
        ue_points = ue_config["num_locations"]
        simulation_area = ue_config["ue_area"]
        safety_offset = ue_config["safety_offset"]
        z_height = ue_config["z_height"]
        objects = config["objects"]

        x_points = np.random.uniform(simulation_area[0] + safety_offset, simulation_area[1] - safety_offset, ue_points)
        y_points = np.random.uniform(simulation_area[2] + safety_offset, simulation_area[3] - safety_offset, ue_points)
        samples = np.column_stack((x_points, y_points, z_height * np.ones(ue_points)))

        nr_ue_locs = ue_points + (N_RUs * N_stripes)
        logger.info(f"Total samples: {nr_ue_locs}")

        # generate dataset of ue locations
        samples_grid = np.zeros((N_RUs * N_stripes, 3))
        stripe_labels = []
        ru_labels = []
        for stripe_idx in range(N_stripes):
            for RU_idx in range(N_RUs):
                # compute RU position
                rux = x_pos[RU_idx]
                ruy = y_pos[RU_idx]

                if stripe_direction == "y":
                    rux += stripe_idx * space_between_stripes
                elif stripe_direction == "x":
                    ruy += stripe_idx * space_between_stripes
                else:
                    raise ValueError(f"Invalid stripe direction: {stripe_direction}")

                samples_grid[stripe_idx * N_RUs + RU_idx, :] = [rux, ruy, z_height]
                stripe_labels.append(stripe_idx)
                ru_labels.append(RU_idx)

        # Combine samples into a single array
        all_samples = np.vstack([samples, samples_grid])
        nr_ue_locs = all_samples.shape[0]

        invalid_point_labels = []
        for s in all_samples:
            invalid_point = is_point_invalid(objects, *s)
            if invalid_point:
                logger.info(f"Point {s} - point outside of zone.")
            invalid_point_labels.append(invalid_point)

        zone_labels = np.array(["Zone 1"] * samples.shape[0] + ["Grid"] * samples_grid.shape[0])

        # Additional boolean to check if the ue is under the stripe grid or in a zone.
        ue_on_stripe_grid = np.array([False] * samples.shape[0] + [True] * samples_grid.shape[0])

        stripe_labels = np.array([np.nan] * samples.shape[0] + stripe_labels)
        ru_labels = np.array([np.nan] * samples.shape[0] + ru_labels)

        # unique id per user
        user_ids = np.arange(nr_ue_locs)

        # Create the Dataset
        ds_users = xr.Dataset(
            data_vars={
                "user_id": ("user", user_ids),
                "x": ("user", all_samples[:, 0].astype(np.float32)),
                "y": ("user", all_samples[:, 1].astype(np.float32)),
                "z": ("user", all_samples[:, 2].astype(np.float32)),
                "zone": ("user", zone_labels),
                "ue_stripe_idx": ("user", stripe_labels),
                "ue_ru_idx": ("user", ru_labels),
                "ue_on_stripe_grid": ("user", ue_on_stripe_grid),
                "invalid_point": ("user", invalid_point_labels),
            }
        )

        # Save
        create_folder(ue_path)
        file_name = os.path.join(ue_path, f"ue_locations.nc")
        ds_users.to_netcdf(file_name)
        logger.info(f"Saved samples to {file_name}")

    return ds_users, dataset_path
