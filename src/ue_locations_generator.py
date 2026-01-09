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
        N_RUs = config['stripe_config']['N_RUs']# adjust to size of the room (along y axis)
        N_stripes = config['stripe_config']['N_stripes'] # adjust to size of the room (alang x axis)
        stripe_direction = config["stripe_config"]["stripe_direction"]
        space_between_stripes = config['stripe_config']['space_between_stripes'] # in meters
        ue_config = config["ue_locations_config"]
        ue_zones = ue_config.get("zones", None)
        objects = config["objects"]

        if ue_zones is None:
            ue_points = ue_config['num_locations']
            simulation_area = ue_config["ue_area"]
            safety_offset = ue_config["safety_offset"]
            z_height = ue_config["z_height"]

            x_points = np.random.uniform(simulation_area[0] + safety_offset, simulation_area[1]  - safety_offset, ue_points)
            y_points = np.random.uniform(simulation_area[2] + safety_offset, simulation_area[3] - safety_offset, ue_points)
            samples = np.column_stack((x_points, y_points, z_height * np.ones(ue_points)))

            nr_ue_locs = ue_points + (N_RUs * N_stripes)

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

            # Additional boolean to check if the ue is under the stripe grid or in a zone.
            ue_on_stripe_grid = np.array([False] * samples.shape[0] + [True] * samples_grid.shape[0])

            stripe_labels = np.array([np.nan] * samples.shape[0] + stripe_labels)
            ru_labels = np.array([np.nan] * samples.shape[0] + ru_labels)
        else:
            samples = []

            for zone in ue_zones.values():
                bounds = zone["bounds"]
                steps = zone["steps"]
                cols = zone["cols"]
                rows = zone["rows"]

                x_points = np.linspace(bounds[0], bounds[1], cols)
                y_points = np.linspace(bounds[2], bounds[3], rows)
                z_points = np.linspace(bounds[4], bounds[5], rows)

                x_rep = np.repeat(x_points, len(y_points))
                y_tile = np.tile(y_points, len(x_points))
                z_rep = np.tile(z_points, cols)

                samples.append(np.column_stack((x_rep, y_tile, z_rep)))
            
            all_samples = np.concatenate(samples)
            # Additional boolean to check if the ue is under the stripe grid or in a zone.
            ue_on_stripe_grid = np.array([False] * all_samples.shape[0])

            stripe_labels = np.array([np.nan] * all_samples.shape[0])
            ru_labels = np.array([np.nan] * all_samples.shape[0])


        nr_ue_locs = all_samples.shape[0]
        logger.info(f"Total samples: {nr_ue_locs}")

        invalid_point_labels = []
        for s in all_samples:
            invalid_point = is_point_invalid(objects, *s)
            if invalid_point:
                logger.info(f"Point {s} - point outside of zone.")
            invalid_point_labels.append(invalid_point)

        # unique id per user
        user_ids = np.arange(nr_ue_locs)

        # Create the Dataset
        ds_users = xr.Dataset(
            data_vars={
                "user_id": ("user", user_ids),
                "x": ("user", all_samples[:, 0].astype(np.float32)),
                "y": ("user", all_samples[:, 1].astype(np.float32)),
                "z": ("user", all_samples[:, 2].astype(np.float32)),
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
