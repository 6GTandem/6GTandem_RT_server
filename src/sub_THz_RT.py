import os
import yaml
import sionna.rt
import time
import xarray as xr
import mitsuba as mi
import numpy as np
import matplotlib.pyplot as plt

from sionna.rt import (
    load_scene,
    PlanarArray,
    Transmitter,
    Receiver,
    PathSolver,
    subcarrier_frequencies,
)
from utils import (
    ituf_glass_callback,
    ituf_concrete_callback,
    ituf_metal_callback,
    ituf_polystyrene_callback,
    ituf_mdf_callback,
    create_folder,
)
from ue_locations_generator import create_user_location_dataset
from patterns import MeasuredPattern
import logging
import datetime
from sionna.rt import ITURadioMaterial

logger = logging.getLogger(__name__)

simulation_environment = "industry_hall"

# For the custom materials, use an ITU material and change its callback.
def custom_mat(props, callback):
    itu_material = ITURadioMaterial(props=props)
    itu_material.frequency_update_callback = callback

    return itu_material


# Custom material BSDFs. These must match with the BSDF names in the .xml file.
# In the XML file the BSDF material must have a <string name="type" value="glass"/>
# where value is an existing ITU material.
mi.register_bsdf("custom_glass", lambda props: custom_mat(props, ituf_glass_callback))
mi.register_bsdf("custom_polystyrene", lambda props: custom_mat(props, ituf_polystyrene_callback))
mi.register_bsdf("custom_concrete", lambda props: custom_mat(props, ituf_concrete_callback))
mi.register_bsdf("custom_mdf", lambda props: custom_mat(props, ituf_mdf_callback))
mi.register_bsdf("custom_metal", lambda props: custom_mat(props, ituf_metal_callback))


def check_materials(config, scene):
    # check conductivity and relative permittivity at different frequencies
    # loop through material names and print them
    sub_GHz = config["sub10GHz_config"]["fc"]
    sub_THz = config["subTHz_config"]["fc"]
    logger.info(f"Checking materials at {sub_GHz / 1e9} GHz and {sub_THz / 1e9} GHz")
    for key, value in scene.objects.items():
        logger.info(f"---------------{key=}----------------")
        # Print name of assigned radio material for different frequenies
        for f in [sub_GHz, sub_THz]:  # Print for differrent frequencies
            scene.frequency = f
            value.radio_material.frequency_update()  # update the frequency of the objects
            logger.info(f"\nRadioMaterial: {value.radio_material.name} at {scene.frequency[0] / 1e9} GHz")
            logger.info(f"Conductivity: {value.radio_material.conductivity.numpy()}")
            logger.info(f"Relative permittivity: {value.radio_material.relative_permittivity.numpy()}")
            logger.info(f"Scattering coefficient: {value.radio_material.scattering_coefficient.numpy()}")
            logger.info(f"XPD coefficient: {value.radio_material.xpd_coefficient.numpy()}")


if __name__ == "__main__":

    # Configure logging
    log_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"run_{log_time}.log"
    logging.basicConfig(
        filename=log_filename,  # Log file name
        filemode="a",  # Append mode
        level=logging.INFO,  # Set to DEBUG for more verbosity
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # also see logs in the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info(f"Sionna version: {sionna.rt.__version__}")

    # Load config file.
    with open(f"environments/{simulation_environment}/config.yaml", 'r') as file:
        config = yaml.safe_load(file)
    logging.info(f"Config loaded: {config}")

    # Register antenna patterns.
    antenna_path = config["paths"]["antenna_rad_path"]

    # Register one measured pattern.
    path = os.path.join(antenna_path, f"element1.csv")

    def measured_pattern_factory(csv_path=path, normalize=False, **kwargs):
        """Factory method that returns an instance of the antenna pattern"""
        return MeasuredPattern(csv_path=csv_path, normalize=normalize)

    # Register it under a custom name.
    sionna.rt.register_antenna_pattern("custom_measured_element", measured_pattern_factory)

    # load scene
    scene = load_scene(config["paths"]["scenepath"])

    # Check that the right custom materials are set.
    check_materials(config, scene)

    # configure tx and rx arrays
    antenna_conf = config["antenna_config"]
    N_antennas = config["antenna_config"]["N_antennas"]
    logger.info(f"number antennas per axis: {N_antennas}")

    if antenna_conf["pattern"] == "measured":
        pattern = "custom_measured_element"
    elif antenna_conf["pattern"] == "tr38901":
        pattern = "tr38901"
    else:
        raise ValueError(f"Invalid antenna pattern selected: {antenna_conf['pattern']}")

    scene.tx_array = PlanarArray(
        num_cols=N_antennas,
        num_rows=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="tr38901",
        polarization=config["antenna_config"]["polarization"],
    )

    # Configure antenna array for all receivers
    scene.rx_array = PlanarArray(
        num_cols=N_antennas,
        num_rows=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="tr38901",
        polarization=config["antenna_config"]["polarization"],
    )

    # sub-THz stripe specs
    stripe_start_pos = config["stripe_config"]["stripe_start_pos"]
    stripe_end_pos = config["stripe_config"]["stripe_end_pos"]
    array_direction = config["stripe_config"]["array_direction"]
    N_RUs = config["stripe_config"]["N_RUs"]  # adjust to size of the room (along y axis)
    N_stripes = config["stripe_config"]["N_stripes"]  # adjust to size of the room (alang x axis)
    total_N_RUs = N_RUs * N_stripes  # total number of radio units
    space_between_RUs = config["stripe_config"]["space_between_RUs"]  # in meters
    space_between_stripes = config["stripe_config"]["space_between_stripes"]  # in meters
    stripe_direction = config["stripe_config"]["stripe_direction"]

    # Compute the RU positions.
    x_pos = np.linspace(stripe_start_pos[0], stripe_end_pos[0], N_RUs).tolist()
    y_pos = np.linspace(stripe_start_pos[1], stripe_end_pos[1], N_RUs).tolist()
    z_pos = np.linspace(stripe_start_pos[2], stripe_end_pos[2], N_RUs).tolist()
    # Perform a sanity check to see if the positions are ok.
    x_diff = x_pos[1] - x_pos[0]
    y_diff = y_pos[1] - y_pos[0]
    z_diff = z_pos[1] - z_pos[0]
    ru_dist = np.sqrt(x_diff ** 2 + y_diff ** 2 + z_diff ** 2) 

    assert np.isclose(ru_dist, space_between_RUs), f"Actual space between RUs {ru_dist} does not match the one specified in the config {space_between_RUs}"

    # Create or load user dataset.
    ds_users, dataset_path = create_user_location_dataset(config, logger, simulation_environment, x_pos, y_pos)

    # set output path
    channel_output_path = os.path.join(dataset_path, "sub_thz_channels")
    create_folder(channel_output_path)

    # OFDM system parameters
    BW = config["subTHz_config"]["bw"]  # Bandwidth of the system
    num_subcarriers = config["subTHz_config"]["num_subcarriers"]
    logger.info(f"bw type: {type(BW)}")
    logger.info(f"bw type: {type(num_subcarriers)}")

    subcarrier_spacing = BW / num_subcarriers
    frequencies = subcarrier_frequencies(
        num_subcarriers, subcarrier_spacing
    )  # Compute baseband frequencies of subcarriers relative to the carrier frequency
    logger.info(f"subcarrier spacing = {subcarrier_spacing/1e6} MHz")

    # set scene frequency
    scene.frequency = config["subTHz_config"]["fc"]  # Set frequency to fc
    logger.info(f"scene frequency set to: {scene.frequency[0]}")

    # Instantiate a path solver
    # The same path solver can be used with multiple scenes
    p_solver = PathSolver()
    logger.info(f"path solver loop mode: {p_solver.loop_mode}")  # symbolic mode is the fastest!

    # By default the antenna array points towards the x-axis.
    # The orientation specifies three angles, yaw, pitch and roll respectively.
    # Yaw is a rotation around the z-axis, Pitch is around the y-axis and Roll is around the x-axis.
    # So setting pitch to 90deg rotates the RU downwards while the yaw rotates the antenna array orientation.
    # This means that only applying pitch makes the array stay aligned along the y-axis while applying yaw makes
    # it aligned along the x-axis.
    yaw = 0
    if array_direction == "x":
        yaw = np.pi / 2

    # loop over al ue postions
    for ue_idx in range(ds_users.sizes["user"]):
        # output file location
        out_file = os.path.join(channel_output_path, f"channels_thz_ue_{ue_idx}.nc")
        if os.path.exists(out_file):
            logger.info(f"User {ue_idx} already processed. Skipping.")
            continue
        if ds_users.invalid_point.values[ue_idx]:
            logger.info(
                f"User {ue_idx} is at an invalid location (within an object) and will not be processed. Skipping."
            )
            continue

        logger.info(f"Processing user {ue_idx}/{ds_users.sizes['user']}...")

        # get coordinates
        x, y, z = ds_users.x.values[ue_idx], ds_users.y.values[ue_idx], ds_users.z.values[ue_idx]
        ue_pos = mi.Point3f(float(x), float(y), float(z))

        # add velocity to the user
        if config["subTHz_config"]["doppler"]:
            rx_velocity = [np.random.uniform(0, 3), np.random.uniform(0, 3), 0]
            logger.info(f"rx_{ue_idx} velocity: {scene.get(f'rx_{ue_idx}').velocity}")
            scene.get(f"rx_{ue_idx}").velocity = rx_velocity

        # Create a receiver
        orientation = mi.Point3f(yaw, -np.pi/2, 0)
        rx = Receiver(name=f"rx_{ue_idx}", position=ue_pos, orientation=orientation)

        # Add receiver instance to scene
        scene.add(rx)

        # Preallocate channel tensor and index arrays (2x N^2 because cross polarization)
        channel_tensor = np.empty((total_N_RUs, N_antennas, N_antennas, num_subcarriers), dtype=np.complex64)
        stripe_idx_arr = np.empty(total_N_RUs, dtype=np.int32)
        ru_idx_arr = np.empty(total_N_RUs, dtype=np.int32)
        tx_idx = 0

        # start time current ue computation
        t_start_ue = time.time()

        # loop over all stripes
        for stripe_idx in range(N_stripes):
            # loop over all RUs
            for RU_idx in range(N_RUs):
                # Transmitting RU position.
                rux = x_pos[RU_idx]
                ruy = y_pos[RU_idx]
                ruz = z_pos[RU_idx]

                if stripe_direction == "y":
                    rux += stripe_idx * space_between_stripes
                elif stripe_direction == "x":
                    ruy += stripe_idx * space_between_stripes
                else:
                    raise ValueError(f"Invalid stripe direction: {stripe_direction}")

                # Create RU transmitter instance
                orientation = mi.Point3f(yaw, np.pi/2, 0)
                pos = mi.Point3f(rux, ruy, ruz)
                tx = Transmitter(name=f"tx_stripe_{stripe_idx}_RU_{RU_idx}", position=pos, display_radius=0.1, orientation=orientation)

                # Add RU transmitter instance to scene
                scene.add(tx)

                paths = p_solver(
                    scene=scene,
                    max_depth=5,
                    los=True,
                    specular_reflection=True,
                    diffuse_reflection=False,  # no scattering
                    refraction=True,
                    synthetic_array=False,
                    seed=41,
                )

                # Compute channel frequency response
                # Shape: [num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, num_subcarriers]
                h_freq = paths.cfr(frequencies=frequencies, normalize_delays=True, out_type="numpy")

                h_freq = np.squeeze(h_freq)

                # plug into channel tensor
                channel_tensor[tx_idx] = h_freq

                # assign stripe and ru idx
                stripe_idx_arr[tx_idx] = stripe_idx
                ru_idx_arr[tx_idx] = RU_idx

                # increment tx idx counter
                tx_idx += 1

                # remove tx from the scene after computation
                scene.remove(f"tx_stripe_{stripe_idx}_RU_{RU_idx}")

        # remove rx from the scene after computation
        scene.remove(f"rx_{ue_idx}")

        # logging
        t_end_ue = time.time()
        logger.info(f"Finished processing UE {ue_idx}/{ds_users.dims['user']} in {t_end_ue-t_start_ue:.2f} seconds")

        # save channel tensor for curren ue
        # Get user attributes
        user_attrs = {
            "user_idx": int(ue_idx),
            "user_x": float(ds_users["x"][ue_idx]),
            "user_y": float(ds_users["y"][ue_idx]),
            "user_z": float(ds_users["z"][ue_idx]),
            "zone": str(ds_users["zone"][ue_idx].values),
            "ue_stripe_idx": (
                float(ds_users["ue_stripe_idx"][ue_idx]) if not np.isnan(ds_users["ue_stripe_idx"][ue_idx]) else "NaN"
            ),
            "ue_ru_idx": (
                float(ds_users["ue_ru_idx"][ue_idx]) if not np.isnan(ds_users["ue_ru_idx"][ue_idx]) else "NaN"
            ),
        }

        ds_user_channels = xr.Dataset(
            data_vars={"channel": (("tx_pair", "rx_ant", "tx_ant", "subcarrier"), channel_tensor)},
            coords={
                "tx_pair": np.arange(total_N_RUs),
                "stripe_idx": ("tx_pair", stripe_idx_arr),
                "RU_idx": ("tx_pair", ru_idx_arr),
                "rx_ant": np.arange(N_antennas),
                "tx_ant": np.arange(N_antennas),
                "subcarrier": np.arange(num_subcarriers),
            },
            attrs=user_attrs,
        )

        ds_user_channels.to_netcdf(out_file, format="NETCDF4", auto_complex=True)
        logger.info(f"Saved user {ue_idx} to {out_file}")
