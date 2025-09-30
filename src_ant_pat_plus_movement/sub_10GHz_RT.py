import sionna.rt
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from tqdm import tqdm
import os
from sionna.rt import load_scene, AntennaArray, PlanarArray, Transmitter, Receiver, Camera,\
                      PathSolver, RadioMapSolver, subcarrier_frequencies
from utils import ituf_glass_callback, ituf_concrete_callback, ituf_metal_callback, \
                  ituf_polystyrene_callback, ituf_mdf_callback, load_config, create_folder
from ue_locations_generator import create_user_location_dataset
import logging
import datetime

def setup():
    # check versions and set up GPU
    logger.info(f"Sionna version: {sionna.rt.__version__}" )
    logger.info("ftf version: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    logger.info("GPU:", gpus)
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.set_visible_devices(gpus[0], 'GPU') # only use the first GPU
            logical_gpus = tf.config.list_logical_devices('GPU')
            logger.info(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
            logger.info("Using GPU:", gpus[0].name)
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            logger.info(e)
    else:
        logger.info("No GPU found, using CPU.")

def set_materials(scene, config):
    # check which materials are available in the scene
    logger.info("Available materials:")
    for name, obj in scene.objects.items():
        logger.info(f'{name:<15}{obj.radio_material.name}')

    # set materials for the scene
    # add callbacks to the materials
    logger.info("radio materials with associated callbacks:")
    # Ceiling_Detail => polystyrene
    ceiling_object = scene.get("Ceiling_Detail")
    ceiling_object.radio_material.frequency_update_callback = ituf_polystyrene_callback
    logger.info(ceiling_object.radio_material.name)
    logger.info(ceiling_object.radio_material.frequency_update_callback)

    # no-name-1 => ituf_glass
    glass_objects = scene.get("no-name-1")
    glass_objects.radio_material.frequency_update_callback = ituf_glass_callback
    logger.info(glass_objects.radio_material.name)
    logger.info(glass_objects.radio_material.frequency_update_callback)

    # no-name-2 => ituf_concrete
    concrete_objects = scene.get("no-name-2")
    concrete_objects.radio_material.frequency_update_callback = ituf_concrete_callback
    logger.info(concrete_objects.radio_material.name)
    logger.info(concrete_objects.radio_material.frequency_update_callback)

    # no-name-3 => ituf_metal
    metal_objects = scene.get("no-name-3")
    metal_objects.radio_material.frequency_update_callback = ituf_metal_callback
    logger.info(metal_objects.radio_material.name)
    logger.info(metal_objects.radio_material.frequency_update_callback)

    # no-name-4 => ituf_mdf
    metal_mdf = scene.get("no-name-4")
    metal_mdf.radio_material.frequency_update_callback = ituf_mdf_callback
    logger.info(metal_mdf.radio_material.name)
    logger.info(metal_mdf.radio_material.frequency_update_callback)

    # check conductivity and relative permittivity at different frequencies
    # loop through material names and print them
    sub_GHz = config['sub10GHz_config']['fc']
    sub_THz = config['subTHz_config']['fc']
    logger.info(f"Checking materials at {sub_GHz/1e9} GHz and {sub_THz/1e9} GHz")
    for key, value in scene.objects.items():
        logger.info(f'---------------{key=}----------------')
        # Print name of assigned radio material for different frequenies
        for f in [sub_GHz, sub_THz]: # Print for differrent frequencies
            scene.frequency = f
            value.radio_material.frequency_update() # update the frequency of the objects
            logger.info(f"\nRadioMaterial: {value.radio_material.name} at {scene.frequency/1e9} GHz")
            logger.info(f"Conductivity: {value.radio_material.conductivity.numpy()}")
            logger.info(f"Relative permittivity: {value.radio_material.relative_permittivity.numpy()}")
            logger.info(f"Scattering coefficient: {value.radio_material.scattering_coefficient.numpy()}")
            logger.info(f"XPD coefficient: {value.radio_material.xpd_coefficient.numpy()}")


if __name__ == "__main__":

    # Configure logging
    log_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"sub10_run_{log_time}.log"
    logging.basicConfig(
        filename=log_filename,              # Log file name
        filemode='a',                    # Append mode
        level=logging.INFO,              # Set to DEBUG for more verbosity
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logger = logging.getLogger(__name__)

    # also see logs in the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(console_handler)

    # check versions and set up GPU
    setup()

    # load config file
    config = load_config()

    # create or load user dataset
    ds_users, dataset_path = create_user_location_dataset(config, logger)

    # set output path
    channel_output_path = os.path.join(dataset_path, 'sub_10ghz_channels')
    create_folder(channel_output_path)

    # load params from config file
    intermediate_reders = config['random_configs']['intermediate_renders'] # slows down the program a lot => only for debugging!!!

    # load scene# Load scene
    scene = load_scene(config['paths']['scenepath']) 

    # preview scene 
    # Create new camera with different configuration
    my_cam = Camera(position=[9,35,0.5], look_at=[0,0,3])

    # Render scene with new camera
    if intermediate_reders:
        scene.render_to_file(camera=my_cam, filename='empty_scene.png', resolution=[650, 500], num_samples=512, clip_at=20) # Increase num_samples to increase image quality

    # set materials for the scene
    set_materials(scene, config)

    # configure tx and rx arrays
    N_antennas = config['antenna_config']['N_antennas_per_axis']
    logger.info(f'number antennas per axis: {N_antennas}')

    scene.tx_array = PlanarArray(num_rows=N_antennas,
                                num_cols=N_antennas,
                                vertical_spacing=0.5,
                                horizontal_spacing=0.5,
                                pattern=config['antenna_config']['pattern'],
                                polarization=config['antenna_config']['polarization'])

    # Configure antenna array for all receivers
    scene.rx_array = PlanarArray(num_rows=N_antennas,
                                num_cols=N_antennas,
                                vertical_spacing=0.5,
                                horizontal_spacing=0.5,
                                pattern=config['antenna_config']['pattern'],
                                polarization=config['antenna_config']['polarization'])
    
    


    # # sub-THz stripe specs 
    # stripe_start_pos = config['stripe_config']['stripe_start_pos']
    # N_RUs = config['stripe_config']['N_RUs'] # adjust to size of the room (along y axis)
    # N_stripes = config['stripe_config']['N_stripes']# adjust to size of the room (alang x axis)
    # total_N_RUs = N_RUs * N_stripes # total number of radio units
    # space_between_RUs = config['stripe_config']['space_between_RUs'] # in meters
    # space_between_stripses = config['stripe_config']['space_between_stripes'] # in meters

    # sub-10 APs specs
    N_APS = config['sub10GHz_config']['num_APs'] # number of APs in the scene
    ap_1_pos = config['sub10GHz_config']['ap_pos_ceil_1'] # position of AP 1 on the ceiling
    ap_2_pos = config['sub10GHz_config']['ap_pos_ceil_2'] # position of AP 2 on the ceiling
    ap_3_pos = config['sub10GHz_config']['ap_pos_wall_1'] # position of AP 3 on the wall
    ap_4_pos = config['sub10GHz_config']['ap_pos_wall_2'] # position of AP 4 on the wall
    ap_positions = [ap_1_pos, ap_2_pos, ap_3_pos, ap_4_pos] # positions of the APs
    ap_names = [f"AP_ceil_1", f"AP_ceil_2", f"AP_wall_1", f"AP_wall_2"] # names of the APs
   
    # OFDM system parameters
    BW = config['sub10GHz_config']['bw'] # Bandwidth of the system
    num_subcarriers = config['sub10GHz_config']['num_subcarriers']
    logger.info(f'bw type: {type(BW)}')
    logger.info(f'bw type: {type(num_subcarriers)}')

    subcarrier_spacing = BW / num_subcarriers
    frequencies = subcarrier_frequencies(num_subcarriers, subcarrier_spacing) # Compute baseband frequencies of subcarriers relative to the carrier frequency
    logger.info(f'subcarrier spacing = {subcarrier_spacing/1e3} KHz')

    # set scene frequency
    scene.frequency = config['sub10GHz_config']['fc']# Set frequency to fc 
    logger.info(f"scene frequency set to: {scene.frequency}")

    # Instantiate a path solver
    # The same path solver can be used with multiple scenes
    p_solver  = PathSolver()
    logger.info(f'path solver loop mode: {p_solver.loop_mode}') #symbolic mode is the fastest! 

    # add APs to scene
    for ap_idx in range(N_APS):
        print(f'Adding AP {ap_idx+1}/{N_APS} to the scene at position {ap_positions[ap_idx]} with name {ap_names[ap_idx]}')
        # Create AP transmitter instance
        tx = Transmitter(name=ap_names[ap_idx],
                    position=ap_positions[ap_idx],
                    display_radius=0.1)

        # Add RU transmitter instance to scene
        scene.add(tx)
        
        # Point the transmitter perpendicular to the wall or downwards depending on the AP
        if ap_idx < 2:
            # Point the transmitter downwards (ceiling APs)
            tx.look_at([ap_positions[ap_idx][0], ap_positions[ap_idx][1], 0])
        elif ap_idx == 2:
            # point the transmitter perpendicular to the wall (wall APs)
            tx.look_at([5, ap_positions[ap_idx][1], ap_positions[ap_idx][2]]) 
        elif ap_idx == 3:
            # point the transmitter perpendicular to the wall (wall APs)
            tx.look_at([ap_positions[ap_idx][0], 0, ap_positions[ap_idx][2]]) 

        # check orientation
        print(f'tx orientation: {tx.orientation}')
    
    # loop over al ue postions
    for ue_idx in range(ds_users.dims['user']):
        # output file location
        out_file = os.path.join(channel_output_path, f"channels_sub10ghz_ue_{ue_idx}.nc")
        if os.path.exists(out_file):
            logger.info(f"User {ue_idx} already processed. Skipping.")
            continue
        if ds_users.invalid_point.values[ue_idx]:
            logger.info(f'User {ue_idx} is at an invalid location (within an object) and will not be processed. Skipping.')
            continue

        logger.info(f"Processing user {ue_idx}/{ds_users.sizes['user']}...")

        # get coordinates
        x, y, z = ds_users.x.values[ue_idx], ds_users.y.values[ue_idx], ds_users.z.values[ue_idx]
        ue_pos = [float(x), float(y), float(z)]

        # Create a receiver
        rx = Receiver(name=f"rx_{ue_idx}",
                    position=ue_pos,
                    display_radius=0.5)
        
        # Point the receiver upwards
        rx.look_at([ue_pos[0], ue_pos[1], 3.5]) # Receiver points upwards

        # check orientation
        #print(f'rx orientation: {rx.orientation}')

        # Add receiver instance to scene
        scene.add(rx)

        # Preallocate channel tensor for APs and index arrays (2x N^2 because cross polarization)
        channel_tensor = np.empty(
            (N_APS, 2*N_antennas**2, 2*N_antennas**2, num_subcarriers),
            dtype=np.complex64
        ) 

        # start time current ue computation
        t_start_ue = time.time()

        # render scene with tx and rx
        if intermediate_reders:
            logger.info(f' rendering scene prior to path solver')
            scene.render_to_file(camera=my_cam, filename=f'scene_with_aps.png', 
                                resolution=[650, 500], num_samples=512, clip_at=20) 
            logger.info(f' done rendering,  strating path solver')

        # compute paths
        paths = p_solver(scene=scene,
                        max_depth=5,
                        los=True,
                        specular_reflection=True,
                        diffuse_reflection=False, # no scattering
                        refraction=True,
                        synthetic_array=False,
                        seed=41)

        # Compute channel frequency response
        # Shape: [num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, num_subcarriers]
        # note that because of cross polarization we get 2*num_rx_ant and 2*num_tx_ant
        # todo to be checked: structured as [ant_1_pol_1, ant_1_pol_2, ant_2_pol_1, ant_2_pol_2, ..., ant_N_pol_2]
        h_freq = paths.cfr(frequencies=frequencies,
                        normalize_delays=True,
                        out_type="numpy")
        #print("Shape of h_freq: ", h_freq.shape)

        # todo check shapes
        # reshape to [2*nr_rx_antennas, nr_APs, 2*nr_tx_antennas, nr_subcarriers]
        h_freq = np.squeeze(h_freq)
        print("Shape of h_freq post squeeze: ", h_freq.shape)
        channel_tensor = np.transpose(h_freq, (1, 0, 2, 3)) #reshape to [nr_APs, 2*nr_rx_antennas, 2*nr_tx_antennas, nr_subcarriers]

        # remove rx from the scene after computation
        scene.remove(f"rx_{ue_idx}")
        
        # logging
        t_end_ue = time.time()
        logger.info(f"Finished processing UE {ue_idx}/{ds_users.dims['user']} in {t_end_ue-t_start_ue:.2f} seconds - estimated time remaining: {((ds_users.dims['user'] - ue_idx - 1) * (t_end_ue-t_start_ue)):.2f} seconds")

        # save channel tensor for curren ue
        # Get user attributes
        user_attrs = {
            "user_idx": int(ue_idx),
            "user_x": float(ds_users["x"][ue_idx]),
            "user_y": float(ds_users["y"][ue_idx]),
            "user_z": float(ds_users["z"][ue_idx]),
            "zone": str(ds_users["zone"][ue_idx].values),
            "ue_stripe_idx": (
                float(ds_users["ue_stripe_idx"][ue_idx])
                if not np.isnan(ds_users["ue_stripe_idx"][ue_idx])
                else "NaN"
            ),
            "ue_ru_idx": (
                float(ds_users["ue_ru_idx"][ue_idx])
                if not np.isnan(ds_users["ue_ru_idx"][ue_idx])
                else "NaN"
            ),
        }

        ds_user_channels = xr.Dataset(
            data_vars={
                "channel": (
                    ("ap", "rx_ant", "tx_ant", "subcarrier"),
                    channel_tensor
                )
            },
            coords={
                "ap": ap_names,
                "rx_ant": np.arange(2*N_antennas**2),
                "tx_ant": np.arange(2*N_antennas**2),
                "subcarrier": np.arange(num_subcarriers),
            },
            attrs=user_attrs
        )


        ds_user_channels.to_netcdf(out_file, format="NETCDF4", auto_complex=True)
        logger.info(f"Saved user {ue_idx} to {out_file}")




    
    
    
