import sionna.rt
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from tqdm import tqdm
import os
from sionna.rt import load_scene, AntennaArray, PlanarArray, Transmitter, Receiver, Camera,\
                      PathSolver, RadioMapSolver, subcarrier_frequencies, AntennaPattern
from utils import ituf_glass_callback, ituf_concrete_callback, ituf_metal_callback, \
                  ituf_polystyrene_callback, ituf_mdf_callback, load_config, create_folder
from ue_locations_generator import create_user_location_dataset
import logging
import datetime
import mitsuba as mi
import sys
import pandas as pd 
from scipy.interpolate import RegularGridInterpolator

def setup():
    # check versions and set up GPU
    logger.info(f"Sionna version: {sionna.rt.__version__}" )
    logger.info("ftf version: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    logger.info("GPU:", gpus)
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.set_visible_devices(gpus[1], 'GPU') # only use the first GPU
            logical_gpus = tf.config.list_logical_devices('GPU')
            logger.info(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
            logger.info("Using GPU:", gpus[1].name)
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

""" custom patterns for sionna based on measurements """
# antenna pattern based on measurements
class MeasuredPattern(AntennaPattern):
    """
    Custom antenna pattern based on CSV data.
    The CSV must contain columns:
    'Phi[deg]', 'Theta[deg]', 'mag(rERHCP)[mV]', 'ang_deg(rERHCP)[deg]'
    """

    def __init__(self, csv_path, normalize=False):
        super().__init__()

        # ---- Load CSV ----
        self.path = csv_path
        df = pd.read_csv(csv_path)
        phi_deg = df["Phi[deg]"].values # degrees
        theta_deg = df["Theta[deg]"].values # degrees
        mag = df["mag(rERHCP)[mV]"].values/1000  # convert mV to V
        #print(f'max mag in MeasuredPattern: {np.max(mag)*1000} mV')
        ang_deg = df["ang_deg(rERHCP)[deg]"].values

        # ---- Convert to radians and complex field ----
        phi = np.deg2rad(phi_deg)
        theta = np.deg2rad(theta_deg)
        field = mag * np.exp(1j * np.deg2rad(ang_deg))

        # Optional normalization (to make max field = 1)
        if normalize:
            field = field / np.max(np.abs(field))

        # ---- Build regular grids for interpolation ----
        # (assuming uniform or nearly-uniform sampling)
        phi_unique = np.sort(np.unique(phi))
        theta_unique = np.sort(np.unique(theta))

        # reshape to 2D grid [len(theta), len(phi)]
        n_theta = len(theta_unique)
        n_phi = len(phi_unique)
        field_grid = field.reshape(n_theta, n_phi)

        # Interpolator over (theta, phi)
        self._interp = RegularGridInterpolator(
            (theta_unique, phi_unique),
            field_grid,
            bounds_error=False,
            fill_value=1e-8
        )

    @property
    def patterns(self):
        def f(theta, phi):
            # Convert TF tensors to numpy arrays for interpolation
            theta_np = theta.numpy()
            phi_np = phi.numpy()

            # Stack and interpolate
            pts = np.stack([theta_np, phi_np], axis=-1)
            field_np = self._interp(pts)

            # Convert back to TF complex tensor
            field = tf.constant(field_np, dtype=tf.complex64)

            field_np = np.asarray(field_np, dtype=np.complex64)
            E_theta_np = field_np / np.sqrt(2)
            E_phi_np = 1j * field_np / np.sqrt(2)

            # Convert to Dr.Jit type
            c_theta = mi.Complex2f(E_theta_np.real, E_theta_np.imag)
            c_phi   = mi.Complex2f(E_phi_np.real, E_phi_np.imag)
            return c_theta, c_phi

        # Single polarization (RHCP)
        return [f]


class CosinePattern(AntennaPattern):
    """
    Dual-polarized cosine^n radiation pattern.
    Provides both E_theta and E_phi components.
    """

    def __init__(self, n=4):
        super().__init__()
        self.n = n

    @property
    def patterns(self):
        def f(theta, phi):

            # Magnitude pattern
            mag = np.cos(theta) ** self.n
            field = mag

            # Simple dual-polarization:
            E_theta = field
            E_phi = field   # same pattern, orthogonal polarization

            # Convert to Dr.Jit type
            c_theta = mi.Complex2f(E_theta.real, E_theta.imag)
            c_phi   = mi.Complex2f(E_phi.real, E_phi.imag)
            return c_theta, c_phi
        
        return [f]


def cosine_dualpol_factory(n=4, **kwargs):
    return CosinePattern(n=n)



if __name__ == "__main__":

    # Configure logging
    log_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"run_{log_time}.log"
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

    # ant patterns setup
    patern_idx = 1
    antenna_path = config['paths']['antenna_rad_path']
    path = os.path.join(antenna_path, f'element{patern_idx}.csv')

    def measured_pattern_factory(csv_path=path, normalize=False, **kwargs):
        """Factory method that returns an instance of the antenna pattern"""
        return MeasuredPattern(csv_path=csv_path, normalize=normalize)

    # Register it under a custom name
    sionna.rt.register_antenna_pattern(f"custom_measured_element_{patern_idx}", measured_pattern_factory)

    # register cosine pattern
    sionna.rt.register_antenna_pattern("cosine_dualpol", cosine_dualpol_factory)


    # create or load user dataset
    ds_users, dataset_path = create_user_location_dataset(config, logger)

    # set output path
    channel_output_path = os.path.join(dataset_path, 'sub_thz_channels')
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
    N_antennas = config['antenna_config']['N_antennas']
    logger.info(f'number antennas per axis: {N_antennas}')

    # scene.tx_array = PlanarArray(num_rows=N_antennas,
    #                             num_cols=1,
    #                             vertical_spacing=0.5,
    #                             horizontal_spacing=0.5,
    #                             pattern=f"custom_measured_element_{patern_idx}",
    #                                     polarization=config['antenna_config']['polarization'])
    
    scene.tx_array = PlanarArray(num_rows=N_antennas,
                                num_cols=1,
                                vertical_spacing=0.5,
                                horizontal_spacing=0.5,
                                pattern=f"cosine_dualpol",
                                        polarization=config['antenna_config']['polarization'])

    print(f'tx pattern: {scene.tx_array._antenna_pattern.__dict__}')

    # Configure antenna array for all receivers
    # scene.rx_array = PlanarArray(num_rows=N_antennas,
    #                             num_cols=1,
    #                             vertical_spacing=0.5,
    #                             horizontal_spacing=0.5,
    #                             pattern=f"custom_measured_element_{patern_idx}",
    #                                     polarization=config['antenna_config']['polarization'])

    scene.rx_array = PlanarArray(num_rows=N_antennas,
                            num_cols=1,
                            vertical_spacing=0.5,
                            horizontal_spacing=0.5,
                            pattern=f"cosine_dualpol",
                                    polarization=config['antenna_config']['polarization'])

    print(f'rx pattern: {scene.rx_array._antenna_pattern.__dict__}')

    # sub-THz stripe specs 
    stripe_start_pos = config['stripe_config']['stripe_start_pos']
    N_RUs = config['stripe_config']['N_RUs'] # adjust to size of the room (along y axis)
    N_stripes = config['stripe_config']['N_stripes']# adjust to size of the room (alang x axis)
    total_N_RUs = N_RUs * N_stripes # total number of radio units
    space_between_RUs = config['stripe_config']['space_between_RUs'] # in meters
    space_between_stripses = config['stripe_config']['space_between_stripes'] # in meters
   
    # OFDM system parameters
    BW = config['subTHz_config']['bw'] # Bandwidth of the system
    num_subcarriers = config['subTHz_config']['num_subcarriers']
    logger.info(f'bw type: {type(BW)}')
    logger.info(f'bw type: {type(num_subcarriers)}')

    subcarrier_spacing = BW / num_subcarriers
    frequencies = subcarrier_frequencies(num_subcarriers, subcarrier_spacing) # Compute baseband frequencies of subcarriers relative to the carrier frequency
    logger.info(f'subcarrier spacing = {subcarrier_spacing/1e6} MHz')

    # set scene frequency
    scene.frequency = config['subTHz_config']['fc']# Set frequency to fc 
    logger.info(f"scene frequency set to: {scene.frequency}")

    # Instantiate a path solver
    # The same path solver can be used with multiple scenes
    p_solver  = PathSolver()
    logger.info(f'path solver loop mode: {p_solver.loop_mode}') #symbolic mode is the fastest! 
    
    # loop over al ue postions
    for ue_idx in range(ds_users.dims['user']):
        # output file location
        out_file = os.path.join(channel_output_path, f"channels_thz_ue_{ue_idx}.nc")
        if os.path.exists(out_file):
            logger.info(f"User {ue_idx} already processed. Skipping.")
            continue
        if ds_users.invalid_point.values[ue_idx]:
            logger.info(f'User {ue_idx} is at an invalid location (within an object) and will not be processed. Skipping.')
            continue

        logger.info(f"Processing user {ue_idx}/{ds_users.dims['user']}...")

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

        # Preallocate channel tensor and index arrays (2x N^2 because cross polarization)
        channel_tensor = np.empty(
            (total_N_RUs, N_antennas, N_antennas, num_subcarriers),
            dtype=np.complex64
        )
        stripe_idx_arr = np.empty(total_N_RUs, dtype=np.int32)
        ru_idx_arr = np.empty(total_N_RUs, dtype=np.int32)

        tx_idx = 0

        # start time current ue computation
        t_start_ue = time.time()

        # loop over all stripes
        for stripe_idx in range(N_stripes):
            # start time 1 stripe computation
            #t1 = time.time()
            # log
            #logger.info(f"Processing UE {ue_idx}/{ds_users.dims['user']} stripe {stripe_idx} ...")
            # loop over all RUs 
            for RU_idx in range(N_RUs):
             
                # compute RU position
                tx_pos = [stripe_start_pos[0] + stripe_idx * space_between_stripses,
                        stripe_start_pos[1] + RU_idx * space_between_RUs,
                        stripe_start_pos[2]]
                
                # Create RU transmitter instance
                tx = Transmitter(name=f"tx_stripe_{stripe_idx}_RU_{RU_idx}",
                            position=tx_pos,
                            display_radius=0.1)

                # Add RU transmitter instance to scene
                scene.add(tx)

                # Point the transmitter downwards
                tx.look_at([tx_pos[0], tx_pos[1], 0]) # Transmitter points downwards

                # check orientation
                #print(f'tx orientation: {tx.orientation}')

                # render scene with tx and rx
                if intermediate_reders:
                    logger.info(f' rendering scene prior to path solver')
                    scene.render_to_file(camera=my_cam, filename=f'scene_with_stripe_{stripe_idx}_RU_{RU_idx}.png', 
                                        resolution=[650, 500], num_samples=512, clip_at=20) 



                # todo recheck this
                paths = p_solver(scene=scene,
                                max_depth=4,
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

                # scene.render_to_file(camera=my_cam, filename=f'testing_paths.png', paths=paths,
                #         resolution=[650, 500], num_samples=512, clip_at=20) 
                # print(f'rendered to file')
                #print("Shape of h_freq: ", h_freq.shape)

                # todo check shapes
                # reshape to [2*nr_rx_antennas, 2*nr_tx_antennas, nr_subcarriers]
                h_freq = np.squeeze(h_freq)
                #print("Shape of h_freq post squeeze: ", h_freq.shape)

                # plug into channel tensor
                channel_tensor[tx_idx] = h_freq

                # assign stripe and ru idx
                stripe_idx_arr[tx_idx] = stripe_idx
                ru_idx_arr[tx_idx] = RU_idx

                # increment tx idx counter
                tx_idx += 1

                # remove tx from the scene after computation
                scene.remove(f"tx_stripe_{stripe_idx}_RU_{RU_idx}")

                # render scene with tx and rx
                if intermediate_reders:
                    logger.info(f' rendering scene after removing tx')
                    scene.render_to_file(camera=my_cam, filename=f'scene_tx_removed_stripe_{stripe_idx}_RU_{RU_idx}.png', 
                                         resolution=[650, 500], num_samples=512, clip_at=20) 

            # end time for 1 stripe
            #t2 = time.time()
            #logger.info(f"Time to compute stripe {stripe_idx}: {t2-t1:.2f} seconds")

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
                    ("tx_pair", "rx_ant", "tx_ant", "subcarrier"),
                    channel_tensor
                )
            },
            coords={
                "tx_pair": np.arange(total_N_RUs),
                "stripe_idx": ("tx_pair", stripe_idx_arr),
                "RU_idx": ("tx_pair", ru_idx_arr),
                "rx_ant": np.arange(N_antennas),
                "tx_ant": np.arange(N_antennas),
                "subcarrier": np.arange(num_subcarriers),
            },
            attrs=user_attrs
        )


        ds_user_channels.to_netcdf(out_file, format="NETCDF4", auto_complex=True)
        logger.info(f"Saved user {ue_idx} to {out_file}")




    
    
    
