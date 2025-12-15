import os
import sionna.rt
import time
import xarray as xr
import drjit as dr
import mitsuba as mi
import numpy as np
from tqdm import tqdm
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.constants import speed_of_light
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, Camera,\
                      PathSolver, subcarrier_frequencies, AntennaPattern
from utils import ituf_glass_callback, ituf_concrete_callback, ituf_metal_callback, \
                  ituf_polystyrene_callback, ituf_mdf_callback, load_config, create_folder
from ue_locations_generator import create_user_location_dataset
import logging
import datetime
import mitsuba as mi
from sionna.rt import ITURadioMaterial

logger = logging.getLogger(__name__)

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
    sub_GHz = config['sub10GHz_config']['fc']
    sub_THz = config['subTHz_config']['fc']
    logger.info(f"Checking materials at {sub_GHz / 1e9} GHz and {sub_THz / 1e9} GHz")
    for key, value in scene.objects.items():
        logger.info(f'---------------{key=}----------------')
        # Print name of assigned radio material for different frequenies
        for f in [sub_GHz, sub_THz]: # Print for differrent frequencies
            scene.frequency = f
            value.radio_material.frequency_update() # update the frequency of the objects
            logger.info(f"\nRadioMaterial: {value.radio_material.name} at {scene.frequency[0] / 1e9} GHz")
            logger.info(f"Conductivity: {value.radio_material.conductivity.numpy()}")
            logger.info(f"Relative permittivity: {value.radio_material.relative_permittivity.numpy()}")
            logger.info(f"Scattering coefficient: {value.radio_material.scattering_coefficient.numpy()}")
            logger.info(f"XPD coefficient: {value.radio_material.xpd_coefficient.numpy()}")

def sph_to_cart(theta, phi):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.array([x, y, z])

def rotation_matrix(angles):
    """
    Computes the rotation matrix from ZYX Euler angles.

    Parameters
    ----------
    angles : array-like of shape (3,)
        Rotation angles [alpha, beta, gamma] in radians.
        alpha : rotation about Z
        beta  : rotation about Y
        gamma : rotation about X

    Returns
    -------
    rot_mat : ndarray of shape (3, 3)
        Rotation matrix.
    """

    a, b, c = angles  # alpha, beta, gamma

    sin_a, cos_a = np.sin(a), np.cos(a)
    sin_b, cos_b = np.sin(b), np.cos(b)
    sin_c, cos_c = np.sin(c), np.cos(c)

    r_11 = cos_a * cos_b
    r_12 = cos_a * sin_b * sin_c - sin_a * cos_c
    r_13 = cos_a * sin_b * cos_c + sin_a * sin_c

    r_21 = sin_a * cos_b
    r_22 = sin_a * sin_b * sin_c + cos_a * cos_c
    r_23 = sin_a * sin_b * cos_c - cos_a * sin_c

    r_31 = -sin_b
    r_32 = cos_b * sin_c
    r_33 = cos_b * cos_c

    rot_mat = np.array([
        [r_11, r_12, r_13],
        [r_21, r_22, r_23],
        [r_31, r_32, r_33]
    ])

    return rot_mat


def cart_to_sph(v):
    # v is the (3, N) matrix, where N is the number of rays
    x, y, z = v[0, :], v[1, :], v[2, :]
    
    # FIX: Calculate the norm for each ray (column) along axis 0.
    r = np.linalg.norm(v, axis=0) # <--- THIS IS CRITICAL AND MUST BE IN YOUR CODE
    
    z_over_r = np.clip(z / r, -1.0, 1.0)
    theta = np.arccos(z_over_r)
    phi = np.arctan2(y, x)
    
    # We apply the phi wrap here for proper [0, 2pi] indexing
    phi = np.where(phi < 0, phi + 2 * np.pi, phi) 
    
    return theta, phi

def adjust_angles(theta, phi):
    v = sph_to_cart(theta, phi)
    rot_mat = rotation_matrix(np.array([0, -dr.pi/2, 0])) #rotate_z_to_x(v)
    v_rot = rot_mat @ v
    theta_r, phi_r = cart_to_sph(v_rot)

    return theta_r, phi_r

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
        mag = df["mag(rERHCP)[mV]"].values / 1000  # convert mV to V
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

            # adjust radiation pattern orientation (pointing in z direction to pointing in x direction)
            theta_np, phi_np = adjust_angles(theta_np, phi_np) 

            # Stack and interpolate
            pts = np.stack([theta_np, phi_np], axis=-1)
            field_np = self._interp(pts)

            field_np = np.asarray(field_np, dtype=np.complex64)
            E_theta_np = field_np / np.sqrt(2)
            E_phi_np = 1j * field_np / np.sqrt(2)

            # Convert to Dr.Jit type
            c_theta = mi.Complex2f(E_theta_np.real, E_theta_np.imag)
            c_phi   = mi.Complex2f(E_phi_np.real, E_phi_np.imag)
            return c_theta, c_phi

        # Single polarization (RHCP)
        return [f]



if __name__ == "__main__":
    # Configure logging
    log_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"gpu_2_run_{log_time}.log"
    logging.basicConfig(
        filename=log_filename,              # Log file name
        filemode='a',                    # Append mode
        level=logging.INFO,              # Set to DEBUG for more verbosity
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    # also see logs in the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(console_handler)

    logger.info(f"Sionna version: {sionna.rt.__version__}" )

    # load config file
    config = load_config()
    print(f'config loaded: {config}')

    # register antenna patterns
    antenna_path = config['paths']['antenna_rad_path']

    # register all measured patterns
    for i in range(1, 5):
        path = os.path.join(antenna_path, f'element{i}.csv')

        def measured_pattern_factory(csv_path=path, normalize=False, **kwargs):
            """Factory method that returns an instance of the antenna pattern"""
            return MeasuredPattern(csv_path=csv_path, normalize=normalize)

        # Register it under a custom name
        sionna.rt.register_antenna_pattern(f"custom_measured_element_{i}", measured_pattern_factory)

    # create or load user dataset
    ds_users, dataset_path = create_user_location_dataset(config, logger)

    # set output path
    channel_output_path = os.path.join(dataset_path, 'sub_thz_channels')
    create_folder(channel_output_path)

    # load params from config file
    intermediate_reders = config['random_configs']['intermediate_renders'] # slows down the program a lot => only for debugging!!!

    # load scene# Load scene
    scene = load_scene(config['paths']['scenepath']) 

    # Check that the right custom materials are set.
    check_materials(config, scene)

    # preview scene 
    # Create new camera with different configuration
    my_cam = Camera(position=[9,35,0.5], look_at=[0,0,3])

    # Render scene with new camera
    if intermediate_reders:
        scene.render_to_file(camera=my_cam, filename='empty_scene.png', resolution=[650, 500], num_samples=512, clip_at=20) # Increase num_samples to increase image quality

    # set materials for the scene
    #set_materials(scene, config)

    # configure tx and rx arrays
    N_antennas = config['antenna_config']['N_antennas'] # 4x1 antennas
    logger.info(f'number antennas: {N_antennas}')

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
    antenna_spacing = (speed_of_light / scene.frequency[0]) / 2  # half wavelength spacing
    logger.info(f"scene frequency set to: {scene.frequency[0]}")
    logger.info(f"antenna spacing set to: {antenna_spacing} meters")


    # Instantiate a path solver
    # The same path solver can be used with multiple scenes
    p_solver  = PathSolver()
    logger.info(f'path solver loop mode: {p_solver.loop_mode}') #symbolic mode is the fastest! 

    # batch size
    batch_size = config['random_configs']['batch_size']
    logger.info(f'batch size: {batch_size}')
    
    # loop over al ue postions
    split_point = 1500 # process only a subset 
    for ue_idx in range(0, split_point):
    #for ue_idx in range(ds_users.dims['user']):
        # output file location
        out_file = os.path.join(channel_output_path, f"channels_thz_ue_{ue_idx}.nc")
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

        # add velocity to the user 
        if config['subTHz_config']['doppler']:
            rx_velocity = [np.random.uniform(0, 3), np.random.uniform(0, 3), 0]
            #scene.get(f"rx_{ue_idx}").velocity = rx_velocity
            logger.info(f"rx_{ue_idx} velocity: {scene.get(f'rx_{ue_idx}').velocity}")

        # Preallocate channel tensor and index arrays 
        channel_tensor = np.empty(
            (total_N_RUs, N_antennas, N_antennas, num_subcarriers),
            dtype=np.complex64
        )
        stripe_idx_arr = np.empty(total_N_RUs, dtype=np.int32)
        ru_idx_arr = np.empty(total_N_RUs, dtype=np.int32)
        tx_idx = 0

        # start time current ue computation
        t_start_ue = time.time()

        # to create lambda/2 spacing offset
        start_offset = -((N_antennas - 1) / 2) * antenna_spacing

        # loop over ue antenna elemenets
        for ue_ant_idx in range(N_antennas):
            # todo check if antenna pat changes each time!
            # Configure antenna array for all receivers
            scene.rx_array = PlanarArray(num_rows=1,
                                        num_cols=1,
                                        pattern=f"custom_measured_element_{ue_ant_idx+1}",
                                        polarization=config['antenna_config']['polarization'])
            # scene.rx_array.antenna_pattern.show()
            # plt.savefig(f"rx_array_custom_measured_element_{ue_ant_idx+1}")
            
            #print(f'ue_ant_idx: {ue_ant_idx}, pattern: {scene.rx_array._antenna_pattern.__dict__}')

            # Create a receiver
            antenna_spacing_offset = start_offset + (ue_ant_idx * antenna_spacing)
            ue_pos_adjusted = np.array(ue_pos) # Copy the base position [x, y, z]
            ue_pos_adjusted[0] += antenna_spacing_offset # Apply offset to the x-coordinate (for a ULA along X)
            ue_pos_adjusted = ue_pos_adjusted.tolist()
            #print(f'original ue pos: {ue_pos}, adjusted ue pos for ant idx {ue_ant_idx}: {ue_pos_adjusted} - antenna spacing: {antenna_spacing}')
            rx = Receiver(name=f"rx_{ue_idx}",
                        position=ue_pos_adjusted,
                        display_radius=0.5)
            
            # Point the receiver upwards
            rx.look_at([ue_pos_adjusted[0], ue_pos_adjusted[1], 3.5]) # Receiver points upwards

            # Add receiver instance to scene
            scene.add(rx)

            # if doppler is enabled, set velocity of the user
            if config['subTHz_config']['doppler']:
                scene.get(f"rx_{ue_idx}").velocity = rx_velocity

            # loop over ru antenna elemenets
            for ru_ant_idx in range(N_antennas):
                # todo configure ru with antenna idx
                scene.tx_array = PlanarArray(num_rows=1,
                                            num_cols=1,
                                            pattern=f"custom_measured_element_{ru_ant_idx+1}",
                                            polarization=config['antenna_config']['polarization'])

                # scene.tx_array.antenna_pattern.show()
                # plt.savefig(f"tx_array_custom_measured_element_{ru_ant_idx+1}")
                #print(f'ru_ant_idx: {ru_ant_idx}, pattern: {scene.tx_array._antenna_pattern.__dict__}')

                # loop over all stripes
                for stripe_idx in range(N_stripes):
                    #logger.info(f'Processing UE {ue_idx}, UE ant {ue_ant_idx}, RU ant {ru_ant_idx}, stripe {stripe_idx}...')

                    # loop over batches of RUs (because all RUs may not fit in memory)
                    for batch_start in range(0, N_RUs, batch_size):
                        batch_end = min(batch_start + batch_size, N_RUs)

                        tx_list = []  # store TX objects of this batch
                        ru_idx_list = []  # store indices of the RUs of this batch (needed when saving results)

                        # --- Build batch ---
                        for RU_idx in range(batch_start, batch_end):
                            # compute RU position
                            tx_pos = [stripe_start_pos[0] + stripe_idx * space_between_stripses,
                                    stripe_start_pos[1] + RU_idx * space_between_RUs,
                                    stripe_start_pos[2]]

                            antenna_spacing_offset_ru = start_offset + (ru_ant_idx * antenna_spacing)
                            tx_pos_adjusted = np.array(tx_pos) # Copy the base position [x, y, z]
                            tx_pos_adjusted[0] += antenna_spacing_offset # Apply offset to the x-coordinate (for a ULA along X)
                            tx_pos_adjusted = tx_pos_adjusted.tolist()

                            # Create RU transmitter instance
                            tx = Transmitter(name=f"tx_stripe_{stripe_idx}_RU_{RU_idx}",
                                        position=tx_pos_adjusted,
                                        display_radius=0.1)
                            tx_list.append(f"tx_stripe_{stripe_idx}_RU_{RU_idx}")
                            ru_idx_list.append(RU_idx)

                            # Add RU transmitter instance to scene
                            scene.add(tx)

                            # Point the transmitter downwards
                            tx.look_at([tx_pos_adjusted[0], tx_pos_adjusted[1], 0]) # Transmitter points downwards

                        # solve paths for current batch
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
                        h_freq = paths.cfr(frequencies=frequencies,
                                        normalize_delays=True,
                                        out_type="numpy")
                        
                        # print("Shape of h_freq: ", h_freq.shape)
                        # plt.stem(np.abs(h_freq[0, 0, 0, 0, 0, :]))
                        # plt.xlim(0, 100)
                        # plt.xlabel('Subcarrier index')
                        # plt.ylabel('|H|')
                        # plt.savefig(f'/home/user/6GTandem_RT_server/testingchannel.png')

                        # reshape to [batchsize, nr_subcarriers] # nrRUs in batch for 1 tx ant and 1 rx ant 
                        h_freq = np.squeeze(h_freq)
                        h_freq = h_freq[:, np.newaxis, np.newaxis, :] # add ant dims back [batchsize, 1, 1, nr_subcarriers]

                        # saving results
                        for i_in_batch, RU_idx in enumerate(ru_idx_list):
                            global_idx = stripe_idx * N_RUs + RU_idx # tx_idx in old (unbatched) code
                            channel_tensor[global_idx, ue_ant_idx, ru_ant_idx , :] = h_freq[i_in_batch] # store into channel tensor
                            stripe_idx_arr[global_idx] = stripe_idx
                            ru_idx_arr[global_idx]     = RU_idx

                        # remove all RUs of current batch from the scene
                        for tx_string in tx_list:
                            scene.remove(tx_string)

            # remove rx from the scene after computation
            scene.remove(f"rx_{ue_idx}")

        """ did one ue position """

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

        # logging
        t_end_ue = time.time()
        time_1_user = t_end_ue - t_start_ue
        logger.info(f"Finished processing UE {ue_idx}/{min(split_point, ds_users.sizes['user'])} in {time_1_user:.2f} seconds")
        logger.info(f"=====================> Estimated time left (h): {(time_1_user * (ds_users.sizes['user'] - ue_idx - 1)) / 3600:.2f} hours")