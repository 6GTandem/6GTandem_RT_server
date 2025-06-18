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
                  ituf_polystyrene_callback, ituf_mdf_callback, load_config

def setup():
    # check versions and set up GPU
    print("Sionna version:", sionna.rt.__version__)
    print("tf version:", tf.__version__)
    gpus = tf.config.list_physical_devices('GPU')
    print("GPU:", gpus)
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.set_visible_devices(gpus[0], 'GPU') # only use the first GPU
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
            print("Using GPU:", gpus[0].name)
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)
    else:
        print("No GPU found, using CPU.")

def set_materials(scene, config):
    # check which materials are available in the scene
    print("Available materials:")
    for name, obj in scene.objects.items():
        print(f'{name:<15}{obj.radio_material.name}')

    # set materials for the scene
    # add callbacks to the materials
    print("radio materials with associated callbacks:")
    # Ceiling_Detail => polystyrene
    ceiling_object = scene.get("Ceiling_Detail")
    ceiling_object.radio_material.frequency_update_callback = ituf_polystyrene_callback
    print(ceiling_object.radio_material.name)
    print(ceiling_object.radio_material.frequency_update_callback)

    # no-name-1 => ituf_glass
    glass_objects = scene.get("no-name-1")
    glass_objects.radio_material.frequency_update_callback = ituf_glass_callback
    print(glass_objects.radio_material.name)
    print(glass_objects.radio_material.frequency_update_callback)

    # no-name-2 => ituf_concrete
    concrete_objects = scene.get("no-name-2")
    concrete_objects.radio_material.frequency_update_callback = ituf_concrete_callback
    print(concrete_objects.radio_material.name)
    print(concrete_objects.radio_material.frequency_update_callback)

    # no-name-3 => ituf_metal
    metal_objects = scene.get("no-name-3")
    metal_objects.radio_material.frequency_update_callback = ituf_metal_callback
    print(metal_objects.radio_material.name)
    print(metal_objects.radio_material.frequency_update_callback)

    # no-name-4 => ituf_mdf
    metal_mdf = scene.get("no-name-4")
    metal_mdf.radio_material.frequency_update_callback = ituf_mdf_callback
    print(metal_mdf.radio_material.name)
    print(metal_mdf.radio_material.frequency_update_callback)

    # check conductivity and relative permittivity at different frequencies
    # loop through material names and print them
    sub_GHz = config['sub10GHz_config']['fc']
    sub_THz = config['subTHz_config']['fc']
    print(f"Checking materials at {sub_GHz/1e9} GHz and {sub_THz/1e9} GHz")
    for key, value in scene.objects.items():
        print(f'---------------{key=}----------------')
        # Print name of assigned radio material for different frequenies
        for f in [sub_GHz, sub_THz]: # Print for differrent frequencies
            scene.frequency = f
            value.radio_material.frequency_update() # update the frequency of the objects
            print(f"\nRadioMaterial: {value.radio_material.name} @ {scene.frequency/1e9}GHz")
            print("Conductivity:", value.radio_material.conductivity.numpy())
            print("Relative permittivity:", value.radio_material.relative_permittivity.numpy())
            print("Scattering coefficient:", value.radio_material.scattering_coefficient.numpy())
            print("XPD coefficient:", value.radio_material.xpd_coefficient.numpy())


if __name__ == "__main__":
    # check versions and set up GPU
    setup()

    # load config file
    config = load_config()

    # set output path
    channel_output_path = os.path.join(config['paths']['basepath'], 'dataset', 'sub_thz_channels')

    # compute or load all UE posotions s
    basepath = config['paths']['basepath']
    ue_path = os.path.join(basepath, 'dataset','ue_locations')
    ds_users = xr.load_dataset(os.path.join(ue_path, 'ue_locations_547.nc')) # todo get .nc filename automatically based on config

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
    print(f'number antennas per axis: {N_antennas}')

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
    print(f'bw type: {type(BW)}')
    print(f'bw type: {type(num_subcarriers)}')

    subcarrier_spacing = BW / num_subcarriers
    frequencies = subcarrier_frequencies(num_subcarriers, subcarrier_spacing) # Compute baseband frequencies of subcarriers relative to the carrier frequency
    print(f'subcarrier spacing = {subcarrier_spacing/1e6} MHz')

    # set scene frequency
    scene.frequency = config['subTHz_config']['fc']# Set frequency to fc 
    print(f"scene frequency set to: {scene.frequency}")

    # Instantiate a path solver
    # The same path solver can be used with multiple scenes
    p_solver  = PathSolver()
    print(f'path solver loop mode: {p_solver.loop_mode}') #symbolic mode is the fastest! 
    
    # loop over al ue postions
    for ue_idx in range(ds_users.dims['user']):
        # output file location
        out_file = os.path.join(channel_output_path, f"channels_thz_ue_{ue_idx}.nc")
        if os.path.exists(out_file):
            print(f"User {ue_idx} already processed. Skipping.")
            continue
        if ds_users.invalid_point.values[ue_idx]:
            print(f'User {ue_idx} is at an invalid location (within an object) and will not be processed. Skipping.')
            continue

        print(f"Processing user {ue_idx}...")

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
            (total_N_RUs, 2*N_antennas**2, 2*N_antennas**2, num_subcarriers),
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
            t1 = time.time()
            # log
            print(f'Processing UE {ue_idx} stripe {stripe_idx} ...')
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
                    print(f' rendering scene prior to path solver')
                    scene.render_to_file(camera=my_cam, filename=f'scene_with_stripe_{stripe_idx}_RU_{RU_idx}.png', 
                                        resolution=[650, 500], num_samples=512, clip_at=20) 



                # todo recheck this
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
                    print(f' rendering scene after removing tx')
                    scene.render_to_file(camera=my_cam, filename=f'scene_tx_removed_stripe_{stripe_idx}_RU_{RU_idx}.png', 
                                         resolution=[650, 500], num_samples=512, clip_at=20) 

            # end time for 1 stripe
            t2 = time.time()
            print(f"Time to compute stripe {stripe_idx}: {t2-t1:.2f} seconds")

        # remove rx from the scene after computation
        scene.remove(f"rx_{ue_idx}")
        
        # logging
        t_end_ue = time.time()
        print(f'Finished processing UE {ue_idx} in {t_end_ue-t_start_ue:.2f} seconds')

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
                "rx_ant": np.arange(2*N_antennas**2),
                "tx_ant": np.arange(2*N_antennas**2),
                "subcarrier": np.arange(num_subcarriers),
            },
            attrs=user_attrs
        )


        ds_user_channels.to_netcdf(out_file, format="NETCDF4", auto_complex=True)
        print(f"Saved user {ue_idx} to {out_file}")


        # todo: intermediate save of the results to file 

        # todo: remove receiver from the scene



    
    
    


    """ example for 1 RU
    # Create sub-THz stripes
    stripe_start_pos = [2, 2.5, 3.5]
    N_RUs = 1 #40 # todo adjust to size of the room (along y axis)
    N_stripes = 1 #10 # todo adjust to size of the room (alang x axis)
    space_between_RUs = 0.5 # in meters
    space_between_stripses = 0.5 # in meters
    for stripe_idx in range(N_stripes):
        for RU_idx in range(N_RUs):
            # compute RU position
            tx_pos = [stripe_start_pos[0] + stripe_idx * space_between_stripses,
                    stripe_start_pos[1] + RU_idx * space_between_RUs,
                    stripe_start_pos[2]]
            
            tx = Transmitter(name=f"tx_stripe_{stripe_idx}_RU_{RU_idx}",
                            position=tx_pos,
                            display_radius=0.1)

            # Add RU transmitter instance to scene
            scene.add(tx)

            # Point the transmitter downwards
            tx.look_at([tx_pos[0], tx_pos[1], 0]) # Transmitter points downwards


    # Create a receiver
    ue_idx = 1 # todo change dynamically 
    rx = Receiver(name=f"rx_{ue_idx}",
                position=ue_pos,
                display_radius=0.5)

    # Add receiver instance to scene
    scene.add(rx)

    # Render scene with new camera*
    scene.render_to_file(camera=my_cam, filename='scene_with_stripes.png', resolution=[650, 500], num_samples=512) # Increase num_samples to increase image quality

    # set scene frequency
    scene.frequency = 157.75e9 # Set frequency to 170 GHz

    # Instantiate a path solver
    # The same path solver can be used with multiple scenes
    p_solver  = PathSolver()

    # OFDM system parameters
    BW = 12.5e9 # Bandwidth of the system
    num_subcarriers = 2**15 #=32768
    subcarrier_spacing= BW / num_subcarriers
    f_axis = scene.frequency - BW/2 + subcarrier_spacing * np.arange(num_subcarriers)  # array of frequencies
    print(f'subcarrier spacing = {subcarrier_spacing/1e6} MHz')
    print(f' own faxis = faxis = {f_axis/1e9} GHz')

    # Compute propagation paths
    t1 = time.time()
    paths = p_solver(scene=scene,
                    max_depth=5,
                    los=True,
                    specular_reflection=True,
                    diffuse_reflection=False,
                    refraction=True,
                    synthetic_array=False,
                    seed=41)


    # Compute frequencies of subcarriers relative to the carrier frequency
    frequencies = subcarrier_frequencies(num_subcarriers, subcarrier_spacing)

    # Compute channel frequency response
    h_freq = paths.cfr(frequencies=frequencies,
                    normalize=True, # Normalize energy
                    normalize_delays=True,
                    out_type="numpy")

    t2 = time.time()
    print(f"Time to compute paths: {t2-t1:.2f} seconds")

    # Shape: [num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, num_subcarriers]
    print("Shape of h_freq: ", h_freq.shape)
    print(f' builtin faxis = faxis = {frequencies/1e9} GHz')

    # Plot absolute value
    habs = np.abs(h_freq)[0,0,0,0,0,:]
    plt.figure()
    plt.plot(np.squeeze(frequencies)/1e9, np.squeeze(habs));
    plt.xlabel("Baseband frequency (GHz)");
    plt.ylabel(r"|$h_\text{freq}$|");
    plt.title("Channel frequency response");
    plt.savefig("cfr.png")

    plt.figure()
    plt.plot(np.squeeze(f_axis)/1e9, np.squeeze(habs));
    plt.xlabel("frequency (GHz)");
    plt.ylabel(r"|$h_\text{freq}$|");
    plt.title("Channel frequency response");
    plt.savefig("cfr_2.png")
"""
    print(f'done')