import sionna.rt
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from sionna.rt import load_scene, AntennaArray, PlanarArray, Transmitter, Receiver, Camera,\
                      PathSolver, RadioMapSolver, subcarrier_frequencies
from utils import ituf_glass_callback, ituf_concrete_callback, ituf_metal_callback, \
                  ituf_polystyrene_callback, ituf_mdf_callback

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

def set_materials(scene):
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
    sub_GHz = 3.5e9
    sub_THz = 157.75e9
    print(f"Checking materials at {sub_GHz/1e9}GHz and {sub_THz/1e12}THz")
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

    # todo compute or load all UE posotions 
    # todo load file name from yaml config file
    ds = xr.load_dataset(r"/home/user/6GTandem_RT_server/ue_locations/ue_locations_579.nc")

    ue_pos = [7, 10, 1]
    ue_positions = [ue_pos]

    # TODO load params from config file
    intermediate_reders = False # slows down the program a lot => only for debugging!!!

    # load scene# Load scene
    scene = load_scene(r"/home/user/6GTandem_RT_server/6G_Tandem_kantoorruimte_v10/office_space.xml") 

    # preview scene 
    # Create new camera with different configuration
    my_cam = Camera(position=[9,35,0.5], look_at=[0,0,3])

    # Render scene with new camera*
    if intermediate_reders:
        scene.render_to_file(camera=my_cam, filename='empty_scene.png', resolution=[650, 500], num_samples=512, clip_at=20) # Increase num_samples to increase image quality

    # set materials for the scene
    set_materials(scene)

    # todo change because this is in y-z plane
    N_antennas = 4 # Number of antennas in each direction (x and y)
    scene.tx_array = PlanarArray(num_rows=N_antennas,
                                num_cols=N_antennas,
                                vertical_spacing=0.5,
                                horizontal_spacing=0.5,
                                pattern="iso",
                                polarization="cross")

    # Configure antenna array for all receivers
    scene.rx_array = PlanarArray(num_rows=N_antennas,
                                num_cols=N_antennas,
                                vertical_spacing=0.5,
                                horizontal_spacing=0.5,
                                pattern="iso",
                                polarization="cross")


    # todo: sub-THz stripe specs => load from yamls config file
    stripe_start_pos = [2, 2.5, 3.5]
    N_RUs = 5 #40 # todo adjust to size of the room (along y axis)
    N_stripes = 2 #10 # todo adjust to size of the room (alang x axis)
    space_between_RUs = 0.5 # in meters
    space_between_stripses = 0.5 # in meters
    # todo load from yaml file OFDM system parameters
    BW = 12.5e9 # Bandwidth of the system
    num_subcarriers = 2**15 #=32768
    subcarrier_spacing= BW / num_subcarriers
    f_axis = scene.frequency - BW/2 + subcarrier_spacing * np.arange(num_subcarriers)  # array of frequencies 
    frequencies = subcarrier_frequencies(num_subcarriers, subcarrier_spacing) # Compute baseband frequencies of subcarriers relative to the carrier frequency
    print(f'subcarrier spacing = {subcarrier_spacing/1e6} MHz')

    # set scene frequency
    scene.frequency = 157.75e9 # Set frequency to fc 

    # Instantiate a path solver
    # The same path solver can be used with multiple scenes
    p_solver  = PathSolver()
    print(f'path solver loop mode: {p_solver.loop_mode}') #symbolic mode is the fastest! 
    
    for ue_idx in range(ds.dims['user']): # loop over all UE postions
        x, y, z = ds.x.values[ue_idx], ds.y.values[ue_idx], ds.z.values[ue_idx]
        print(f'type of x: {type(x)}')
        ue_pos = [float(x), float(y), float(z)]
        print(f'type of uepos: {type(ue_pos)}')

        # Create a receiver
        rx = Receiver(name=f"rx_{ue_idx}",
                    position=ue_pos,
                    display_radius=0.5)

        # Add receiver instance to scene
        scene.add(rx)

        # loop over all stripes
        for stripe_idx in range(N_stripes):
            # loop over all RUs in the stripe
            for RU_idx in range(N_RUs):
                print(f'Processing UE {ue_idx} stripe {stripe_idx} RU {RU_idx}...')
            
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

                # render scene with tx and rx
                if intermediate_reders:
                    print(f' rendering scene prior to path solver')
                    scene.render_to_file(camera=my_cam, filename=f'scene_with_stripe_{stripe_idx}_RU_{RU_idx}.png', 
                                        resolution=[650, 500], num_samples=512, clip_at=20) 


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



                # Compute channel frequency response
                h_freq = paths.cfr(frequencies=frequencies,
                                normalize=True, # Normalize energy
                                normalize_delays=True,
                                out_type="numpy")

                t2 = time.time()
                print(f"Time to compute paths and CFR: {t2-t1:.2f} seconds")
                # Shape: [num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, num_subcarriers]
                print("Shape of h_freq: ", h_freq.shape)
                print(f'type of h_freq: {h_freq.dtype}')

                # todo store values of CFR

                # remove tx from the scene after computation
                scene.remove(f"tx_stripe_{stripe_idx}_RU_{RU_idx}")

                # render scene with tx and rx
                if intermediate_reders:
                    print(f' rendering scene after removing tx')
                    scene.render_to_file(camera=my_cam, filename=f'scene_tx_removed_stripe_{stripe_idx}_RU_{RU_idx}.png', 
                                         resolution=[650, 500], num_samples=512, clip_at=20) 

                # todo check if GPU memory is freed

        # logging
        print(f'Finished processing UE {ue_idx} with all stripes and RUs.')

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