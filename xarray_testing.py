import numpy as np 
import xarray as xr

ds = xr.tutorial.load_dataset('air_temperature')

print(ds.air.data)







# eventual goal: 
RU_idx = 10
stripe_idx = 4
ur_loc = -1 #todo loaded from generated locations dataArray
nr_rx_ant = 16
nr_tx_ant = 16
nr_subcarriers = 128
H = np.ones((nr_rx_ant, nr_tx_ant, nr_subcarriers))

xr.Dataset()
