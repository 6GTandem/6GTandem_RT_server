## Sionna Raytracing for 6GTandem


## Requirements
Python 3.11.12  
Tensorflow 2.15  
Sionna 1.0.2
Cuda 12.2   
(Compatibility depends on GPU => see: https://www.tensorflow.org/install/source )  

## Dataset folder structure

```dataset/
├── ue_locations/
│ └── ue_locations.nc
├── sub_thz_channels/
│ └── channels_thz_ue_0001.txt
│ └── channels_thz_ue_0002.txt
└── sub_10_ghz_channels/
  └── channels_ghz_ue_0001.txt
  └── channels_ghz_ue_0002.txt
```

## Polarization in Sionna
If we use a 2x2 array at the Tx and Rx we expect a channel tensor of size 4x4xQ (where Q is the number of subcarriers).
When we utilize dual polarization, we get an 8x8xQ tensor due to the double polarization at both the Rx and Tx.

When using 'VH' polarization in Sionna, the slant angles that will be applied to a vertically polarized antenna pattern are [0, $\pi$/2]. Hence, the resulting channel matrix for one subcarrier will have the following form:
<pre> H = [ H_VV (Co-Pol), H_VH (Cross-Pol) ]
          [ H_HV (Cross-Pol), H_HH (Co-Pol)]   </pre>

Hence, if you want to only use the co-polarized channel matrix you would use the upper left or lower right submatrices.

When using 'cross' polarization in Sionna the slant angles are [-π/4, π/4]. Hence, the resulting channel matrix for one subcarrier will have the following form:
<pre> H = [ H_-π/4, -π/4 (Cross-Pol), H_-π/4, π/4 (Co-Pol) ]
          [ H_π/4, -π/4 (Co-Pol), H_π/4, π/4 (Cross-Pol)]   </pre>
Due to the different angles, the transmitter and receiver are only co-polarized when one has angle $\pi/4$ and one has angle $-\pi/4$. Consequently, if you want to only use the co-polarized channel matrix you would use the upper right or lower left submatrices.

In this dataset we use cross polarized antennas hence we can only use the upper right or lowwer left submatrix if we want to consider co-polarized antennas.
These can be obtained as
<pre> H[4:, 0:4, 0]  or H[0:4, 4:, 0]  </pre>






