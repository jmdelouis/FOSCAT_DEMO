# FOSCAT_DEMO

This package provides scripts to show how to use the library foscat to compute syntheic field from one image.

# Install foscat library

The last version of the foscat library can be installed using PyPi:
```
pip install foscat
```

# Spherical field demo

## compute a synthetic field
```
python demo.py -n=32 -k -c
```
This software is a demo of the foscat library. It synthesises an healpix map with the same Cross Wavelet Scatering Tranform than an input map. 
```
python demo.py -n=8 [-c|--cov][-s|--steps=3000][-S=1234|--seed=1234][-x|--xstat] [-g|--gauss][-k|--k5x5][-d|--data][-o|--out][-K|--k128][-r|--orient] [-p|--path]

```
* -n : is the nside of the input map (nside max = 256 with the default map)
* --cov (optional): use scat_cov instead of scat.
* --steps (optional): number of iteration, if not specified 1000.
* --seed  (optional): seed of the random generator.
* --xstat (optional): work with cross statistics.
* --path  (optional): Define the path where output file are written (default data)
* --gauss (optional): convert Venus map in gaussian field.
* --k5x5  (optional): Work with a 5x5 kernel instead of a 3x3.
* --k128  (optional): Work with 128 pixel kernel reproducing wignercomputation instead of a 3x3.
* --data  (optional): If not specified use LSS_map_nside128.npy.
* --out   (optional): If not specified save in *_demo_*.
* --orient(optional): If not specified use 4 orientation

## plot the result
Using the next script a set of plots show various aspect of the synthesis. 

```
python plotdemo.py -n=32 -c
```

# 2D field demo

# compute a synthetic turbulent field 

```
python demo2D.py -n=32 -k -c
```

# plot the result
```
python plotdemo.py -n=32 -c
```


