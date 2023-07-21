# FOSCAT_DEMO

The python scripts *demo.py* included in this package demonstrate how to use the foscat library to generate synthetic fields that have patterns with the same statistical properties as a specified image.

# Install foscat library

Before installing, make sure you have python installed in your enviroment.  
The last version of the foscat library can be installed using PyPi:
```
pip install foscat
```
Load the FOSCAT_DEMO package from github.
```
git clone https://github.com/jmdelouis/FOSCAT_DEMO.git
```


##  Recommended installing procedures  for mac users 
It is recomended to use python=3.9*.  
```
micromamba create -n FOSCAT
micromamba install -n FOSCAT ‘python==3.9*’
micromamba activate FOSCAT
pip install foscat
git clone https://github.com/jmdelouis/FOSCAT_DEMO.git

```

##  Recommended installing procedures  HPC users 
It is recomended to install tensorflow in advance.  For [DATARMOR](https://pcdm.ifremer.fr/Equipement)  for using GPU ; 

```
micromamba create -n FOSCAT
micromamba install -n FOSCAT ‘python==3.9*’
micromamba install -n FOSCAT ‘tensorflow==2.11.0’
micromamba activate FOSCAT
pip install foscat
git clone https://github.com/jmdelouis/FOSCAT_DEMO.git

```



# Spherical data example

## compute a synthetic image
```
python demo.py -n=32 -k -c -s=100
```
The *demo.py* script serves as a demonstration of the capabilities of the foscat library. It utilizes the Cross Wavelet Scattering Transform to generate a Healpix map that possesses the same characteristics as a specified input map. 
- ```-n=32``` computes map with nside=32.
- ```-k``` uses 5x5 kernel.
- ```-c``` uses Scattering Covariance.
- ```-l``` uses LBFGS minimizer.
- ```-s=100``` computes 100 steps. 
```
python demo.py -n=8 [-c|--cov][-s|--steps=3000][-S=1234|--seed=1234][-k|--k5x5][-d|--data][-o|--out][-r|--orient] [-p|--path][-a|--adam]

```
* The "-n" option specifies the nside of the input map. The maximum nside value is 256 with the default map.
* The "--cov" option (optional) uses scat_cov instead of scat.
* The "--steps" option (optional) specifies the number of iterations. If not specified, the default value is 1000.
* The "--seed" option (optional) specifies the seed of the random generator.
* The "--path" option (optional) allows you to define the path where the output files will be written. The default path is "data".
* The "--k5x5" option (optional) uses a 5x5 kernel instead of a 3x3.
* The "--data" option (optional) specifies the input data file to be used. If not specified, the default file "LSS_map_nside128.npy" will be used.
* The "--out" option (optional) specifies the output file name. If not specified, the output file will be saved in "demo".
* The "--orient" option (optional) specifies the number of orientations. If not specified, the default value is 4.
* The "--adam" option (optional) makes the synthesis using the ADAM optimizer instead of the L_BFGS.

## plot the result

The following script generates a series of plots that showcase different aspects of the synthesis process using the *demo.py* script.

```
python plotdemo.py -n=32 -c
```

# 2D field demo

# compute a synthetic turbulent field

The python scripts *demo2D.py* included in this package demonstrate how to use the foscat library to generate a 2D synthetic fields that have patterns with the same statistical properties as a specified 2D image. In this particular case, the input field is a sea surface temperature extracted from a north atlantic ocean simulation.

```
python demo2d.py -n=32 -k -c
```

# plot the result

The following script generates a series of plots that showcase different aspects of the synthesis process using the *demo2D.py* script.
```
python plotdemo2d.py -n=32 -c
```


