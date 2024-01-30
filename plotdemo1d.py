import numpy as np
import os, sys
import healpy as hp
import matplotlib.pyplot as plt
import getopt

def usage():
    print(' This software plots the demo results:')
    print('>python plotdemo2d.py -n=8 [-c|cov] [-o|--out] [-c|--cmap] [-g|--geo]')
    print('-n : is the nside of the input map (nside max = 256 with the default map)')
    print('--cov     (optional): use scat_cov instead of scat')
    print('--out     (optional): If not specified save in *_demo_*.')
    print('--map=jet (optional): If not specified use cmap=jet')
    print('--geo     (optional): If specified use cartview')
    print('--vmin|-i    (optional): specify the minimum value')
    print('--vmax|-a    (optional): specify the maximum value')
    print('--path  (optional): Define the path where output file are written (default data)')
    exit(0)
    
def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "n:co:m:gi:a:", ["nside", "cov","out","map","geo","vmin","vmax","path"])
        
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    cov=False
    nside=-1
    outname='demo'
    cmap='jet'
    docart=False
    vmin=-3
    vmax=3
    outpath='data/'
    
    for o, a in opts:
        if o in ("-c","--cov"):
            cov = True
        elif o in ("-g","--geo"):
            docart=True
        elif o in ("-m","--map"):
            cmap=a[1:]
        elif o in ("-n", "--nside"):
            nside=int(a[1:])
        elif o in ("-o", "--out"):
            outname=a[1:]
        elif o in ("-i", "--vmin"):
            vmin=float(a[1:])
        elif o in ("-a", "--vmax"):
            vmax=float(a[1:])
        elif o in ("-p", "--path"):
            outpath=a[1:]
        else:
            print(o,a)
            assert False, "unhandled option"

    if nside<2 or nside!=2**(int(np.log(nside)/np.log(2))):
        print('nside should be a pwer of 2 and in [2,...,256]')
        exit(0)

    print('Work with nside=%d'%(nside))

    if cov:
        import foscat.scat_cov1D as sc
    else:
        import foscat.scat1D as sc

    print('Save ',outpath+'out1d_%s_%d'%(outname,nside))
    refX  = sc.read(outpath+'in1d_%s_%d'%(outname,nside))
    start = sc.read(outpath+'st1d_%s_%d'%(outname,nside))
    out   = sc.read(outpath+'out1d_%s_%d'%(outname,nside))

    refX.plot(name='Model',lw=6)
    start.plot(name='Input',color='orange',hold=False)
    out.plot(name='Output',color='red',hold=False)
    #(refX-out).plot(name='Diff',color='purple',hold=False)

    im = np.load(outpath+'in1d_%s_map_%d.npy'%(outname,nside))
    om = np.load(outpath+'out1d_%s_map_%d.npy'%(outname,nside))

    n=im.shape[0]
    plt.figure(figsize=(10,6))
    plt.subplot(1,2,1)
    plt.plot(om,color='b',lw=0.5,label='Synthesised')
    plt.legend(frameon=False)
    plt.subplot(1,2,2)
    plt.plot(im,color='b',lw=0.5,label='Input Data')
    plt.legend(frameon=False)
    plt.show()

if __name__ == "__main__":
    main()
