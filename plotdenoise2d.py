import numpy as np
import os, sys
import healpy as hp
import matplotlib.pyplot as plt
import getopt

def usage():
    print(' This software plots the demo results:')
    print('>python plotdemo2d.py -n=8 [-c|cov] [-o|--out] [-c|--cmap]')
    print('-n : is the nside of the input map (nside max = 256 with the default map)')
    print('--cov     (optional): use scat_cov instead of scat')
    print('--out     (optional): If not specified save in *_demo_*.')
    print('--map=jet (optional): If not specified use cmap=jet')
    exit(0)
    
def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "n:co:m:", ["nside", "cov","out","map"])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    cov=False
    nside=-1
    outname='demo'
    cmap='viridis'
    
    for o, a in opts:
        if o in ("-c","--cov"):
            cov = True
        elif o in ("-m","--map"):
            cmap=a[1:]
        elif o in ("-n", "--nside"):
            nside=int(a[1:])
        elif o in ("-o", "--out"):
            outname=a[1:]
        else:
            print(o,a)
            assert False, "unhandled option"

    
    log=np.load('out2d_%s_log_%d.npy'%(outname,nside))
    
    plt.figure(figsize=(6,6))
    plt.plot(np.arange(log.shape[0])+1,log,color='black')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('Loss')
    plt.xlabel('Number of iteration')
    
    im=np.load('in2d_%s_map_%d.npy'%(outname,nside))
    omap=np.load('out2d_%s_map_%d.npy'%(outname,nside))

    tf1=abs(np.fft.fft2(im))
    tf2=abs(np.fft.fft2(omap))
    xs=nside//2
    plt.subplot(1,2,1)
    plt.imshow(np.log(np.roll(np.roll(tf1,-xs,0),-xs,1)),cmap='jet',vmin=-1,vmax=2) #,vmin=-0.03,vmax=0.03)
    plt.subplot(1,2,2)
    plt.imshow(np.log(np.roll(np.roll(tf2,-xs,0),-xs,1)),cmap='jet',vmin=-1,vmax=2) #,vmin=-0.03,vmax=0.03)
    plt.show()
    n=im.shape[0]
    amp=im.std()*3
    plt.figure(figsize=(16,5))
    plt.subplot(1,3,1)
    plt.imshow(im,cmap=cmap,origin='lower',aspect='auto',vmin=-amp,vmax=amp)
    plt.title('Input')
    plt.subplot(1,3,2)
    plt.imshow(omap,cmap=cmap,origin='lower',aspect='auto',vmin=-amp,vmax=amp)
    plt.title('Ouput')
    plt.subplot(1,3,3)
    plt.imshow(im-omap,cmap=cmap,origin='lower',aspect='auto',vmin=-amp,vmax=amp)
    plt.title('difference')
    plt.show()

if __name__ == "__main__":
    main()
