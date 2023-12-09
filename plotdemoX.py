import numpy as np
import os, sys
import healpy as hp
import matplotlib.pyplot as plt
import getopt

def usage():
    print(' This software plots the demo results:')
    print('>python plotdemo.py -n=8 [-c|cov] [-o|--out] [-c|--cmap] [-g|--geo] [-i|--vmin] [-a|--vmax] [-p|--path]')
    print('-n : is the nside of the input map (nside max = 256 with the default map)')
    print('--cov|-c     (optional): use scat_cov instead of scat')
    print('--out|-o     (optional): If not specified save in *_demo_*.')
    print('--map=jet|-m (optional): If not specified use cmap=jet')
    print('--geo|-g     (optional): If specified use cartview')
    print('--vmin|-i    (optional): specify the minimum value')
    print('--vmax|-a    (optional): specify the maximum value')
    print('--path|-p    (optional): Define the path where output file are written (default data)')
    exit(0)
    
def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "n:co:m:gi:a:p:", ["nside", "cov","out","map","geo","vmin","vmax","path"])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    cov=False
    nside=-1
    outname='demo'
    cmap='viridis'
    docart=False
    vmin=-300
    vmax= 300
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

    if nside<2 or nside!=2**(int(np.log(nside)/np.log(2))) or nside>256:
        print('nside should be a pwer of 2 and in [2,...,256]')
        exit(0)

    print('Work with nside=%d'%(nside))

    if cov:
        import foscat.scat_cov as sc
    else:
        import foscat.scat as sc

    
    log= np.load(outpath+'out_%s_log_%d.npy'%(outname,nside))
    plt.figure(figsize=(6,6))
    plt.plot(np.arange(log.shape[0])+1,log,color='black')
    plt.xscale('log')
    plt.yscale('log')

    ref   = sc.read(outpath+'in1_%s_%d'%(outname,nside))
    start = sc.read(outpath+'st1_%s_%d'%(outname,nside))
    out   = sc.read(outpath+'out1_%s_%d'%(outname,nside))
    
    ref.plot(name='Model 1',lw=6)
    start.plot(name='Input 1',color='orange',hold=False)
    out.plot(name='Output 1',color='red',hold=False)
    
    ref   = sc.read(outpath+'in2_%s_%d'%(outname,nside))
    start = sc.read(outpath+'st2_%s_%d'%(outname,nside))
    out   = sc.read(outpath+'out2_%s_%d'%(outname,nside))
    
    ref.plot(name='Model 2',lw=6)
    start.plot(name='Input 2',color='orange',hold=False)
    out.plot(name='Output 2',color='red',hold=False)

    im = np.load(outpath+'in_%s_map_%d.npy'%(outname,nside))
    try:
        mm = np.load(outpath+'mm_%s_map_%d.npy'%(outname,nside))
    except:
        mm = np.ones([im.shape[0]])
    sm = np.load(outpath+'st_%s_map_%d.npy'%(outname,nside))
    om = np.load(outpath+'out_%s_map_%d.npy'%(outname,nside))

    idx=hp.ring2nest(nside,np.arange(12*nside**2))

    for k in range(2):
        plt.figure(figsize=(6,6))
        if om.dtype!='complex64' and om.dtype!='complex128':
            if docart:
                hp.cartview(im[k,idx],cmap=cmap,min=vmin,max=vmax,hold=False,sub=(2,1,1),nest=False,title='Model')
                hp.cartview(om[k],cmap=cmap,min=vmin,max=vmax,hold=False,sub=(2,1,2),nest=True,title='Output')
            else:
                hp.mollview(im[k,idx],cmap=cmap,min=vmin,max=vmax,hold=False,sub=(2,1,1),nest=False,title='Model')
                hp.mollview(om[k],cmap=cmap,min=vmin,max=vmax,hold=False,sub=(2,1,2),nest=True,title='Output')
        else:
            if docart:
                hp.cartview(im[k,idx].real,cmap=cmap,min=vmin,max=vmax,hold=False,sub=(2,2,1),nest=False,title='Model')
                hp.cartview(im[k,idx].imag,cmap=cmap,min=vmin,max=vmax,hold=False,sub=(2,2,2),nest=False,title='Model')
                hp.cartview(om[k].real,cmap=cmap,min=vmin,max=vmax,hold=False,sub=(2,2,3),nest=True,title='Output')
                hp.cartview(om[k].imag,cmap=cmap,min=vmin,max=vmax,hold=False,sub=(2,2,4),nest=True,title='Output')
            else:
                hp.mollview(im[k,idx].real,cmap=cmap,min=vmin,max=vmax,hold=False,sub=(2,2,1),nest=False,title='Model')
                hp.mollview(im[k,idx].imag,cmap=cmap,min=vmin,max=vmax,hold=False,sub=(2,2,2),nest=False,title='Model')
                hp.mollview(om[k].real,cmap=cmap,min=vmin,max=vmax,hold=False,sub=(2,2,3),nest=True,title='Output')
                hp.mollview(om[k].imag,cmap=cmap,min=vmin,max=vmax,hold=False,sub=(2,2,4),nest=True,title='Output')

        cli=hp.anafast((mm[0]*(im[k]-np.median(im[k])))[idx])
        clo=hp.anafast((mm[0]*(om[k]-np.median(om[k])))[idx])
        cldiff=hp.anafast((mm[0]*(im[k]-om[k]-np.median(om[k])))[idx])

        plt.figure(figsize=(6,6))
        plt.plot(cli,color='blue',label=r'Model',lw=6)
        plt.plot(clo,color='orange',label=r'Output')
        plt.plot(cldiff,color='red',label=r'Diff')
        plt.yscale('log')
        plt.xscale('log')
        plt.legend()
        plt.xlabel('Multipoles')
        plt.ylabel('C(l)')
    plt.show()

if __name__ == "__main__":
    main()
