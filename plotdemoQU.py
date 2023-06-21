import numpy as np
import os, sys
import healpy as hp
import matplotlib.pyplot as plt
import getopt

def usage():
    print(' This software plots the demo results:')
    print('>python plotdemo.py -n=8 [-c|cov] [-o|--out] [-c|--cmap] [-g|--geo] [-i|--vmin] [-a|--vmax]')
    print('-n : is the nside of the input map (nside max = 256 with the default map)')
    print('--cov|-c     (optional): use scat_cov instead of scat')
    print('--out|-o     (optional): If not specified save in *_demo_*.')
    print('--map=jet|-m (optional): If not specified use cmap=jet')
    print('--geo|-g     (optional): If specified use cartview')
    print('--vmin|-i    (optional): specify the minimum value')
    print('--vmax|-a    (optional): specify the maximum value')
    exit(0)
    
def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "n:co:m:gi:a:", ["nside", "cov","out","map","geo","vmin","vmax"])
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
    outpath='results/'
    
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

                       
    def dodown(a,nout,axis=0):
        nin=int(np.sqrt(a.shape[axis]//12))
        
        if nin==nside:
            return(a)
        
        if axis==0:
            return(np.mean(a.reshape(12*nout*nout,(nin//nout)**2),1))
        if axis==1:
            return(np.mean(a.reshape(a.shape[0],12*nout*nout,(nin//nout)**2),2))

    tab=['10','08','06','04']
    mask=np.ones([4,12*nside*nside])

    for i in range(4):
        mask[i,:]=dodown(np.load('/travail/jdelouis/heal_cnn/MASK_GAL%s_256.npy'%(tab[i])),nside)


    im = np.load(outpath+'in_%s_map_%d.npy'%(outname,nside))
    try:
        mm = np.load(outpath+'mm_%s_map_%d.npy'%(outname,nside))
    except:
        mm = np.ones([im.shape[0]])
    sm = np.load(outpath+'st_%s_map_%d.npy'%(outname,nside))
    sm1 = np.load(outpath+'st1_%s_map_%d.npy'%(outname,nside))
    sm2 = np.load(outpath+'st2_%s_map_%d.npy'%(outname,nside))
    om = np.load(outpath+'out_%s_map_%d.npy'%(outname,nside))

    try:
        log= np.load(outpath+'out_%s_log_%d.npy'%(outname,nside))
        plt.figure(figsize=(6,6))
        plt.plot(np.arange(log.shape[0])+1,log,color='black')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Number of iteration')
        plt.ylabel('Loss')
    except:
        print('No log as been already saved')

    try:
        p=['Q','U']
        for ii in range(2):
            refX  = sc.read(outpath+'in_%s_%d_%d'%(outname,nside,ii))
            start = sc.read(outpath+'st_%s_%d_%d'%(outname,nside,ii))

            refX.plot(name='Model %s'%(p[ii]),lw=6)
            for k in range(10):
                out   = sc.read(outpath+'outn_%s_%d_%d_%d'%(outname,nside,ii,k))
                out.plot(name='Noise[%d] %s'%(k,p[ii]),color='gray',hold=False,legend=(k==0))

            out   = sc.read(outpath+'out_%s_%d_%d'%(outname,nside,ii))
            start.plot(name='Input %s'%(p[ii]),color='orange',hold=False,lw=2)
            out.plot(name='Output %s'%(p[ii]),color='red',hold=False,lw=2)
    except:
        #=================================================================================
        # COMPUTE THE WAVELET TRANSFORM OF THE REFERENCE MAP
        #=================================================================================
        scat_op=sc.funct(NORIENT=4,          # define the number of wavelet orientation
                         KERNELSZ=3,  # define the kernel size
                         OSTEP=0,            # get very large scale (nside=1)
                         LAMBDA=1.2,
                         TEMPLATE_PATH='data',
                         slope=1.0,
                         use_R_format=True,
                         all_type='float32')

        print('No scat has been saved yet, compute it')
        for i in range(2):
            refX = scat_op.eval_fast(im[i],image2=im[i],mask=mask)
            refX.plot(name='Model %s'%(p[ii]),lw=6)
            refX = scat_op.eval_fast(sm1[i],image2=sm2[i],mask=mask)
            refX.plot(name='Cross %s'%(p[ii]),color='purple',hold=False,lw=2)
            start = scat_op.eval_fast(sm[i],image2=sm[i],mask=mask)
            out = scat_op.eval_fast(om[i],image2=om[i],mask=mask)
            start.plot(name='Input %s'%(p[ii]),color='orange',hold=False,lw=2)
            out.plot(name='Output %s'%(p[ii]),color='red',hold=False,lw=2)
        
        
    print(im.shape,om.shape)

    idx=hp.ring2nest(nside,np.arange(12*nside**2))
    
    plt.figure(figsize=(6,6))
    if docart:
        hp.cartview(sm[0,idx],cmap=cmap,min=vmin,max=vmax,hold=False,sub=(3,2,1),nest=False,title='Input Q')
        hp.cartview(sm[1,idx],cmap=cmap,min=vmin,max=vmax,hold=False,sub=(3,2,2),nest=False,title='Input U')
        hp.cartview(om[0,:],cmap=cmap,min=vmin,max=vmax,hold=False,sub=(3,2,3),nest=True,title='Output Q')
        hp.cartview(om[1,:],cmap=cmap,min=vmin,max=vmax,hold=False,sub=(3,2,4),nest=True,title='Output U')
        hp.cartview(om[0,:]-im[0],cmap=cmap,min=vmin,max=vmax,hold=False,sub=(3,2,5),nest=True,title='Residu Q')
        hp.cartview(om[1,:]-im[1],cmap=cmap,min=vmin,max=vmax,hold=False,sub=(3,2,6),nest=True,title='Residu U')
    else:
        hp.mollview(sm[0,idx],cmap=cmap,min=vmin,max=vmax,hold=False,sub=(3,2,1),nest=False,title='Input Q')
        hp.mollview(sm[1,idx],cmap=cmap,min=vmin,max=vmax,hold=False,sub=(3,2,2),nest=False,title='Input U')
        hp.mollview(om[0,:],cmap=cmap,min=vmin,max=vmax,hold=False,sub=(3,2,3),nest=True,title='Output Q')
        hp.mollview(om[1,:],cmap=cmap,min=vmin,max=vmax,hold=False,sub=(3,2,4),nest=True,title='Output U')
        hp.mollview(om[0]-im[0]-np.median(om[0]-im[0]),cmap=cmap,min=vmin,max=vmax,hold=False,sub=(3,2,5),nest=True,title='Residu Q')
        hp.mollview(om[1]-im[1]-np.median(om[1]-im[1]),cmap=cmap,min=vmin,max=vmax,hold=False,sub=(3,2,6),nest=True,title='Residu U')

    plt.figure(figsize=(12,6))
    hp.gnomview(sm[0,idx],cmap=cmap,min=vmin,max=vmax,hold=False,sub=(1,3,1),nest=False,title='Input Q',rot=[-40,30],reso=8,xsize=400)
    hp.gnomview(om[0,idx],cmap=cmap,min=vmin,max=vmax,hold=False,sub=(1,3,2),nest=False,title='Output Q',rot=[-40,30],reso=8,xsize=400)
    hp.gnomview(om[0,idx]-im[0,idx],cmap=cmap,min=vmin,max=vmax,hold=False,sub=(1,3,3),nest=False,title='Diff Q',rot=[-40,30],reso=8,xsize=400)

    def avvcl(cl,dx=1.2):
        i1=2
        x=cl[0].copy()
        y=cl.copy()
        delta=1
        ii=0
        while i1+dx<x.shape[0]:
            x[ii]=i1+(delta-1)/2
            y[:,ii]=np.mean(cl[:,i1:int(i1+delta)],1)
            i1=int(i1+delta)
            delta*=dx
            ii+=1
        return x[0:ii],y[:,0:ii]
            
    def docl(data,mm,idx):
        aim=np.zeros([3,12*nside**2])
        aim[1]=mm[idx]*data[0,idx]
        aim[2]=mm[idx]*data[1,idx]
        aim[0]=np.sqrt(aim[1]**2+aim[2]**2)
        return hp.anafast(aim)
    
    def doclX(data1,data2,mm,idx):
        aim=np.zeros([3,12*nside**2])
        aim[1]=mm[idx]*data1[0,idx]
        aim[2]=mm[idx]*data1[1,idx]
        aim[0]=np.sqrt(aim[1]**2+aim[2]**2)
        aim2=np.zeros([3,12*nside**2])
        aim2[1]=mm[idx]*data2[0,idx]
        aim2[2]=mm[idx]*data2[1,idx]
        aim2[0]=np.sqrt(aim2[1]**2+aim2[2]**2)
        return hp.anafast(aim,map2=aim2)

    plt.figure(figsize=(6,8))
    for k in range(3):
        x1,clx=avvcl(doclX(sm1,sm2,mask[k],idx))
        x2,cli=avvcl(docl(im,mask[k],idx))
        x3,cls=avvcl(docl(sm,mask[k],idx))
        x4,clo=avvcl(docl(om,mask[k],idx))
        x5,cldiff=avvcl(docl(im-om,mask[k],idx))
        
        for ii in range(2):
            plt.subplot(3,2,1+ii+2*k)
            
            plt.plot(x1,clx[ii+1],color='grey',label=r'Cross')
            plt.plot(x2,cli[ii+1],color='black',label=r'Model $f_{sky}=%.2f$'%(mask[k].mean()),lw=6)
            plt.plot(x3,cls[ii+1],color='blue',label=r'Input')
            plt.plot(x4,clo[ii+1],color='orange',label=r'Output')
            plt.plot(x5,cldiff[ii+1],color='red',label=r'Diff')
            plt.yscale('log')
            plt.xscale('log')
            plt.legend()
            plt.xlabel('Multipoles')
            plt.ylabel('C(l)')
            plt.xlim(2,3*nside)
            
    plt.show()

if __name__ == "__main__":
    main()
