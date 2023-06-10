
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import healpy as hp
import getopt

#=================================================================================
# INITIALIZE FoCUS class
#=================================================================================
import foscat.Synthesis as synthe

s2rs1=[0.0,0.1,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]    

def usage():
    print(' This software is a demo of the foscat library:')
    print('>python demo.py -n=8 [-c|--cov][-s|--steps=3000][-S=1234|--seed=1234][-x|--xstat] [-g|--gauss][-k|--k5x5][-d|--data][-o|--out][-K|--k128][-r|--orient] [-p|--path] [-r|rmask][-b|--batch][-l|--nsim][-v|--vsim]')
    print('-n : is the nside of the input map (nside max = 256 with the default map)')
    print('--cov (optional): use scat_cov instead of scat.')
    print('--steps (optional): number of iteration, if not specified 30 (use all available noise x30).')
    print('--seed  (optional): rank of the noise used for simulation and rank+1 will be used for input data.')
    print('--xstat (optional): work with cross statistics.')
    print('--path  (optional): Define the path where output file are written (default data)')
    print('--k5x5  (optional): Work with a 5x5 kernel instead of a 3x3.')
    print('--out   (optional): If not specified save in *_demo_*.')
    print('--orient(optional): If not specified use 4 orientation')
    print('--batch (optional): number of available batch (default 100)')
    exit(0)

# function that generate map with the proper powerspectrum for each mask from noisy map
    
def align(im,imq,imu,mask):
    nside=int(np.sqrt(im.shape[1]//12))
    idx=hp.ring2nest(nside,np.arange(12*nside**2))
    idx2=hp.nest2ring(nside,np.arange(12*nside**2))
    l,m=hp.Alm.getlm(lmax=3*nside-1)
    imap=0*im

    dmask=np.sum(mask,0)

    for k in range(im.shape[0]):
        for i in range(mask.shape[0]):
            cl=hp.anafast((mask[i]/dmask*im[k])[idx])
            if k==0:
                clr=hp.anafast((mask[i]/dmask*imq)[idx])
            else:
                clr=hp.anafast((mask[i]/dmask*imu)[idx])
            tf=np.sqrt(clr/cl)
            tf[0]=1.0
            alm=hp.map2alm((mask[i]/dmask*im[k])[idx])
            imap[k]=imap[k]+hp.alm2map(alm*tf[l],nside)[idx2]
            
    return(imap)
    
def computespectromap(itmp,mask,lmin=90,loff=10):
    cl={}
    imap=(mask[0]-mask[1])*itmp
    nside=int(np.sqrt(itmp.shape[1]//12))
    idx=hp.ring2nest(nside,np.arange(12*nside**2))
    idx2=hp.nest2ring(nside,np.arange(12*nside**2))
    l,m=hp.Alm.getlm(lmax=3*nside-1)
    for k in range(itmp.shape[0]):
        for i in range(1,mask.shape[0]):
            if i<mask.shape[0]-1:
                dmask=mask[i]-mask[i+1]
            else:
                dmask=mask[i]
            cl=hp.anafast(((dmask)*itmp[k])[idx],map2=((dmask)*itmp[k])[idx])
            a=np.polyfit(np.log(np.arange(lmin-loff)+loff),np.log(cl[loff:lmin]),1)
            clmod=np.exp(a[1]+a[0]*np.log(np.arange(nside*3)))
            clmod[0]=0.0
            tf=np.sqrt(clmod/cl)
            tf[0:lmin]=1.0
            alm=hp.map2alm(((dmask)*itmp[k])[idx])
            imap[k]=imap[k]+hp.alm2map(alm*tf[l],nside)[idx2]
    return imap


def main():
    test_mpi=False
    for ienv in os.environ:
        if 'OMPI_' in ienv:
            test_mpi=True
        if 'PMI_' in ienv:
            test_mpi=True

    size=1
    
    if test_mpi:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

    if size>1:
        print('Use mpi facilities',rank,size)
        isMPI=True
    else:
        size=1
        rank=0
        isMPI=False
    try:
        opts, args = getopt.getopt(sys.argv[1:], "n:cS:s:ko:r:b:l:vb:", \
                                   ["nside", "cov","seed","steps","k5x5","out","orient","batch","nsim","vsim","bstep"])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    cov=False
    nside=-1
    nstep=30
    KERNELSZ=3
    seed=0
    outname='demo'
    outpath='results/'
    instep=16
    norient=4
    nnoise=1
    nsim=100
    bstep=1
    dosim=False
    
    for o, a in opts:
        print(o,a)
        if o in ("-c","--cov"):
            cov = True
        elif o in ("-v","--vsim"):
            dosim = True
        elif o in ("-b","--bstep"):
            bstep=int(a[1:])
            print('bstep = ',bstep)
        elif o in ("-n", "--nside"):
            nside=int(a[1:])
        elif o in ("-s", "--steps"):
            nstep=int(a[1:])
        elif o in ("-S", "--seed"):
            seed=int(a[1:])
        elif o in ("-b", "--batch"):
            nnoise=int(a[1:])
            print('Size of batch = ',nnoise)
        elif o in ("-l", "--nsim"):
            nsim=int(a[1:])
            print('Number of SIMs = ',nsim)
            nsim=nsim+2
        elif o in ("-o", "--out"):
            outname=a[1:]
            print('Save data in ',outname)
        elif o in ("-k", "--k5x5"):
            KERNELSZ=5
        elif o in ("-r", "--orient"):
            norient=int(a[1:])
            print('Use %d orientations'%(norient))
        else:
            assert False, "unhandled option"

    print('Use SEED = ',seed,' Converted to ',seed%(nsim-2))
    seed=seed%(nsim-2)
    if nside<2 or nside!=2**(int(np.log(nside)/np.log(2))) or (nside>256 and KERNELSZ<=5) or (nside>2**instep and KERNELSZ>5) :
        print('nside should be a power of 2 and in [2,...,256] ')
        usage()
        exit(0)

    print('Work with nside=%d'%(nside))
    sys.stdout.flush()

    if cov:
        import foscat.scat_cov as sc
        print('Work with ScatCov')
    else:
        import foscat.scat as sc
        print('Work with Scat')
    sys.stdout.flush()
        
    #=================================================================================
    # DEFINE A PATH FOR scratch data
    # The data are storred using a default nside to minimize the needed storage
    #=================================================================================
    scratch_path = 'data'

    #=================================================================================
    # Function to reduce the data used in the FoCUS algorithm 
    #=================================================================================
    def dodown(a,nout,axis=0):
        nin=int(np.sqrt(a.shape[axis]//12))
        
        if nin==nside:
            return(a)
        
        if axis==0:
            return(np.mean(a.reshape(12*nout*nout,(nin//nout)**2),1))
        if axis==1:
            return(np.mean(a.reshape(a.shape[0],12*nout*nout,(nin//nout)**2),2))

    # convert M=Q+jU to M=[Q,U]
    def toreal(a):
        b=np.concatenate([np.real(np.expand_dims(a,0)),np.imag(np.expand_dims(a,0))])
        return(b)

    def increasesmall(P0,amp=10):
        a=P0.numpy()
        a[:,0,:]=amp*a[:,0,:]
        return a

    #=================================================================================
    # Get data and convert from nside=256 to the choosen nside
    #=================================================================================
    # read data
    im=toreal(dodown(np.load('353psb_full.npy'),nside))
    im1=toreal(dodown(np.load('353psb_hm1.npy'),nside))
    im2=toreal(dodown(np.load('353psb_hm2.npy'),nside))

    mapT=dodown(np.load('map_857_256_nest.npy'),nside)
    
    if dosim:
        im[0]=np.sqrt(0.801)*dodown(np.load('/travail/jdelouis/heal_cnn/Q_vansingel_256.npy'),nside)
        im[1]=np.sqrt(0.801)*dodown(np.load('/travail/jdelouis/heal_cnn/U_vansingel_256.npy'),nside)
        im1=im.copy()
        im2=im.copy()

    # level of noise added to map (this is for testing for smaller nside)
    # at nside=64 5 is a good number for this demo
    ampnoise=1
    if dosim:
        if nside<32:
            ampnoise=100
        if nside==32:
            ampnoise=20
        if nside==64:
            ampnoise=10
        
    # read 100 noise simulation
    noise  = np.zeros([2,nsim,12*nside*nside])
    noise1 = np.zeros([2,nsim,12*nside*nside])
    noise2 = np.zeros([2,nsim,12*nside*nside])

    idx=hp.nest2ring(nside,np.arange(12*nside*nside))
    
    for i in range(nsim):
        for k in range(2):
            noise[k,i]  = ampnoise*1E6*hp.ud_grade(hp.read_map('/travail/jdelouis/DownGrade_256/JAN18r60_%03d_353psb_353psb_full_IQU.fits'%(i+1),k+1),nside)[idx]
            noise1[k,i] = ampnoise*1E6*hp.ud_grade(hp.read_map('/travail/jdelouis/DownGrade_256/JAN18r60_%03d_353psb_353psb_hm1_IQU.fits'%(i+1),k+1),nside)[idx]
            noise2[k,i] = ampnoise*1E6*hp.ud_grade(hp.read_map('/travail/jdelouis/DownGrade_256/JAN18r60_%03d_353psb_353psb_hm2_IQU.fits'%(i+1),k+1),nside)[idx]

    tab=['10','08','06','04']
    
    imask=np.ones([5,im.shape[1]])
    mask=np.ones([5,im.shape[1]])
    
    for i in range(4):
        imask[1+i,:]=dodown(np.load('/travail/jdelouis/heal_cnn/MASK_GAL%s_256.npy'%(tab[i])),nside)
    
    for i in range(4):
        mask[i,:]=imask[i]-imask[i+1]
        mask[i]/=mask[i].mean()

    mask[4]=imask[4]/imask[4].mean()
        
    #=================================================================================
    # Generate a random noise with the same coloured than the input data
    #=================================================================================
    
    imap=np.zeros([2,12*nside**2])
    imap1=np.zeros([2,12*nside**2])
    imap2=np.zeros([2,12*nside**2])

    if dosim==False:
        
        for k in range(2):
            imap[k]=im[k]
            imap1[k]=im1[k]
            imap2[k]=im2[k]
    else:
        for k in range(2):
            imap[k]=im[k]+noise[k,seed]
            imap1[k]=im[k]+noise1[k,seed]
            imap2[k]=im[k]+noise2[k,seed]
        noise[:,seed]=noise[:,-1]
        noise1[:,seed]=noise1[:,-1]
        noise2[:,seed]=noise2[:,-1]

    inoise=noise[:,seed+1]

    noise[:,seed+1]=noise[:,-2]
    noise1[:,seed+1]=noise1[:,-2]
    noise2[:,seed+1]=noise2[:,-2]

    noise=noise[:,0:-2]
    noise1=noise1[:,0:-2]
    noise2=noise2[:,0:-2]

    nsim=noise.shape[1]
    
    lam=1.2
    if KERNELSZ==5:
        lam=1.0

    l_slope=1.0
    r_format=True
    all_type='float64'
    
    #=================================================================================
    # COMPUTE THE WAVELET TRANSFORM OF THE REFERENCE MAP
    #=================================================================================
    scat_op=sc.funct(NORIENT=4,          # define the number of wavelet orientation
                     KERNELSZ=KERNELSZ,  # define the kernel size
                     OSTEP=0,            # get very large scale (nside=1)
                     LAMBDA=lam,
                     TEMPLATE_PATH=scratch_path,
                     slope=l_slope,
                     isMPI=isMPI,
                     gpupos=0,
                     use_R_format=r_format,
                     all_type=all_type,
                     mpi_size=size,
                     mpi_rank=rank,
                     nstep_max=instep)

    # map use to compute the sigma noise. In this example uses the input map
    """
    if dosim==False:
        imq=np.sqrt(0.801)*dodown(np.load('/travail/jdelouis/heal_cnn/Q_vansingel_256.npy'),nside)
        imu=np.sqrt(0.801)*dodown(np.load('/travail/jdelouis/heal_cnn/U_vansingel_256.npy'),nside)
        model_map=align(np.load('results/out_hwstB0_map_256.npy'),imq,imu,mask)
        np.save('ALIGNMAP.npy',model_map)
        model_map1=model_map
        model_map2=model_map
    else:
        model_map=im
        np.save('NOISEMAP.npy',model_map)
        model_map1=im
        model_map2=im
    """
    
    if dosim==False:
        model_map=imap
        model_map1=imap1
        model_map2=imap2
    else:
        model_map=imap
        model_map1=imap1
        model_map2=imap2

    init_map=np.zeros([2,12*nside**2])
    idx=hp.ring2nest(nside,np.arange(12*nside**2))
    idx2=hp.nest2ring(nside,np.arange(12*nside**2))
    for k in range(2):
        init_map[k,idx]=hp.smoothing(imap[k,idx],5.0/180.0*np.pi)+inoise[k,idx]-hp.smoothing(inoise[k,idx],5.0/180.0*np.pi)
    """
    hp.mollview(init_map[0],cmap='jet',norm='hist',nest=True)
    plt.show()
    exit(0)
    """
    
    #=================================================================================
    # DEFINE A LOSS FUNCTION AND THE SYNTHESIS
    #=================================================================================

    def update_bias(obias,bias,ratio=0.5):
        if obias is None:
            return ratio*bias
        else: 
            return (obias+ratio*(bias-obias))

    # the first loss function definition:
    # Loss = sum^{n_noise}_k { sum_s0,s1,s2,P00 {\frac{P(d_1,d_2)-P(x+n_k,1,x+n_k,2)}{\sigma_s0,s1,s2,p00}}}
    # where:
    # P(x,y) is the CWST of the map x and y that compute 4 coefficient sets (s_0,s_1,s_2,p00)
    # d_1,d_2 are the two half mission of the same map
    # x is the map to find
    # n_k,1,n_k,2 is the simulated k th noise respectively of the first and second half mission
    
    def loss(x,batch,scat_operator,args):
        ref = args[0]
        mask = args[1]
        i = args[2]

        bias = batch['bias']
        sig = batch['sig']

        tmp = scat_operator.eval(x[i],image2=x[i],mask=mask)
        
        learn = scat_operator.ldiff(sig,ref - bias - tmp)
        
        loss = scat_operator.reduce_mean(learn)
        
        return loss

    def comp_first_bias(y,off=1,deg=2):
        xx=1+np.arange(deg+2)
        yy=y[1:deg+3]
        yres=y.copy()
        idx=np.where(yy>0)[0]
        if len(idx)>deg:
            a=np.polyfit(xx[idx],np.log(yy[idx]),deg)
            yres[0]=np.exp(a[deg])
        return yres
    
    def batch_loss(data,istep,init=False):

        if init:
            sys.stdout.flush()
            m=data['m']
            m1=data['m1']
            m2=data['m2']
            k=data['k']
            ma=data['mask']
            noise1=data['noise1']
            noise2=data['noise2']
            nsim=data['nsim']

            # Compute reference spectra
            ref=scat_op.eval_fast(m1[k],image2=m2[k],mask=ma)
            """
            np.save('refS1_%s.npy'%(data['outname']),ref.S1.numpy())
            np.save('refP0_%s.npy'%(data['outname']),ref.P00.numpy())
            np.save('refS2L2_%s.npy'%(data['outname']),ref.S2.numpy())

            alls1=np.zeros([nsim,ref.S1.numpy().shape[0],ref.S1.numpy().shape[1],ref.S1.numpy().shape[2]])
            allp0=np.zeros([nsim,ref.S1.numpy().shape[0],ref.S1.numpy().shape[1],ref.S1.numpy().shape[2]])
            alls2=np.zeros([nsim,ref.S2.numpy().shape[0],ref.S2.numpy().shape[1],ref.S2.numpy().shape[2],ref.S2.numpy().shape[3]])
            """
            savv=None
            for i in range(nsim):
                basen=scat_op.eval_fast(m1[k]+noise1[k,i],image2=m2[k]+noise2[k,i],mask=ma)
                avv=basen-ref
                """
                allp0[i]=basen.P00.numpy()
                alls1[i]=basen.S1.numpy()
                alls2[i]=basen.S2.numpy()
                """
                if savv is None:
                    savv=avv
                    savv2=avv*avv
                else:
                    savv=savv+avv
                    savv2=savv2+avv*avv
            """
            np.save('noiseP0_%s.npy'%(data['outname']),allp0)
            np.save('noiseS1_%s.npy'%(data['outname']),alls1)
            np.save('noiseS2L2_%s.npy'%(data['outname']),alls2)
            exit(0)
            """
            savv=savv/(nsim)
            savv2=savv2/(nsim)

            bias=data['ref']-(data['ref']-savv).relu()
            bias=ref-(ref-savv).relu()
            
            data['res']={}
            data['res']['bias']=bias
            ref.save('TMP/MOD_%s_%d'%(data['outname'],data['Itt']))
            bias.save('TMP/BIAS_%s_%d'%(data['outname'],data['Itt']))
            data['ref'].save('TMP/REF_%s_%d'%(data['outname'],data['Itt']))
            sig=1/scat_op.sqrt(savv2-savv*savv)
            sig.P00=increasesmall(sig.P00)
            alpha=s2rs1[data['Itt']]
            print('INIT LOSS ',data['k'],data['Itt'],alpha)
            sig.S1=alpha*sig.S1
            sig.S2=alpha*sig.S2
            #sig.S1=5*sig.S1
            data['res']['sig']=sig

            data['Itt']=data['Itt']+1
        return data['res']

    def batch_loss_update(data,result):
        print('UPDATE LOSS DONE')
        sys.stdout.flush()
        data['m']=result
        data['m1']=result
        data['m2']=result


    # the first loss function definition:
    # Loss = sum^{n_noise}_k { sum_s0,s1,s2,P00 {\frac{P(d_1,d_2)-P(x+n_k,1,x+n_k,2)}{\sigma_s0,s1,s2,p00}}}
    # where:
    # P(x,y) is the CWST of the map x and y that compute 4 coefficient sets (s_0,s_1,s_2,p00)
    # d_1,d_2 are the two half mission of the same map
    # x is the map to find
    # n_k,1,n_k,2 is the simulated k th noise respectively of the first and second half mission
    
    def lossD(x,batch,scat_operator,args):
        mask = args[0]
        i = args[1]
        imap= args[2]
        ref= args[3]

        bias = batch['bias']
        bias_x = batch['bias_x']
        sig = batch['sig']

        #ref = scat_operator.eval(x[i],image2=x[i],mask=mask)
        tmp = scat_operator.eval(imap,image2=x[i],mask=mask)-bias
        
        learn = scat_operator.ldiff(sig,ref-bias_x - tmp)
        
        loss = scat_operator.reduce_mean(learn)
        
        return loss
    
    def batch_lossD(data,istep,init=False):

        if init:
            sys.stdout.flush()
            m=data['m']
            m1=data['m1']
            m2=data['m2']
            k=data['k']
            ma=data['mask']
            noise=data['noise']
            noise1=data['noise1']
            noise2=data['noise2']
            nsim=data['nsim']

            # Compute reference spectra
            ref=scat_op.eval_fast(m1[k],image2=m2[k],mask=ma)

            savv=None
            savv_x=None
            for i in range(nsim):
                basen=scat_op.eval_fast(m1[k]+noise[k,i],image2=m2[k],mask=ma)
                avv=basen-ref

                if savv is None:
                    savv=avv
                    savv2=avv*avv
                else:
                    savv=savv+avv
                    savv2=savv2+avv*avv

                basen_x=scat_op.eval_fast(m1[k]+noise1[k,i],image2=m2[k]+noise2[k,i],mask=ma)
                avv_x=basen_x-ref

                if savv_x is None:
                    savv_x=avv_x
                    savv2_x=avv_x*avv_x
                else:
                    savv_x=savv_x+avv_x
                    savv2_x=savv2_x+avv_x*avv_x
                    
            savv=savv/(nsim)
            savv2=savv2/(nsim)
            savv_x=savv_x/(nsim)
            savv2_x=savv2_x/(nsim)

            savv=ref-(ref-savv).relu()
            savv_x=data['ref']-(data['ref']-savv_x).relu()
            savv_x=ref-(ref-savv_x).relu()

            bias=savv
            data['res']={}
            data['res']['bias']=bias
            sig=1/scat_op.sqrt(savv2-savv*savv + savv2_x-savv_x*savv_x)

            sig.P00=increasesmall(sig.P00)
            alpha=s2rs1[data['Itt']]
            print('INIT LOSSD ',data['k'],data['Itt'],alpha)
            sig.S1=alpha*sig.S1
            sig.S2=alpha*sig.S2
      
            data['res']['sig']=sig
            data['res']['bias_x']=savv_x

            data['Itt']=data['Itt']+1
        return data['res']
        
    
    # the first loss function definition:
    # Loss = sum^{n_noise}_k { sum_s0,s1,s2,P00 {\frac{P(d_1,d_2)-P(x+n_k,1,x+n_k,2)}{\sigma_s0,s1,s2,p00}}}
    # where:
    # P(x,y) is the CWST of the map x and y that compute 4 coefficient sets (s_0,s_1,s_2,p00)
    # d_1,d_2 are the two half mission of the same map
    # x is the map to find
    # n_k,1,n_k,2 is the simulated k th noise respectively of the first and second half mission
    
    def lossT(x,batch,scat_operator,args):
        
        ref = args[0]
        mask = args[1]
        i = args[2]
        imapT= args[3]

        bias = batch['bias']
        sig = batch['sig']

        tmp = scat_operator.eval(imapT,image2=x[i],mask=mask)
        
        learn = scat_operator.ldiff(sig,ref - tmp)
        
        loss = scat_operator.reduce_mean(learn)
        
        return loss
    
    def batch_lossT(data,istep,init=False):

        if init:
            sys.stdout.flush()
            m=data['m']
            k=data['k']
            mapT=data['mapT']
            ma=data['mask']
            noise=data['noise']
            nsim=data['nsim']

            # Compute reference spectra
            ref=scat_op.eval_fast(mapT,image2=m[k],mask=ma)

            savv=None
            for i in range(nsim):
                basen=scat_op.eval_fast(mapT,image2=m[k]+noise[k,i],mask=ma)
                avv=basen-ref

                if savv is None:
                    savv=avv
                    savv2=avv*avv
                else:
                    savv=savv+avv
                    savv2=savv2+avv*avv

            savv=savv/(nsim)
            savv2=savv2/(nsim)

            if data['notcov']:
                savv2.S0=0*savv2.S0+1.0

            bias=savv

            data['res']={}
            data['res']['bias']=bias
            sig=1/scat_op.sqrt(savv2-savv*savv)
            sig.P00=increasesmall(sig.P00)
            alpha=s2rs1[data['Itt']]
            print('INIT LOSST ',data['k'],data['Itt'],alpha)
            sig.S1=alpha*sig.S1
            sig.S2=alpha*sig.S2
            data['res']['sig']=sig

            data['Itt']=data['Itt']+1
        return data['res']

    def batch_lossT_update(data,result):
        print('UPDATE LOSST DONE')
        sys.stdout.flush()
        data['m']=result

    # the cross loss function definition:
    # Loss = sum^{n_noise}_k { sum_s0,s1,s2,P00 {\frac{P(Q,U)-P(x[0]+n_{k,q},x[1]+n_{k,u})}{\sigma_s0,s1,s2,p00}}}
    # where:
    # P(x,y) is the CWST of the map x and y that compute 4 coefficient sets (s_0,s_1,s_2,p00)
    # Q,U are the two Q,U map 
    # x is the maps to find x[0] will be the clean Q map and x[1] is the clean U map
    # n_{k,q},n_{k,u} is the simulated k th noise respectively of the first and second half mission
    
    def lossX(x,batch,scat_operator,args):
        
        ref  = args[0]
        mask = args[1]

        bias = batch['bias']
        sig = batch['sig']
        
        tmp = scat_operator.eval(x[0],image2=x[1],mask=mask,Auto=False)

        learn = scat_operator.ldiff(sig,ref-bias -tmp)
        
        loss = scat_operator.reduce_mean(learn)
        
        return loss

    def batch_lossX(data,istep,init=False):

        if init:
            sys.stdout.flush()
            m1=data['m1']
            m2=data['m2']
            ma=data['mask']
            noise=data['noise']
            nsim=data['nsim']

            # Compute reference spectra
            ref=scat_op.eval_fast(m1[0],image2=m2[1],mask=ma)

            savv=None
            for i in range(nsim):
                basen=scat_op.eval_fast(m1[0]+noise[0,i],image2=m2[1]+noise[1,i],mask=ma)
                avv=basen-ref

                if savv is None:
                    savv=avv
                    savv2=avv*avv
                else:
                    savv=savv+avv
                    savv2=savv2+avv*avv

            savv=savv/(nsim)
            savv2=savv2/(nsim)

            bias=savv

            data['res']={}
            data['res']['bias']=bias
            sig=1/scat_op.sqrt(savv2-savv*savv)
            sig.P00=increasesmall(sig.P00)
            alpha=s2rs1[data['Itt']]
            print('INIT LOSSX ',data['Itt'],alpha)
            sig.S1=alpha*sig.S1
            sig.S2=alpha*sig.S2
            data['res']['sig']=sig

            data['Itt']=data['Itt']+1
        return data['res']

    def batch_lossX_update(data,result):
        data['m']=result
        data['m1']=result
        data['m2']=result
        print('SAVE TEMPORARY RESULTS')
        sys.stdout.flush()
        np.save(data['outpath'] +'in_%s%d_map_%d.npy'%(data['outname'] ,data['itt'],data['nside']),data['im'])
        np.save(data['outpath'] +'st_%s%d_map_%d.npy'%(data['outname'] ,data['itt'],data['nside']),data['imap'])
        np.save(data['outpath'] +'st1_%s%d_map_%d.npy'%(data['outname'],data['itt'],data['nside']),data['imap1'])
        np.save(data['outpath'] +'st2_%s%d_map_%d.npy'%(data['outname'],data['itt'],data['nside']),data['imap2'])
        np.save(data['outpath'] +'out_%s%d_map_%d.npy'%(data['outname'],data['itt'],data['nside']),result)
        data['itt']=data['itt']+1
        
    # the cross loss function definition:
    # Loss = sum^{n_noise}_k { sum_s0,s1,s2,P00 {\frac{P(Q,U)-P(x[0]+n_{k,q},x[1]+n_{k,u})}{\sigma_s0,s1,s2,p00}}}
    # where:
    # P(x,y) is the CWST of the map x and y that compute 4 coefficient sets (s_0,s_1,s_2,p00)
    # Q,U are the two Q,U map 
    # x is the maps to find x[0] will be the clean Q map and x[1] is the clean U map
    # n_{k,q},n_{k,u} is the simulated k th noise respectively of the first and second half mission
    
    def lossN(x,scat_operator,args):
        
        ref =args[0]
        mask = args[1]
        i = int(args[2])
        sig = args[3]
        imap= args[4]
        
        tmp = scat_operator.eval(imap-x[i],image2=imap-x[i],mask=mask)

        learn = scat_operator.ldiff(sig,ref-tmp)
        
        loss = scat_operator.reduce_mean(learn)
        
        return loss

    allsize=9
    
    # all mpi rank that are consistent with 0 are computing the loss for P(Q,U) ~ P(x[0]+n_q,x[1]+n_u)
    if rank%allsize==0%size:

        refX=scat_op.eval_fast(imap[0],image2=imap[1],Auto=False,mask=mask)

        infoX={}
        infoX['Itt']=0
        infoX['m']=model_map
        infoX['m1']=model_map1
        infoX['m2']=model_map2
        infoX['mask']=mask
        infoX['noise']=noise
        infoX['nsim']=nsim
        # information to save data at each itteration
        infoX['outname']=outname
        infoX['nside']=nside
        infoX['outpath']=outpath
        infoX['itt']=0
        infoX['im']=im
        infoX['imap']=imap
        infoX['imap1']=imap1
        infoX['imap2']=imap2
            
        loss1=synthe.Loss(lossX,scat_op,refX,mask,batch=batch_lossX,batch_data=infoX,batch_update=batch_lossX_update)

        # If parallel declare one synthesis function per mpi process
        if size>1:
            sy = synthe.Synthesis([loss1])

    loss2={}
    loss3={}
    loss4={}
    loss5={}

    info2={}
    infoD={}
    infoT={}

    for pol in range(2):

        if rank%allsize==(1+pol)%size:
            ref=scat_op.eval_fast(imap1[pol],image2=imap2[pol],mask=mask)
            info2[pol]={}
            info2[pol]['Itt']=0
            info2[pol]['m']=model_map
            info2[pol]['m1']=model_map1
            info2[pol]['m2']=model_map2
            info2[pol]['k']=pol
            info2[pol]['mask']=mask
            info2[pol]['noise1']=noise1
            info2[pol]['noise2']=noise2
            info2[pol]['outname']=outname
            info2[pol]['nsim']=nsim
            info2[pol]['ref']=ref

            loss2[pol]=synthe.Loss(loss,scat_op,ref,mask,pol,batch=batch_loss,batch_data=info2[pol],batch_update=batch_loss_update)

            # If parallel declare one synthesis function per mpi process
            if size>1:
                sy = synthe.Synthesis([loss2[pol]])

        if rank%allsize==(3+pol)%size:
            
            ref=scat_op.eval_fast(imap1[pol],image2=imap2[pol],mask=mask)

            infoD[pol]={}
            infoD[pol]['Itt']=0
            infoD[pol]['m']=model_map
            infoD[pol]['m1']=model_map1
            infoD[pol]['m2']=model_map2
            infoD[pol]['k']=pol
            infoD[pol]['mask']=mask
            infoD[pol]['noise']=noise
            infoD[pol]['noise1']=noise1
            infoD[pol]['noise2']=noise2
            infoD[pol]['nsim']=nsim
            infoD[pol]['ref']=ref

            loss3[pol]=synthe.Loss(lossD,scat_op,mask,pol,imap[pol],ref,
                                   batch=batch_lossD,batch_data=infoD[pol],batch_update=batch_loss_update)
            
            if size>1:
                sy = synthe.Synthesis([loss3[pol]])
        
        if rank%allsize==(5+pol)%size:

            infoT[pol]={}
            infoT[pol]['Itt']=0
            infoT[pol]['m']=model_map
            infoT[pol]['k']=pol
            infoT[pol]['mapT']=mapT
            infoT[pol]['mask']=mask
            infoT[pol]['noise']=noise
            infoT[pol]['nsim']=nsim
            infoT[pol]['notcov']=(cov==False)

            ref=scat_op.eval_fast(mapT,image2=imap[pol],mask=mask)
            
            loss4[pol]=synthe.Loss(lossT,scat_op,ref,mask,pol,mapT,batch=batch_lossT,batch_data=infoT[pol],batch_update=batch_lossT_update)
            
            if size>1:
                sy = synthe.Synthesis([loss4[pol]])
                
        """
        if rank%allsize==(7+pol)%size:

            # Compute sigma for each CWST coeffients using simulation

            basen=scat_op.eval_fast(noise[pol,0],image2=noise[pol,0],mask=mask)

            avv=basen
            savv=avv
            savv2=avv*avv
            for i in range(1,nsim):
                basen=scat_op.eval_fast(noise[pol,i],image2=noise[pol,i],mask=mask)
                avv=basen
                savv=savv+avv
                savv2=savv2+avv*avv
                
            savv=savv/(nsim)
            savv2=savv2/(nsim)

            if not cov:
                savv2=0*savv2+1.0

            sig=1/scat_op.sqrt(savv2-savv*savv)


            #sig8.P00=10*sig8.P00
            sig.P00=5*sig.P00
            sig.S1=5*sig.S1
            loss5[pol]=synthe.Loss(lossN,scat_op,savv,mask,pol,sig,scat_op.to_R(imap[pol]))
            
            if size>1:
                sy = synthe.Synthesis([loss5[pol]])
        """
    if size==1:
        sy = synthe.Synthesis([loss1,
                               loss2[0],loss2[1],
                               loss3[0],loss3[1],
                               loss4[0],loss4[1]])
        """,
                               loss5[0],loss5[1]])
        """

    #=================================================================================
    # RUN ON SYNTHESIS
    #=================================================================================
    if dosim:
        MESSAGE='SIMQU-'
    else:
        MESSAGE='HWSTQU-'

    omap=sy.run(init_map,
                EVAL_FREQUENCY = 10,
                NUM_EPOCHS = nstep,
                NUM_STEP_BIAS=bstep,
                SHOWGPU=False, #True,
                do_lbfgs=True,
                axis=1,
                MESSAGE=MESSAGE)

    #=================================================================================
    # STORE RESULTS
    #=================================================================================

    if rank==0%size:
        # save input data
        for ii in range(2):
            ref=scat_op.eval_fast(im[ii],mask=mask)
            start=scat_op.eval_fast(imap[ii],mask=mask)
            
            ref.save( outpath+'in_%s_%d_%d'%(outname,nside,ii))
            start.save(outpath+'st_%s_%d_%d'%(outname,nside,ii))
                
        for ii in range(2):
            out =scat_op.eval_fast(omap[ii],mask=mask)
            
            out.save(  outpath+'out_%s_%d_%d'%(outname,nside,ii))
            
            for k in range(10):
                out =scat_op.eval_fast(omap[ii]+noise[ii,k],mask=mask)
                out.save(outpath+'outn_%s_%d_%d_%d'%(outname,nside,ii,k))

        np.save(outpath+'in_%s_map_%d.npy'%(outname,nside),im)
        np.save(outpath+'mm_%s_map_%d.npy'%(outname,nside),mask[0])
        np.save(outpath+'st_%s_map_%d.npy'%(outname,nside),imap)
        np.save(outpath+'st1_%s_map_%d.npy'%(outname,nside),imap1)
        np.save(outpath+'st2_%s_map_%d.npy'%(outname,nside),imap2)
        np.save(outpath+'out_%s_map_%d.npy'%(outname,nside),omap)
        np.save(outpath+'out_%s_log_%d.npy'%(outname,nside),sy.get_history())
        
    # map use to compute the sigma noise. In this example uses the input map
    model_map=omap.copy()
    model_map1=omap.copy()
    model_map2=omap.copy()
    init_map=omap.copy()

    print('Computation Done')
    sys.stdout.flush()

if __name__ == "__main__":
    main()


    
