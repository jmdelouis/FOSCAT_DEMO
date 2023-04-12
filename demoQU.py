import numpy as np
import os, sys
import matplotlib.pyplot as plt
import healpy as hp
import getopt

#=================================================================================
# INITIALIZE FoCUS class
#=================================================================================
import foscat.Synthesis as synthe

    
def usage():
    print(' This software is a demo of the foscat library:')
    print('>python demo.py -n=8 [-c|--cov][-s|--steps=3000][-S=1234|--seed=1234][-x|--xstat] [-g|--gauss][-k|--k5x5][-d|--data][-o|--out][-K|--k128][-r|--orient] [-p|--path] [-r|rmask]')
    print('-n : is the nside of the input map (nside max = 256 with the default map)')
    print('--cov (optional): use scat_cov instead of scat.')
    print('--steps (optional): number of iteration, if not specified 1000.')
    print('--seed  (optional): seed of the random generator.')
    print('--xstat (optional): work with cross statistics.')
    print('--path  (optional): Define the path where output file are written (default data)')
    print('--k5x5  (optional): Work with a 5x5 kernel instead of a 3x3.')
    print('--out   (optional): If not specified save in *_demo_*.')
    print('--orient(optional): If not specified use 4 orientation')
    exit(0)
    
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
        opts, args = getopt.getopt(sys.argv[1:], "n:cS:s:ko:r:", \
                                   ["nside", "cov","seed","steps","k5x5","out","orient"])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    cov=False
    nside=-1
    nstep=1000
    KERNELSZ=3
    seed=1234
    outname='demo'
    outpath='results/'
    instep=16
    norient=4
    
    for o, a in opts:
        if o in ("-c","--cov"):
            cov = True
        elif o in ("-n", "--nside"):
            nside=int(a[1:])
        elif o in ("-s", "--steps"):
            nstep=int(a[1:])
        elif o in ("-S", "--seed"):
            seed=int(a[1:])
            print('Use SEED = ',seed)
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

    if nside<2 or nside!=2**(int(np.log(nside)/np.log(2))) or (nside>256 and KERNELSZ<=5) or (nside>2**instep and KERNELSZ>5) :
        print('nside should be a power of 2 and in [2,...,256] ')
        usage()
        exit(0)

    print('Work with nside=%d'%(nside))

    if cov:
        import foscat.scat_cov as sc
        print('Work with ScatCov')
    else:
        import foscat.scat as sc
        print('Work with Scat')
        
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
    
    #=================================================================================
    # Get data and convert from nside=256 to the choosen nside
    #=================================================================================
    # read data
    im=toreal(dodown(np.load('353psb_full.npy'),nside))
    im1=toreal(dodown(np.load('353psb_hm1.npy'),nside))
    im2=toreal(dodown(np.load('353psb_hm2.npy'),nside))
    if nside<=64:
        im1=im.copy()
        im2=im.copy()

    # read 100 noise simulation
    noise=toreal(dodown(np.load('noise_353psb_full.npy'),nside,axis=1))
    noise1=toreal(dodown(np.load('noise_353psb_hm1.npy'),nside,axis=1))
    noise2=toreal(dodown(np.load('noise_353psb_hm2.npy'),nside,axis=1))

    tab=['10','08','06','04']
    mask=np.ones([5,im.shape[1]])

    for i in range(4):
        mask[1+i,:]=dodown(np.load('MASK_GAL%s_256.npy'%(tab[i])),nside)
    
    #=================================================================================
    # Generate a random noise with the same coloured than the input data
    #=================================================================================
        
    np.random.seed(seed)

    # level of noise added to map (this is for testing for smaller nside)
    # at nside=64 5 is a good number for this demo
    ampnoise=0
    if nside<32:
        ampnoise=100
    if nside==32:
        ampnoise=20
    if nside==64:
        ampnoise=10
    
    imap=np.zeros([2,12*nside**2])
    imap[0]=im[0]+ampnoise*noise[0,0]
    imap[1]=im[1]+ampnoise*noise[1,0]
    imap1=np.zeros([2,12*nside**2])
    imap1[0]=im1[0]+ampnoise*noise1[0,0]
    imap1[1]=im1[1]+ampnoise*noise1[1,0]
    imap2=np.zeros([2,12*nside**2])
    imap2[0]=im2[0]+ampnoise*noise2[0,0]
    imap2[1]=im2[1]+ampnoise*noise2[1,0]

    if nside>64:
        ampnoise=1
        
    noise=noise[:,1:]
    noise1=noise1[:,1:]
    noise2=noise2[:,1:]
    
    nsim=noise.shape[1]
    nnoise = 4
    
    lam=1.2
    if KERNELSZ==5:
        lam=1.0

    l_slope=1.0
    r_format=True
    all_type='float32'
    
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
                     use_R_format=r_format,
                     all_type=all_type,
                     mpi_size=size,
                     mpi_rank=rank,
                     nstep_max=instep)
    
    # save input data
    if rank==0%size:
        for ii in range(2):
            ref=scat_op.eval(im[ii],mask=mask)
            start=scat_op.eval(imap[ii],mask=mask)
            
            ref.save( outpath+'in_%s_%d_%d'%(outname,nside,ii))
            start.save(outpath+'st_%s_%d_%d'%(outname,nside,ii))
            

    # map use to compute the sigma noise. In this example uses the input map
    model_map=imap.copy()
    model_map1=imap1.copy()
    model_map2=imap2.copy()
    
    #=================================================================================
    # DEFINE A LOSS FUNCTION AND THE SYNTHESIS
    #=================================================================================

    # call back function that compute the data use by the loss function.
    # here, mainly the noise to be added to the statistic.
    # *idx* is a table of index of the selected batch for the curent iteration computation
    
    def batch_loss(data,idx):
        nx,ny,nz=data[0].shape
        nsim=len(data)//2
        res=np.zeros([len(idx),2,nx,ny,nz],dtype=all_type)
        
        for i in range(len(idx)):
            res[i,0]= data[idx[i]].get()
            res[i,1]= data[idx[i]+nsim].get()
            
        return res

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
        sig = args[3]

        n=batch.shape[0]
        
        print('Number of noise ',n)
        tmp = scat_operator.eval(x[i]+batch[0,0],image2=x[i]+batch[0,1],mask=mask)
        learn = scat_operator.ldiff(sig,ref - tmp)
        for k in range(1,n):
            learn = learn + scat_operator.ldiff(sig,ref - scat_operator.eval(x[i]+batch[k,0],image2=x[i]+batch[k,1],mask=mask))

        learn=learn/n

        loss = scat_operator.reduce_mean(learn)
        
        return loss
    
    # the cross loss function definition:
    # Loss = sum^{n_noise}_k { sum_s0,s1,s2,P00 {\frac{P(Q,U)-P(x[0]+n_{k,q},x[1]+n_{k,u})}{\sigma_s0,s1,s2,p00}}}
    # where:
    # P(x,y) is the CWST of the map x and y that compute 4 coefficient sets (s_0,s_1,s_2,p00)
    # Q,U are the two Q,U map 
    # x is the maps to find x[0] will be the clean Q map and x[1] is the clean U map
    # n_{k,q},n_{k,u} is the simulated k th noise respectively of the first and second half mission
    
    def lossX(x,batch,scat_operator,args):
        
        ref = args[0]
        mask = args[1]
        sig = args[2]
        
        n=batch.shape[0]
        
        print('Number of noise ',n)
        
        learn = scat_operator.ldiff(sig,ref - scat_operator.eval(x[0]+batch[0,0],image2=x[1]+batch[0,1],mask=mask,Auto=False))
        
        for k in range(1,n):
            learn = learn + scat_operator.ldiff(sig,ref -scat_operator.eval(x[0]+batch[k,0],image2=x[1]+batch[k,1],mask=mask,Auto=False))

        learn = learn/n

        loss = scat_operator.reduce_mean(learn)
        return loss

    # all mpi rank that are consistent with 0 are computing the loss for P(Q,U) ~ P(x[0]+n_q,x[1]+n_u)
    if rank%3==2%size:
        # stores simulated noise to give to the synthesis
        Rnoise={}
        for k in range(nsim):
            Rnoise[k]=scat_op.backend.bk_cast(scat_op.to_R(ampnoise*noise[0,k]))
            Rnoise[k+nsim]=scat_op.backend.bk_cast(scat_op.to_R(ampnoise*noise[1,k]))
            
        # Compute reference spectra
        refX=scat_op.eval(imap[0],image2=imap[1],Auto=False,mask=mask)
        
        # Compute sigma for each CWST coeffients using simulation
        basen=scat_op.eval(model_map[0]+ampnoise*noise[0,0],image2=model_map[1]+ampnoise*noise[1,0],mask=mask)
        avv=basen-refX
        savv=avv
        savv2=avv*avv
        for i in range(1,nsim):
            basen=scat_op.eval(model_map[0]+ampnoise*noise[0,i],image2=model_map[1]+ampnoise*noise[1,i],mask=mask)
            avv=(basen-refX)
            savv=savv+avv
            savv2=savv2+avv*avv
            
        savv=savv/(nsim)
        savv2=savv2/(nsim)
        sig1=1/scat_op.sqrt(savv2-savv*savv)
        
        # create a loss class to declare it to the synthesis function
        loss1=synthe.Loss(lossX,scat_op,refX,mask,sig1,batch=batch_loss,batch_data=Rnoise)
        
        # If parallel declare one synthesis function per mpi process
        if size>1:
            sy = synthe.Synthesis([loss1])
            
    if rank%3==0%size:
        
        # stores simulated noise to give to the synthesis
        Rnoise={}
        for k in range(nsim):
            Rnoise[k]=scat_op.backend.bk_cast(scat_op.to_R(ampnoise*noise1[0,k]))
            Rnoise[k+nsim]=scat_op.backend.bk_cast(scat_op.to_R(ampnoise*noise2[0,k]))
            
        refR=scat_op.eval(imap1[0],image2=imap2[0],mask=mask)
        
        # Compute sigma for each CWST coeffients using simulation
        basen=scat_op.eval(model_map1[0]+ampnoise*noise1[0,0],image2=model_map2[0]+ampnoise*noise2[0,0],mask=mask)
        avv=basen-refR
        savv=avv
        savv2=avv*avv
        for i in range(1,nsim):
            basen=scat_op.eval(model_map1[0]+ampnoise*noise1[0,i],image2=model_map2[0]+ampnoise*noise2[0,i],mask=mask)
            avv=(basen-refR)
            savv=savv+avv
            savv2=savv2+avv*avv

        savv=savv/(nsim)
        savv2=savv2/(nsim)
        sig2=1/scat_op.sqrt(savv2-savv*savv)
        """
        refR.plot(name='data+noise')
        base=scat_op.eval(im1[0],image2=im2[0],mask=mask)
        base.plot(color='red',name='data',hold=False)
        (refR-savv).plot(name='debiased',color='black',hold=False)
        plt.show()
        exit(0)
        """
        # create a loss class to declare it to the synthesis function
        loss2=synthe.Loss(loss,scat_op,refR,mask,0,sig2,batch=batch_loss,batch_data=Rnoise)
        
        # If parallel declare one synthesis function per mpi process
        if size>1:
            sy = synthe.Synthesis([loss2])
        
    if rank%3==1%size:
        # stores simulated noise to give to the synthesis
        Rnoise={}
        for k in range(nsim):
            Rnoise[k]=scat_op.backend.bk_cast(scat_op.to_R(ampnoise*noise1[1,k]))
            Rnoise[k+nsim]=scat_op.backend.bk_cast(scat_op.to_R(ampnoise*noise2[1,k]))
            
        refI=scat_op.eval(imap1[1],image2=imap2[1],mask=mask)
        
        # Compute sigma for each CWST coeffients using simulation
            
        basen=scat_op.eval(model_map1[1]+ampnoise*noise1[1,0],image2=model_map2[1]+ampnoise*noise2[1,0],mask=mask)
        
        avv=basen-refI
        savv=avv
        savv2=avv*avv
        for i in range(1,nsim):
            basen=scat_op.eval(model_map1[1]+ampnoise*noise1[1,i],image2=model_map2[1]+ampnoise*noise2[1,i],mask=mask)
            avv=(basen-refI)
            savv=savv+avv
            savv2=savv2+avv*avv
        savv=savv/(nsim)
        savv2=savv2/(nsim)
        sig3=1/scat_op.sqrt(savv2-savv*savv)

        loss3=synthe.Loss(loss,scat_op,refI,mask,1,sig3,batch=batch_loss,batch_data=Rnoise)
        if size>1:
            sy = synthe.Synthesis([loss3])
            
    if size==1:
        sy = synthe.Synthesis([loss1,loss2,loss3])
        
    #=================================================================================
    # RUN ON SYNTHESIS
    #=================================================================================
    
    omap=sy.run(imap,
                EVAL_FREQUENCY = 10,
                DECAY_RATE=0.9995,
                NUM_EPOCHS = nstep,
                LEARNING_RATE = 0.3,
                EPSILON = 1E-15,
                SHOWGPU=True,
                batchsz=nnoise,
                totalsz=nsim,
                do_lbfgs=True,
                axis=1)

    #=================================================================================
    # STORE RESULTS
    #=================================================================================
    
    if rank==0%size:
        for ii in range(2):
            out =scat_op.eval(omap[ii],mask=mask)
            
            out.save(  outpath+'out_%s_%d_%d'%(outname,nside,ii))

            for k in range(10):
                out =scat_op.eval(omap[ii]+ampnoise*noise[ii,k],mask=mask)
                out.save(outpath+'outn_%s_%d_%d_%d'%(outname,nside,ii,k))
                
    
        np.save(outpath+'in_%s_map_%d.npy'%(outname,nside),im)
        np.save(outpath+'mm_%s_map_%d.npy'%(outname,nside),mask[0])
        np.save(outpath+'st_%s_map_%d.npy'%(outname,nside),imap)
        np.save(outpath+'out_%s_map_%d.npy'%(outname,nside),omap)
        np.save(outpath+'out_%s_log_%d.npy'%(outname,nside),sy.get_history())


    print('Computation Done')
    sys.stdout.flush()

if __name__ == "__main__":
    main()


    
