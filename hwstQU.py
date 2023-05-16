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
    print('>python demo.py -n=8 [-c|--cov][-s|--steps=3000][-S=1234|--seed=1234][-x|--xstat] [-g|--gauss][-k|--k5x5][-d|--data][-o|--out][-K|--k128][-r|--orient] [-p|--path] [-r|rmask][-b|--batch][-l|--nsim][-v|--vsim]')
    print('-n : is the nside of the input map (nside max = 256 with the default map)')
    print('--cov (optional): use scat_cov instead of scat.')
    print('--steps (optional): number of iteration, if not specified 30 (use all available noise x30).')
    print('--seed  (optional): seed of the random generator.')
    print('--xstat (optional): work with cross statistics.')
    print('--path  (optional): Define the path where output file are written (default data)')
    print('--k5x5  (optional): Work with a 5x5 kernel instead of a 3x3.')
    print('--out   (optional): If not specified save in *_demo_*.')
    print('--orient(optional): If not specified use 4 orientation')
    print('--batch (optional): number of available batch (default 100)')
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
        opts, args = getopt.getopt(sys.argv[1:], "n:cS:s:ko:r:b:l:v", \
                                   ["nside", "cov","seed","steps","k5x5","out","orient","batch","nsim","vsim"])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    cov=False
    nside=-1
    nstep=30
    KERNELSZ=3
    seed=1234
    outname='demo'
    outpath='results/'
    instep=16
    norient=4
    nnoise=1
    nsim=100
    dosim=False
    
    for o, a in opts:
        print(o,a)
        if o in ("-c","--cov"):
            cov = True
        elif o in ("-v","--vsim"):
            dosim = True
        elif o in ("-n", "--nside"):
            nside=int(a[1:])
        elif o in ("-s", "--steps"):
            nstep=int(a[1:])
        elif o in ("-S", "--seed"):
            seed=int(a[1:])
            print('Use SEED = ',seed)
        elif o in ("-b", "--batch"):
            nnoise=int(a[1:])
            print('Size of batch = ',nnoise)
        elif o in ("-l", "--nsim"):
            nsim=int(a[1:])
            print('Number of SIMs = ',nsim)
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
        
    nsim=nsim+1
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
    mask=np.ones([5,im.shape[1]])
    
    for i in range(4):
        mask[1+i,:]=dodown(np.load('/travail/jdelouis/heal_cnn/MASK_GAL%s_256.npy'%(tab[i])),nside)
        mask[1+i]/=mask[1+i].mean()
        
    #=================================================================================
    # Generate a random noise with the same coloured than the input data
    #=================================================================================
        
    np.random.seed(seed)
    
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
            imap[k]=im[k]+noise[k,0]
            imap1[k]=im[k]+noise1[k,0]
            imap2[k]=im[k]+noise2[k,0]
    
    noise=noise[:,1:]
    noise1=noise1[:,1:]
    noise2=noise2[:,1:]

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
                     OSTEP=3,            # get very large scale (nside=1)
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
    model_map=imap.copy()
    model_map1=imap1.copy()
    model_map2=imap2.copy()
    
    #=================================================================================
    # DEFINE A LOSS FUNCTION AND THE SYNTHESIS
    #=================================================================================

    # the first loss function definition:
    # Loss = sum^{n_noise}_k { sum_s0,s1,s2,P00 {\frac{P(d_1,d_2)-P(x+n_k,1,x+n_k,2)}{\sigma_s0,s1,s2,p00}}}
    # where:
    # P(x,y) is the CWST of the map x and y that compute 4 coefficient sets (s_0,s_1,s_2,p00)
    # d_1,d_2 are the two half mission of the same map
    # x is the map to find
    # n_k,1,n_k,2 is the simulated k th noise respectively of the first and second half mission
    
    def loss(x,scat_operator,args):
        
        ref = args[0]
        mask = args[1]
        i = args[2]
        sig = args[3]

        tmp = scat_operator.eval(x[i],image2=x[i],mask=mask)
        
        learn = scat_operator.ldiff(sig,ref - tmp)
        
        loss = scat_operator.reduce_mean(learn)
        
        return loss

    # the first loss function definition:
    # Loss = sum^{n_noise}_k { sum_s0,s1,s2,P00 {\frac{P(d_1,d_2)-P(x+n_k,1,x+n_k,2)}{\sigma_s0,s1,s2,p00}}}
    # where:
    # P(x,y) is the CWST of the map x and y that compute 4 coefficient sets (s_0,s_1,s_2,p00)
    # d_1,d_2 are the two half mission of the same map
    # x is the map to find
    # n_k,1,n_k,2 is the simulated k th noise respectively of the first and second half mission
    
    def lossD(x,scat_operator,args):
        
        bias = args[0]
        mask = args[1]
        i = args[2]
        sig = args[3]
        imap= args[4]
        #ref= args[5]

        ref = scat_operator.eval(x[i],image2=x[i],mask=mask)
        tmp = scat_operator.eval(imap,image2=x[i],mask=mask)-bias
        
        learn = scat_operator.ldiff(sig,ref - tmp)
        
        loss = scat_operator.reduce_mean(learn)
        
        return loss

    
    # the first loss function definition:
    # Loss = sum^{n_noise}_k { sum_s0,s1,s2,P00 {\frac{P(d_1,d_2)-P(x+n_k,1,x+n_k,2)}{\sigma_s0,s1,s2,p00}}}
    # where:
    # P(x,y) is the CWST of the map x and y that compute 4 coefficient sets (s_0,s_1,s_2,p00)
    # d_1,d_2 are the two half mission of the same map
    # x is the map to find
    # n_k,1,n_k,2 is the simulated k th noise respectively of the first and second half mission
    
    def lossT(x,scat_operator,args):
        
        ref = args[0]
        mask = args[1]
        i = args[2]
        sig = args[3]
        imapT= args[4]

        tmp = scat_operator.eval(imapT,image2=x[i],mask=mask)
        
        learn = scat_operator.ldiff(sig,ref - tmp)
        
        loss = scat_operator.reduce_mean(learn)
        
        return loss
    
    # the cross loss function definition:
    # Loss = sum^{n_noise}_k { sum_s0,s1,s2,P00 {\frac{P(Q,U)-P(x[0]+n_{k,q},x[1]+n_{k,u})}{\sigma_s0,s1,s2,p00}}}
    # where:
    # P(x,y) is the CWST of the map x and y that compute 4 coefficient sets (s_0,s_1,s_2,p00)
    # Q,U are the two Q,U map 
    # x is the maps to find x[0] will be the clean Q map and x[1] is the clean U map
    # n_{k,q},n_{k,u} is the simulated k th noise respectively of the first and second half mission
    
    def lossX(x,scat_operator,args):
        
        ref = args[0]
        mask = args[1]
        sig = args[2]
        
        tmp = scat_operator.eval(x[0],image2=x[1],mask=mask,Auto=False)

        learn = scat_operator.ldiff(sig,ref -tmp)
        
        loss = scat_operator.reduce_mean(learn)
        
        return loss

    allsize=7

    init_map=imap.copy()
    
    for itt in range(4):
        # all mpi rank that are consistent with 0 are computing the loss for P(Q,U) ~ P(x[0]+n_q,x[1]+n_u)
        if rank%allsize==0%size:

            # Compute reference spectra
            refX=scat_op.eval(model_map[0],image2=model_map[1],mask=mask)

            # Compute sigma for each CWST coeffients using simulation
            basen=scat_op.eval(model_map[0]+noise[0,0],image2=model_map[1]+noise[1,0],mask=mask)
            avv=basen-refX
            savv=avv
            savv2=avv*avv
            for i in range(1,nsim):
                basen=scat_op.eval(model_map[0]+noise[0,i],image2=model_map[1]+noise[1,i],mask=mask)
                avv=(basen-refX)
                savv=savv+avv
                savv2=savv2+avv*avv

            savv=savv/(nsim)
            savv2=savv2/(nsim)
            sig1=1/scat_op.sqrt(savv2-savv*savv)

            refX=scat_op.eval(imap[0],image2=imap[1],Auto=False,mask=mask)

            #savv.P00=0*savv.P00
            sig1.P00=10*sig1.P00
            
            loss1=synthe.Loss(lossX,scat_op,refX-savv,mask,sig1)

            # If parallel declare one synthesis function per mpi process
            if size>1:
                sy = synthe.Synthesis([loss1])

        if rank%allsize==1%size:

            refR=scat_op.eval(model_map1[0],image2=model_map2[0],mask=mask)

            # Compute sigma for each CWST coeffients using simulation
            basen=scat_op.eval(model_map1[0]+noise1[0,0],image2=model_map2[0]+noise2[0,0],mask=mask)
            avv=basen-refR
            savv=avv
            savv2=avv*avv
            for i in range(1,nsim):
                basen=scat_op.eval(model_map1[0]+noise1[0,i],image2=model_map2[0]+noise2[0,i],mask=mask)
                avv=(basen-refR)
                savv=savv+avv
                savv2=savv2+avv*avv

            refR=scat_op.eval(imap1[0],image2=imap2[0],mask=mask)

            savv=savv/(nsim)
            savv2=savv2/(nsim)
            sig2=1/scat_op.sqrt(savv2-savv*savv)
            # create a loss class to declare it to the synthesis function
            #savv.P00=0*savv.P00
            sig2.P00=10*sig2.P00
            loss2=synthe.Loss(loss,scat_op,refR-savv,mask,0,sig2)

            # If parallel declare one synthesis function per mpi process
            if size>1:
                sy = synthe.Synthesis([loss2])

        if rank%allsize==2%size:

            refI=scat_op.eval(model_map1[1],image2=model_map2[1],mask=mask)

            # Compute sigma for each CWST coeffients using simulation

            basen=scat_op.eval(model_map1[1]+noise1[1,0],image2=model_map2[1]+noise2[1,0],mask=mask)

            avv=basen-refI
            savv=avv
            savv2=avv*avv
            for i in range(1,nsim):
                basen=scat_op.eval(model_map1[1]+noise1[1,i],image2=model_map2[1]+noise2[1,i],mask=mask)
                avv=(basen-refI)
                savv=savv+avv
                savv2=savv2+avv*avv
                
            savv=savv/(nsim)
            savv2=savv2/(nsim)
            sig3=1/scat_op.sqrt(savv2-savv*savv)

            refI=scat_op.eval(imap1[1],image2=imap2[1],mask=mask)

            #savv.P00=0*savv.P00
            sig3.P00=10*sig3.P00
            loss3=synthe.Loss(loss,scat_op,refI-savv,mask,1,sig3)
            
            if size>1:
                sy = synthe.Synthesis([loss3])

        if rank%allsize==3%size:
            
            refI=scat_op.eval(model_map1[0],image2=model_map2[0],mask=mask)

            # Compute sigma for each CWST coeffients using simulation
            basen=scat_op.eval(model_map1[0]+noise[0,0],image2=model_map2[0],mask=mask)
            basen2=scat_op.eval(model_map1[0]+noise1[0,0],image2=model_map2[0]+noise2[0,0],mask=mask)

            avv=basen-refI
            savv=avv
            savv2=avv*avv

            avv=basen2-refI
            savvR=avv
            savvR2=avv*avv

            for i in range(1,nsim):
                basen=scat_op.eval(model_map1[0]+noise[0,i],image2=model_map2[0],mask=mask)
                basen2=scat_op.eval(model_map1[0]+noise1[0,i],image2=model_map2[0]+noise2[0,i],mask=mask)

                avv=(basen-refI)
                savv=savv+avv
                savv2=savv2+avv*avv

                avv=(basen2-refI)
                savvR=savvR+avv
                savvR2=savvR2+avv*avv
                
            savv=savv/(nsim)
            savv2=savv2/(nsim)

            savvR=savv/(nsim)
            savvR2=savv2/(nsim)

            if not cov:
                savv2.S0=savv2.S0+1.0
                savvR2.S0=savvR2.S0+1.0

            sig4=1/scat_op.sqrt(savv2+savvR2-savv*savv-savvR*savvR)

            #savv.P00=0*savv.P00
            #savvR.P00=0*savvR.P00

            sig4.P00=10*sig4.P00
            refR=scat_op.eval(imap1[0],image2=imap2[0],mask=mask)

            loss4=synthe.Loss(lossD,scat_op,savv,mask,0,sig4,imap[0],refR-savvR)
            
            if size>1:
                sy = synthe.Synthesis([loss4])

        if rank%allsize==4%size:
            
            refI=scat_op.eval(model_map1[1],image2=model_map2[1],mask=mask)

            # Compute sigma for each CWST coeffients using simulation

            basen=scat_op.eval(model_map1[1]+noise[1,0],image2=model_map2[1],mask=mask)
            basen2=scat_op.eval(model_map1[1]+noise1[1,0],image2=model_map2[1]+noise2[1,0],mask=mask)

            avv=basen-refI
            savv=avv
            savv2=avv*avv

            avv=basen2-refI
            savvR=avv
            savvR2=avv*avv

            for i in range(1,nsim):
                basen=scat_op.eval(model_map1[1]+noise[1,i],image2=model_map2[1],mask=mask)
                basen2=scat_op.eval(model_map1[1]+noise1[1,i],image2=model_map2[1]+noise2[1,i],mask=mask)

                avv=(basen-refI)
                savv=savv+avv
                savv2=savv2+avv*avv

                avv=(basen2-refI)
                savvR=savvR+avv
                savvR2=savvR2+avv*avv
                
            savv=savv/(nsim)
            savv2=savv2/(nsim)

            savvR=savv/(nsim)
            savvR2=savv2/(nsim)

            if not cov:
                savv2.S0=savv2.S0+1.0
                savvR2.S0=savvR2.S0+1.0

            sig5=1/scat_op.sqrt(savv2-savv*savv+savvR2-savvR*savvR)
            
            #savv.P00=0*savv.P00
            #savvR.P00=0*savvR.P00
            sig5.P00=10*sig5.P00

            refR=scat_op.eval(imap1[1],image2=imap2[1],mask=mask)

            loss5=synthe.Loss(lossD,scat_op,savv,mask,1,sig5,imap[1],refR-savvR)
            
            if size>1:
                sy = synthe.Synthesis([loss5])

        
        if rank%allsize==5%size:
            
            refI=scat_op.eval(mapT,image2=model_map[0],mask=mask)

            # Compute sigma for each CWST coeffients using simulation

            basen=scat_op.eval(mapT,image2=model_map[0]+noise[0,0],mask=mask)

            avv=basen-refI
            savv=avv
            savv2=avv*avv
            for i in range(1,nsim):
                basen=scat_op.eval(mapT,image2=model_map[0]+noise[0,i],mask=mask)
                avv=(basen-refI)
                savv=savv+avv
                savv2=savv2+avv*avv
                
            savv=savv/(nsim)
            savv2=savv2/(nsim)
            
            if not cov:
                savv2.S0=savv2.S0+1.0

            sig6=1/scat_op.sqrt(savv2-savv*savv)

            refI=scat_op.eval(mapT,image2=imap[0],mask=mask)
            
            #savv.P00=0*savv.P00
            sig6.P00=10*sig6.P00
            loss6=synthe.Loss(lossT,scat_op,refI-savv,mask,0,sig6,mapT)
            
            if size>1:
                sy = synthe.Synthesis([loss6])

        if rank%allsize==6%size:
            
            refI=scat_op.eval(mapT,image2=model_map[1],mask=mask)

            # Compute sigma for each CWST coeffients using simulation

            basen=scat_op.eval(mapT,image2=model_map[1]+noise[1,0],mask=mask)

            avv=basen-refI
            savv=avv
            savv2=avv*avv
            for i in range(1,nsim):
                basen=scat_op.eval(mapT,image2=model_map[1]+noise[1,i],mask=mask)
                avv=(basen-refI)
                savv=savv+avv
                savv2=savv2+avv*avv
                
            savv=savv/(nsim)
            savv2=savv2/(nsim)
            if not cov:
                savv2.S0=savv2.S0+1.0

            sig7=1/scat_op.sqrt(savv2-savv*savv)
            
            refI=scat_op.eval(mapT,image2=imap[1],mask=mask)
            
            #savv.P00=0*savv.P00
            sig7.P00=10*sig7.P00
            loss7=synthe.Loss(lossT,scat_op,refI-savv,mask,1,sig7,mapT)
            
            if size>1:
                sy = synthe.Synthesis([loss7])

        if size==1:
            sy = synthe.Synthesis([loss1,loss2,loss3,loss4,loss5,loss6,loss7])

        #=================================================================================
        # RUN ON SYNTHESIS
        #=================================================================================

        number_of_sim=nsim
        if nnoise==1:
            number_of_sim=1

        omap=sy.run(init_map,
                    EVAL_FREQUENCY = 10,
                    NUM_EPOCHS = nstep,
                    SHOWGPU=True,
                    do_lbfgs=True,
                    axis=1,
                    MESSAGE='ITT%02d-'%(itt))

        #nstep=int(nstep*1.25)

        #=================================================================================
        # STORE RESULTS
        #=================================================================================

        if rank==0%size:
            # save input data
            for ii in range(2):
                ref=scat_op.eval(im[ii],mask=mask)
                start=scat_op.eval(imap[ii],mask=mask)
            
                ref.save( outpath+'in_%s%d_%d_%d'%(outname,itt,nside,ii))
                start.save(outpath+'st_%s%d_%d_%d'%(outname,itt,nside,ii))
                
            for ii in range(2):
                out =scat_op.eval(omap[ii],mask=mask)

                out.save(  outpath+'out_%s%d_%d_%d'%(outname,itt,nside,ii))

                for k in range(10):
                    out =scat_op.eval(omap[ii]+noise[ii,k],mask=mask)
                    out.save(outpath+'outn_%s%d_%d_%d_%d'%(outname,itt,nside,ii,k))

            np.save(outpath+'in_%s%d_map_%d.npy'%(outname,itt,nside),im)
            np.save(outpath+'mm_%s%d_map_%d.npy'%(outname,itt,nside),mask[0])
            np.save(outpath+'st_%s%d_map_%d.npy'%(outname,itt,nside),imap)
            np.save(outpath+'st1_%s%d_map_%d.npy'%(outname,itt,nside),imap1)
            np.save(outpath+'st2_%s%d_map_%d.npy'%(outname,itt,nside),imap2)
            np.save(outpath+'out_%s%d_map_%d.npy'%(outname,itt,nside),omap)
            np.save(outpath+'out_%s%d_log_%d.npy'%(outname,itt,nside),sy.get_history())


        # map use to compute the sigma noise. In this example uses the input map
        model_map=omap.copy()
        model_map1=omap.copy()
        model_map2=omap.copy()
        #init_map=omap.copy()

    print('Computation Done')
    sys.stdout.flush()

if __name__ == "__main__":
    main()


    
