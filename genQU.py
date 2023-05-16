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
    print('-n : is the nside of the input map (nside max = 2048 with the default map)')
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

    if nside<2 or nside!=2**(int(np.log(nside)/np.log(2))) or (nside>2048 and KERNELSZ<=5) or (nside>2**instep and KERNELSZ>5) :
        print('nside should be a power of 2 and in [2,...,2048] ')
        usage()
        exit(0)

    print('Work with nside=%d'%(nside))
    inside=256
    add_scale=int((np.log(nside)-np.log(inside))/np.log(2))

    print('Work with add_scale=%d'%(add_scale))

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
        
        if nin==nout:
            return(a)
        
        if axis==0:
            return(np.mean(a.reshape(12*nout*nout,(nin//nout)**2),1))
        if axis==1:
            return(np.mean(a.reshape(a.shape[0],12*nout*nout,(nin//nout)**2),2))

    def doup(a,nout,axis=0):

        nin=int(np.sqrt(a.shape[axis]//12))
        
        if nin==nout:
            return(a)
        
        idx=hp.ring2nest(nin,np.arange(12*nin*nin))
        th,ph=hp.pix2ang(nout,np.arange(12*nout*nout),nest=True)
        if axis==0:
            return hp.get_interp_val(a[idx],th,ph)
        if axis==1:
            res=np.zeros([a.shape[0],12*nout*nout])
            for k in range(a.shape[0]):
                res[k,:]=hp.get_interp_val(a[k,idx],th,ph)
            return res

    # convert M=Q+jU to M=[Q,U]
    def toreal(a):
        b=np.concatenate([np.real(np.expand_dims(a,0)),np.imag(np.expand_dims(a,0))])
        return(b)
    
    #=================================================================================
    # Get data and convert from nside=256 to the choosen nside
    #=================================================================================
    # read data
    

    refmap=dodown(np.load(outpath+'out_%s_map_%d.npy'%(outname,inside)),inside,axis=1)

    """
    if add_scale>0:
        srefmap=doup(refmap,nside,axis=1)
    else:
        srefmap=refmap.copy()
    """

    idx=hp.nest2ring(nside,np.arange(12*nside**2))

    im=np.zeros([2,12*nside**2])
    im1=np.zeros([2,12*nside**2])
    im2=np.zeros([2,12*nside**2])

    for i in range(2):
        cmb=hp.read_map('/travail/jdelouis/SROLL20/COM_CMB_IQU-smica_2048_R3.00_full.fits',1+i)
        im[i]=1E6*hp.ud_grade(hp.read_map('/travail/jdelouis/SROLL20/SRoll20_SkyMap_353psb_full.fits',1+i)-cmb,nside)[idx]
        im1[i]=1E6*hp.ud_grade(hp.read_map('/travail/jdelouis/SROLL20/SRoll20_SkyMap_353psb_halfmission-1.fits',1+i)-cmb,nside)[idx]
        im2[i]=1E6*hp.ud_grade(hp.read_map('/travail/jdelouis/SROLL20/SRoll20_SkyMap_353psb_halfmission-2.fits',1+i)-cmb,nside)[idx]

    mapT=hp.ud_grade(hp.read_map('/travail/jdelouis/SROLL20/SRoll22_SkyMap_857ghz_full.fits'),nside)[idx]


    tab=[6,4,2,1]
    mask=np.ones([5,im.shape[1]])
    
    for i in range(4):
        mask[1+i,:]=hp.ud_grade(hp.read_map('/travail/jdelouis/SROLL20/HFI_Mask_GalPlane-apo5_2048_R2.00.fits',i),nside)[idx]
        mask[1+i]/=mask[1+i].mean()

    if add_scale>0:
        smapT=dodown(mapT,inside)
        smask=dodown(mask,inside,axis=1)
    else:
        smapT=mapT
        smask=mask

    imap=np.zeros([2,12*nside**2])
    imap1=np.zeros([2,12*nside**2])
    imap2=np.zeros([2,12*nside**2])

    for k in range(2):
        imap[k]=im[k]
        imap1[k]=im1[k]
        imap2[k]=im2[k]
    
    lam=1.2
    if KERNELSZ==5:
        lam=1.0

    l_slope=1.0
    r_format=True
    all_type='float64'
    
    np.random.seed(seed)

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
        
        mask = args[0]
        i = args[1]
        sig = args[2]
        imap= args[3]
        ref= args[4]

        tmp = scat_operator.eval(imap,image2=x[i],mask=mask)
        
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

    idx=hp.nest2ring(inside,np.arange(12*inside*inside))
    
    # all mpi rank that are consistent with 0 are computing the loss for P(Q,U) ~ P(x[0]+n_q,x[1]+n_u)
    if rank%allsize==0%size:

        # Compute reference spectra
        refX=scat_op.eval(refmap[0],image2=refmap[1],Auto=False,mask=smask)

        #=================================================================================
        # Get noise to evaluate sigma for each loss
        #=================================================================================

        savv=None
        savv2=None

        for i in range(nsim):


            noise1 = 1E6*hp.ud_grade(hp.read_map('/travail/jdelouis/DownGrade_256/JAN18r60_%03d_353psb_353psb_hm1_IQU.fits'%(i+1),1),inside)[idx]
            noise2 = 1E6*hp.ud_grade(hp.read_map('/travail/jdelouis/DownGrade_256/JAN18r60_%03d_353psb_353psb_hm2_IQU.fits'%(i+1),2),inside)[idx]

            basen=scat_op.eval(refmap[0]+noise1,image2=refmap[1]+noise2,Auto=False,mask=smask)

            avv=basen-refX

            if savv is None:
                savv=avv
                savv2=avv*avv
            else:
                savv=savv+avv
                savv2=savv2+avv*avv

        savv=savv/(nsim)
        savv2=savv2/(nsim)

        sig1=1/scat_op.sqrt(savv2-savv*savv)

        #FAIRE PROLONGATION VERS AUTRE ECHELLES ICI !
        if add_scale>0:
            sig1=sig1.extrapol(add_scale)
            refX=refX.extrapol(add_scale)
            
        loss1=synthe.Loss(lossX,scat_op,refX,mask,sig1)

        # If parallel declare one synthesis function per mpi process
        if size>1:
            sy = synthe.Synthesis([loss1])

    if rank%allsize==1%size:

        refR=scat_op.eval(refmap[0],image2=refmap[0],mask=smask)

        #=================================================================================
        # Get noise to evaluate sigma for each loss
        #=================================================================================

        savv=None
        savv2=None

        for i in range(nsim):

            noise1 = 1E6*hp.ud_grade(hp.read_map('/travail/jdelouis/DownGrade_256/JAN18r60_%03d_353psb_353psb_hm1_IQU.fits'%(i+1),1),inside)[idx]
            noise2 = 1E6*hp.ud_grade(hp.read_map('/travail/jdelouis/DownGrade_256/JAN18r60_%03d_353psb_353psb_hm2_IQU.fits'%(i+1),1),inside)[idx]

            basen=scat_op.eval(refmap[0]+noise1,image2=refmap[0]+noise2,mask=smask)

            avv=basen-refR

            if savv is None:
                savv=avv
                savv2=avv*avv
            else:
                savv=savv+avv
                savv2=savv2+avv*avv

        savv=savv/(nsim)
        savv2=savv2/(nsim)

        sig2=1/scat_op.sqrt(savv2-savv*savv)

        #FAIRE PROLONGATION VERS AUTRE ECHELLES ICI !
        if add_scale>0:
            sig2=sig2.extrapol(add_scale)
            refR=refR.extrapol(add_scale)

        loss2=synthe.Loss(loss,scat_op,refR,mask,0,sig2)

        # If parallel declare one synthesis function per mpi process
        if size>1:
            sy = synthe.Synthesis([loss2])

    if rank%allsize==2%size:

        refI=scat_op.eval(refmap[1],image2=refmap[1],mask=smask)

        #=================================================================================
        # Get noise to evaluate sigma for each loss
        #=================================================================================

        savv=None
        savv2=None

        for i in range(nsim):

            noise1 = 1E6*hp.ud_grade(hp.read_map('/travail/jdelouis/DownGrade_256/JAN18r60_%03d_353psb_353psb_hm1_IQU.fits'%(i+1),2),inside)[idx]
            noise2 = 1E6*hp.ud_grade(hp.read_map('/travail/jdelouis/DownGrade_256/JAN18r60_%03d_353psb_353psb_hm2_IQU.fits'%(i+1),2),inside)[idx]

            basen=scat_op.eval(refmap[1]+noise1,image2=refmap[1]+noise2,mask=smask)

            avv=basen-refI

            if savv is None:
                savv=avv
                savv2=avv*avv
            else:
                savv=savv+avv
                savv2=savv2+avv*avv

        savv=savv/(nsim)
        savv2=savv2/(nsim)

        sig3=1/scat_op.sqrt(savv2-savv*savv)

        #FAIRE PROLONGATION VERS AUTRE ECHELLES ICI !
        if add_scale>0:
            sig3=sig3.extrapol(add_scale)
            refI=refI.extrapol(add_scale)

        loss3=synthe.Loss(loss,scat_op,refI,mask,1,sig3)
            
        if size>1:
            sy = synthe.Synthesis([loss3])

    if rank%allsize==3%size:
            
        refR=scat_op.eval(refmap[0],image2=refmap[0],mask=smask)

        #=================================================================================
        # Get noise to evaluate sigma for each loss
        #=================================================================================

        savv=None
        savv2=None

        for i in range(nsim):
            
            noise  = 1E6*hp.ud_grade(hp.read_map('/travail/jdelouis/DownGrade_256/JAN18r60_%03d_353psb_353psb_full_IQU.fits'%(i+1),1),inside)[idx]

            basen=scat_op.eval(refmap[0]+noise,image2=refmap[0],Auto=False,mask=smask)

            avv=basen-refR

            if savv is None:
                savv=avv
                savv2=avv*avv
            else:
                savv=savv+avv
                savv2=savv2+avv*avv

        savv=savv/(nsim)
        savv2=savv2/(nsim)

        sig4=1/scat_op.sqrt(savv2-savv*savv)

        #FAIRE PROLONGATION VERS AUTRE ECHELLES ICI (sig4,refR)!
        if add_scale>0:
            sig4=sig4.extrapol(add_scale)
            refR=refR.extrapol(add_scale)

        loss4=synthe.Loss(lossD,scat_op,mask,0,sig4,imap[0],refR)
            
        if size>1:
            sy = synthe.Synthesis([loss4])

    if rank%allsize==4%size:
            
        refI=scat_op.eval(refmap[1],image2=refmap[1],mask=smask)

        #=================================================================================
        # Get noise to evaluate sigma for each loss
        #=================================================================================

        savv=None
        savv2=None

        for i in range(nsim):
            
            noise  = 1E6*hp.ud_grade(hp.read_map('/travail/jdelouis/DownGrade_256/JAN18r60_%03d_353psb_353psb_full_IQU.fits'%(i+1),2),inside)[idx]

            basen=scat_op.eval(refmap[1]+noise,image2=refmap[1],mask=smask)

            avv=basen-refI

            if savv is None:
                savv=avv
                savv2=avv*avv
            else:
                savv=savv+avv
                savv2=savv2+avv*avv

        savv=savv/(nsim)
        savv2=savv2/(nsim)

        sig5=1/scat_op.sqrt(savv2-savv*savv)
            
        #FAIRE PROLONGATION VERS AUTRE ECHELLES ICI (sig5,refR)!
        if add_scale>0:
            sig5=sig5.extrapol(add_scale)
            refI=refI.extrapol(add_scale)

        loss5=synthe.Loss(lossD,scat_op,mask,1,sig5,imap[1],refI)
        
        if size>1:
            sy = synthe.Synthesis([loss5])

    if rank%allsize==5%size:
            
        refR=scat_op.eval(smapT,image2=refmap[0],mask=smask)

        #=================================================================================
        # Get noise to evaluate sigma for each loss
        #=================================================================================

        savv=None
        savv2=None

        for i in range(nsim):
            
            noise  = 1E6*hp.ud_grade(hp.read_map('/travail/jdelouis/DownGrade_256/JAN18r60_%03d_353psb_353psb_full_IQU.fits'%(i+1),1),inside)[idx]
            basen=scat_op.eval(smapT,image2=refmap[0]+noise,mask=smask)

            avv=basen-refR

            if savv is None:
                savv=avv
                savv2=avv*avv
            else:
                savv=savv+avv
                savv2=savv2+avv*avv

        savv=savv/(nsim)
        savv2=savv2/(nsim)
        if not cov:
            savv2.S0=savv2.S0+1.0

        sig6=1/scat_op.sqrt(savv2-savv*savv)

        #FAIRE PROLONGATION VERS AUTRE ECHELLES ICI !
        if add_scale>0:
            sig6=sig6.extrapol(add_scale)
            refR=refR.extrapol(add_scale)

        loss6=synthe.Loss(lossT,scat_op,refR,mask,0,sig6,mapT)
            
        if size>1:
            sy = synthe.Synthesis([loss6])

    if rank%allsize==6%size:
            
        refI=scat_op.eval(smapT,image2=refmap[1],mask=smask)

        #=================================================================================
        # Get noise to evaluate sigma for each loss
        #=================================================================================

        savv=None
        savv2=None

        for i in range(nsim):
            
            noise  = 1E6*hp.ud_grade(hp.read_map('/travail/jdelouis/DownGrade_256/JAN18r60_%03d_353psb_353psb_full_IQU.fits'%(i+1),2),inside)[idx]

            basen=scat_op.eval(smapT,image2=refmap[1]+noise,mask=smask)

            avv=basen-refI

            if savv is None:
                savv=avv
                savv2=avv*avv
            else:
                savv=savv+avv
                savv2=savv2+avv*avv

        savv=savv/(nsim)
        savv2=savv2/(nsim)
        if not cov:
            savv2.S0=savv2.S0+1.0

        sig7=1/scat_op.sqrt(savv2-savv*savv)

        #FAIRE PROLONGATION VERS AUTRE ECHELLES ICI !
        if add_scale>0:
            sig7=sig7.extrapol(add_scale)
            refI=refI.extrapol(add_scale)

        loss7=synthe.Loss(lossT,scat_op,refI,mask,1,sig7,mapT)
            
        if size>1:
            sy = synthe.Synthesis([loss7])

    if size==1:
        sy = synthe.Synthesis([loss1,loss2,loss3,loss4,loss5,loss6,loss7])

    #=================================================================================
    # RUN ON SYNTHESIS
    #=================================================================================

    omap=sy.run(init_map,
                EVAL_FREQUENCY = 10,
                NUM_EPOCHS = nstep,
                SHOWGPU=True,
                do_lbfgs=True,
                axis=1,
                MESSAGE='GEN-')

    #=================================================================================
    # STORE RESULTS
    #=================================================================================

    if rank==0%size:

        if cov:
            np.save(outpath+'out_%sgc_map_%d.npy'%(outname,nside),omap)
            np.save(outpath+'out_%sgc_log_%d.npy'%(outname,nside),sy.get_history())
        else:
            np.save(outpath+'out_%sg_map_%d.npy'%(outname,nside),omap)
            np.save(outpath+'out_%sg_log_%d.npy'%(outname,nside),sy.get_history())

    print('Computation Done')
    sys.stdout.flush()

if __name__ == "__main__":
    main()


    
