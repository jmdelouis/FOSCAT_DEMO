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
    print('>python demo.py -n=8 [-c|--cov][-s|--steps=3000][-S=1234|--seed=1234][-x|--xstat] [-g|--gauss][-k|--k5x5][-d|--data][-o|--out][-K|--k128][-r|--orient] [-p|--path] [-r|rmask][-l|--lbfgs]')
    print('-n : is the nside of the input map (nside max = 256 with the default map)')
    print('--cov (optional): use scat_cov instead of scat.')
    print('--steps (optional): number of iteration, if not specified 1000.')
    print('--seed  (optional): seed of the random generator.')
    print('--path  (optional): Define the path where output file are written (default value is "data").')
    print('--k5x5  (optional): Work with a 5x5 kernel instead of a 3x3.')
    print('--data  (optional): If not specified use LSS_map_nside128.npy.')
    print('--out   (optional): If not specified output file names built using *_demo_*.')
    print('--orient(optional): If not specified use 4 orientations.')
    print('--mask  (optional): if specified use a mask.')
    print('--adam (optional): If specified the ADAM minimisation instead of the L-BFGS one.')
    exit(0)
    
def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "n:cS:s:xp:gkd:o:Kr:m:a", \
                                   ["nside", "cov","seed","steps","path","k5x5",
                                    "data","out","orient","mask","adam"])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    cov=False
    nside=-1
    nstep=300
    KERNELSZ=3
    dok128=False
    seed=1234
    outname='demo'
    data="ref_simAS0_map_256.npy"
    instep=16
    norient=4
    outpath='data/'
    imask=None
    dolbfgs=True
    
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
        elif o in ("-d", "--data"):
            data=a[1:]
            print('Read data from ',data)
        elif o in ("-k", "--k5x5"):
            KERNELSZ=5
        elif o in ("-a", "--adam"):
            dolbfgs=False
        elif o in ("-r", "--orient"):
            norient=int(a[1:])
            print('Use %d orientations'%(norient))
        elif o in ("-m", "--mask"):
            imask=np.load(a[1:])
            print('Use %s mask'%(a[1:]))
        elif o in ("-p", "--path"):
            outpath=a[1:]
        else:
            assert False, "unhandled option"

    if nside<2 or nside!=2**(int(np.log(nside)/np.log(2))) or (nside>256 and KERNELSZ<=5) or (nside>2**instep and KERNELSZ>5) :
        print('nside should be a power of 2 and in [2,...,256] or [2,...,%d] if -K|-k128 option has been choosen'%(2**instep))
        usage()
        exit(0)

    print('Work with nside=%d'%(nside))
    
    #=================================================================================
    # Choose the type of Scattering Transform to be used
    #=================================================================================

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
    # Function to reduce the data used in the FoCUS algorithm (addapted to nested odering) 
    #=================================================================================
    def dodown(a,nside):
        nin=int(np.sqrt(a.shape[0]//12))
        if nin==nside:
            return(a)
        return(np.mean(a.reshape(12*nside*nside,(nin//nside)**2),1))

    #=================================================================================
    # Get data
    #=================================================================================
    lim=np.load(data)
    im=np.zeros([2,12*nside**2])
    for i in range(2):
        im[i]=dodown(lim[i],nside)
    del lim

    #=================================================================================
    # Compute mask using argsort
    #=================================================================================

    
    if imask is None:
        mask=np.ones([2,im.shape[1]])
        mask[0,:]=(im[0]!=hp.UNSEEN)*(abs(im[0])<3*np.median(abs(im[0])))
        mask[1,:]=(im[0]!=hp.UNSEEN)*(abs(im[0])>=3*np.median(abs(im[0])))
        print(np.mean(mask[0,:]))
        print(np.mean(mask[1,:]))
        im[im==hp.UNSEEN]=0.0
    else:
        mask=np.ones([2,im.shape[1]])
        mask[0,:]=dodown(imask,nside)*(im[0]!=hp.UNSEEN)
        mask[1,:]=(im[0]!=hp.UNSEEN)
        im[im==hp.UNSEEN]=0.0
        
    #=================================================================================
    # COMPUTE THE WAVELET TRANSFORM OF THE REFERENCE MAP
    #=================================================================================
    scat_op=sc.funct(NORIENT=4,          # define the number of wavelet orientation
                     KERNELSZ=KERNELSZ,  #KERNELSZ,  # define the kernel size
                     JmaxDelta=0,        # The used Jmax is Jmax-JmaxDelta
                     TEMPLATE_PATH=scratch_path,
                     all_type='float32',
                     nstep_max=instep)
    
    #=================================================================================
    # DEFINE A LOSS FUNCTION AND THE SYNTHESIS
    #=================================================================================
    
    def The_loss(x,scat_operator,args):
        
        ref = args[0]
        mask = args[1]
        k    = args[2]

        learn=scat_operator.eval(x[k],mask=mask)

        loss=scat_operator.reduce_mean(scat_operator.square(ref-learn))      

        return(loss)
    
    def The_lossX(x,scat_operator,args):
        
        ref = args[0]
        mask = args[1]

        learn=scat_operator.eval(x[0],image2=x[1],mask=mask)

        loss=scat_operator.reduce_mean(scat_operator.square(ref-learn))      

        return(loss)

    ref1=scat_op.eval(im[0],mask=mask)
    ref2=scat_op.eval(im[1],mask=mask)
    refx=scat_op.eval(im[0],image2=im[1],mask=mask)
    
    loss1=synthe.Loss(The_loss,scat_op,ref1,mask,0)
    loss2=synthe.Loss(The_loss,scat_op,ref2,mask,1)
    loss3=synthe.Loss(The_lossX,scat_op,refx,mask)
        
    sy = synthe.Synthesis([loss1,loss2,loss3])
    #=================================================================================
    # RUN ON SYNTHESIS
    #=================================================================================

    #=================================================================================
    # Generate a random noise with the same coloured than the input data
    #=================================================================================
    np.random.seed(seed)
    imap=np.random.randn(2,12*nside**2)*np.std(im[0])
    
    omap=sy.run(imap,
                EVAL_FREQUENCY=10,
                NUM_EPOCHS = nstep,
                do_lbfgs=dolbfgs)

    #=================================================================================
    # STORE RESULTS
    #=================================================================================
    
    start1=scat_op.eval(imap[0],mask=mask)
    out1 =scat_op.eval(omap[0],mask=mask)
    start2=scat_op.eval(imap[1],mask=mask)
    out2 =scat_op.eval(omap[1],mask=mask)
    
    np.save(outpath+'in_%s_map_%d.npy'%(outname,nside),im)
    np.save(outpath+'mm_%s_map_%d.npy'%(outname,nside),mask[0])
    np.save(outpath+'st_%s_map_%d.npy'%(outname,nside),imap)
    np.save(outpath+'out_%s_map_%d.npy'%(outname,nside),omap)
    np.save(outpath+'out_%s_log_%d.npy'%(outname,nside),sy.get_history())

    ref1.save( outpath+'in1_%s_%d'%(outname,nside))
    ref2.save( outpath+'in2_%s_%d'%(outname,nside))
    start1.save(outpath+'st1_%s_%d'%(outname,nside))
    start2.save(outpath+'st2_%s_%d'%(outname,nside))
    out1.save(  outpath+'out1_%s_%d'%(outname,nside))
    out2.save(  outpath+'out2_%s_%d'%(outname,nside))

    print('Computation Done')
    sys.stdout.flush()

if __name__ == "__main__":
    main()


    
