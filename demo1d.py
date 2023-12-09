import numpy as np
import os, sys
import matplotlib.pyplot as plt
import healpy as hp
import getopt
from scipy.ndimage import gaussian_filter

#=================================================================================
# INITIALIZE FoCUS class
#=================================================================================
import foscat.Synthesis as synthe

def usage():
    print(' This software is a demo of the foscat library:')
    print('>python demo2d.py -n=8 [-c|--cov][-s|--steps=3000][-S=1234|--seed=1234][-x|--xstat] [-g|--gauss][-k|--k5x5][-d|--data][-o|--out] [-p|--path]')
    print('-n : is the n of the input map (nxn)')
    print('--cov (optional): use scat_cov instead of scat.')
    print('--steps (optional): number of iteration, if not specified 1000.')
    print('--seed  (optional): seed of the random generator.')
    print('--xstat (optional): work with cross statistics.')
    print('--gauss (optional): convert Venus map in gaussian field.')
    print('--k5x5  (optional): Work with a 5x5 kernel instead of a 3x3.')
    print('--data  (optional): If not specified use TURBU.npy.')
    print('--out   (optional): If not specified save in *_demo_*.')
    print('--path  (optional): Define the path where output file are written (default data)')
    exit(0)
    
def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "n:cS:s:xpgkd:o:p:", \
                                   ["nside", "cov","seed","steps","xstat","p00","gauss","k5x5","data","out","path"])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    cov=False
    nside=-1
    nstep=1000
    docross=False
    dogauss=False
    KERNELSZ=3
    seed=1234
    outname='demo'
    outpath='data/'
    data="data/TURBU.npy"
    
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
        elif o in ("-g", "--gauss"):
            dogauss=True
        elif o in ("-k", "--k5x5"):
            KERNELSZ=5
        elif o in ("-p", "--path"):
            outpath=a[1:]
        else:
            assert False, "unhandled option"

    if nside<2:
        print('n should be in [2,...]')
        usage()
        exit(0)

    print('Work with n=%d'%(nside))

    #=================================================================================
    # Choose the type of Scattering Transform to be used
    #=================================================================================
    if cov:
        import foscat.scat_cov1D as sc
        print('Work with ScatCov')
    else:
        import foscat.scat1D as sc
        print('Work with Scat')
        
    #=================================================================================
    # DEFINE A PATH FOR scratch data
    # The data are storred using a default nside to minimize the needed storage
    #=================================================================================
    scratch_path = 'data'

    #=================================================================================
    # Get data
    #=================================================================================
    im=np.load('press.npy')

    l1=im[700:700+nside]
        
    #=================================================================================
    # Generate a random noise with the same coloured than the input data
    #=================================================================================

    lam=1.2
    if KERNELSZ==5:
        lam=1.0

    #=================================================================================
    # COMPUTE THE WAVELET TRANSFORM OF THE REFERENCE MAP
    #=================================================================================
    scat_op=sc.funct(KERNELSZ=5,  # define the kernel size
                     JmaxDelta=0,        # Work with all large scales
                     TEMPLATE_PATH=scratch_path,
                     padding='VALID',
                     all_type='float32')
    
    #=================================================================================
    # DEFINE A LOSS FUNCTION AND THE SYNTHESIS
    #=================================================================================

    mask=np.ones([1,l1.shape[0]])
    mask[0,:KERNELSZ//2]=0
    mask[0,-KERNELSZ//2:]=0
    def The_loss(x,scat_operator,args):
        
        ref = args[0]
        mask= args[1]

        learn=scat_operator.eval(abs(x))

        loss=scat_operator.reduce_sum(scat_operator.square(ref-learn))

        return(loss)

    ref=scat_op.eval(l1,mask=mask)

    loss1=synthe.Loss(The_loss,scat_op,ref,mask)
        
    sy = synthe.Synthesis([loss1])
    #=================================================================================
    # RUN ON SYNTHESIS
    #=================================================================================
    np.random.seed(seed)
    
    np.save(outpath+'in1d_%s_map_%d.npy'%(outname,nside),l1)
    
    np.random.seed(100)
    imap=np.random.rand(l1.shape[0])*np.std(l1)+np.median(l1)
    
    omap=abs(sy.run(imap,
                    EVAL_FREQUENCY = 100,
                    NUM_EPOCHS = nstep))

    #=================================================================================
    # STORE RESULTS
    #=================================================================================
    
    np.save(outpath+'st1d_%s_map_%d.npy'%(outname,nside),imap)
    np.save(outpath+'out1d_%s_map_%d.npy'%(outname,nside),omap)
    np.save(outpath+'out1d_%s_log_%d.npy'%(outname,nside),sy.get_history())

    (scat_op.eval(omap)).save(outpath+'out1d_%s_%d'%(outname,nside))
    (scat_op.eval(imap)).save(outpath+'st1d_%s_%d'%(outname,nside))
    ref.save(outpath+'in1d_%s_%d'%(outname,nside))
    print('Computation Done')
    sys.stdout.flush()

if __name__ == "__main__":
    main()


    
