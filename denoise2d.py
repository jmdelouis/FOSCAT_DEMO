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
    print('>python demo2d.py -n=8 [-c|--cov][-s|--steps=3000][-S=1234|--seed=1234][-x|--xstat][-p|--p00][-g|--gauss][-k|--k5x5][-d|--data][-o|--out]')
    print('-n : is the n of the input map (nxn)')
    print('--cov (optional): use scat_cov instead of scat.')
    print('--steps (optional): number of iteration, if not specified 1000.')
    print('--seed  (optional): seed of the random generator.')
    print('--xstat (optional): work with cross statistics.')
    print('--p00   (optional): Loss only computed on p00.')
    print('--gauss (optional): convert Venus map in gaussian field.')
    print('--k5x5  (optional): Work with a 5x5 kernel instead of a 3x3.')
    print('--data  (optional): If not specified use TURBU.npy.')
    print('--out   (optional): If not specified save in *_sar_*.')
    exit(0)
    
def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "n:cS:s:xpgkd:o:", \
                                   ["nside", "cov","seed","steps","xstat","p00","gauss","k5x5","data","out"])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    cov=False
    nside=-1
    nstep=100
    docross=False
    dop00=False
    dogauss=False
    KERNELSZ=3
    seed=1234
    outname='sar'
    data="TURBU.npy"
    
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
        elif o in ("-x", "--xstat"):
            docross=True
        elif o in ("-g", "--gauss"):
            dogauss=True
        elif o in ("-k", "--k5x5"):
            KERNELSZ=5
        elif o in ("-p", "--p00"):
            dop00=True
        else:
            assert False, "unhandled option"

    if nside<2 or nside!=2**(int(np.log(nside)/np.log(2))) or nside>2048:
        print('n should be a power of 2 and in [2,...,2048]')
        usage()
        exit(0)

    print('Work with n=%d'%(nside))

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
    scratch_path = '../data'

    #=================================================================================
    # Get data
    #=================================================================================
    im=np.load('sar_test.npy')
    im=im[512-nside//2:512+nside//2,512-nside//2:512+nside//2]
    im=im-np.median(im)
    tfin=abs(np.fft.fft2(im))

    tf=gaussian_filter(np.roll(np.roll(tfin,-nside//2,0),-nside//2,1),sigma=4)
    x=np.repeat(np.arange(nside)-nside/2,nside).reshape(nside,nside)
    lim=nside/8.0
    tf=tf*(1-0.9*np.exp(-(x*x+x.T*x.T)/(lim*lim)))
    tf=np.roll(np.roll(tf,-nside//2,0),-nside//2,1)/(nside)

    
    xs=nside//2

    plt.figure(figsize=(16,16))
    plt.subplot(2,2,1)
    plt.imshow(np.log(np.roll(np.roll(tfin,-xs,0),-xs,1)),cmap='jet') #,vmin=-0.03,vmax=0.03)
    
    plt.subplot(2,2,2)
    plt.imshow(im) #,vmin=-0.03,vmax=0.03)

    noise=np.fft.ifft2(np.fft.fft2(np.random.randn(2*xs,2*xs))*tf).real
    plt.subplot(2,2,3)
    plt.imshow(noise) #,vmin=-0.03,vmax=0.03)
    
    tf2=abs(np.fft.fft2(noise))
    plt.subplot(2,2,4)
    plt.imshow(np.log(np.roll(np.roll(tf2,-xs,0),-xs,1)),cmap='jet') #,vmin=-0.03,vmax=0.03)
    plt.show()
    
    #=================================================================================
    # Generate a random noise with the same coloured than the input data
    #=================================================================================

    lam=1.2
    if KERNELSZ==5:
        lam=1.0

    #=================================================================================
    # COMPUTE THE WAVELET TRANSFORM OF THE REFERENCE MAP
    #=================================================================================
    scat_op=sc.funct(NORIENT=4,          # define the number of wavelet orientation
                     KERNELSZ=KERNELSZ,  # define the kernel size
                     OSTEP=0,           # get very large scale (nside=1)
                     LAMBDA=lam,
                     TEMPLATE_PATH=scratch_path,
                     use_R_format=True,
                     chans=1,
                     all_type='float32')
    
    #=================================================================================
    # DEFINE A LOSS FUNCTION AND THE SYNTHESIS
    #=================================================================================
    
    def loss_function(x,scat_operator,args):
        
        im = args[0]
        sigma  = args[1]
        bias  = args[2]

        ref = scat_operator.eval(im,image2=x)-bias
        tmp = scat_operator.eval(x,image2=x)
            
        learn = scat_operator.ldiff(sigma,ref - tmp)

        loss = scat_operator.reduce_mean(learn)
        
        return(loss)
    
    def loss_residu(x,scat_operator,args):
        
        ref = args[0]
        sigma  = args[1]
        im = args[2]

        tmp = scat_operator.eval(im-x)
            
        learn = scat_operator.ldiff(sigma,ref - tmp)

        loss = scat_operator.reduce_mean(learn)
        
        return(loss)
    
    np.random.seed(seed)

    nsim=100
    imap=im.copy()
    model=imap.copy()
    
    for itt in range(4):
        ref=scat_op.eval(model,image2=model)
        noise=np.fft.ifft2(np.fft.fft2(np.random.randn(2*xs,2*xs))*tf).real
        tmp=scat_op.eval(model+noise,image2=model)-ref
        savv=tmp
        savv2=tmp*tmp
        tmp=scat_op.eval(noise)
        navv=tmp
        for i in range(1,nsim):
            noise=np.fft.ifft2(np.fft.fft2(np.random.randn(2*xs,2*xs))*tf).real
            tmp=scat_op.eval(model+noise,image2=model)-ref
            savv=savv+tmp
            savv2=savv2+tmp*tmp
            tmp=scat_op.eval(noise)
            navv=navv+tmp

        savv=savv/(nsim)
        savv2=savv2/(nsim)
        navv=navv/(nsim)

        # manage the 0 problem
        if not cov:
            savv2.S0=savv2.S0+1.0
        
        sigma=1/scat_op.sqrt(savv2-savv*savv)
    
        loss1=synthe.Loss(loss_function,scat_op,im,sigma,savv)

        loss2=synthe.Loss(loss_residu,scat_op,navv,sigma,scat_op.to_R(im,chans=1))
        
        sy = synthe.Synthesis([loss1,loss2])
        #=================================================================================
        # RUN ON SYNTHESIS
        #=================================================================================
        omap=sy.run(imap,
                    EVAL_FREQUENCY = 10,
                    do_lbfgs=True,
                    NUM_EPOCHS = nstep)

        #=================================================================================
        # STORE RESULTS
        #=================================================================================
    
        start=scat_op.eval(imap)
        out =scat_op.eval(omap)
    
        np.save('in2d_%s%d_map_%d.npy'%(outname,itt,nside),im)
        np.save('st2d_%s%d_map_%d.npy'%(outname,itt,nside),imap)
        np.save('out2d_%s%d_map_%d.npy'%(outname,itt,nside),omap)
        np.save('out2d_%s%d_log_%d.npy'%(outname,itt,nside),sy.get_history())

        model=omap.numpy().copy()

    print('Computation Done')
    sys.stdout.flush()

if __name__ == "__main__":
    main()


    
