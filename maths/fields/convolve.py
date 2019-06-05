import numpy as np
from maths.fields.resize import crop_pad
from maths.fields.grid import r_grid

def nd_convolve(signal,kernel,kernel_args=[],delta_r=[],periodic=True,trim=True,normalise_kernel=True,return_fft=False,correlate=False):
    
    """
    
    Function: a (somewhat) stripped down fft convolution function
    
    Arguments
    ---------
        
        signal: ndarray
            ndarray of underlying signal
            
        kernel: ndarray or function
            either centred ndarray of kernel, or spherically symmetric function f(r,*kernel_args)
            if ndarray, kernel.shape must match signal.shape
            
        kernel_args: list
            optional list of kernel arguments
            
        delta_r: list
            optional list of box edge-lengths
            
        peridoic: boolean
            Periodic convolution?
            
        trim: boolean
            If not periodic, trim back to original signal size (does not conserve "mass")
            
        normalise_kernel: boolean
            Normalise kernel to one?
            
        return_fft: boolean
            Return fft of convolution
            
        correlate: boolean
            correlate kernel with signal instead of convolve
            
    Result
    ------
    
        convolution: ndarray
            convolution of signal and kernel (or fft if return_fft==True)
                
    
    """
    
    
    # preprocess signal
    
    # zero pad signal if periodic!=True
    if not periodic:
        signal_copy=crop_pad(signal,np.array(signal.shape)*2)
    else:
        signal_copy=signal.copy()
        
    
    # preprocess kernel
     
    # set kernel array
    if callable(kernel):
        
        # set box edge-length to 1 if no arguement provided
        if len(delta_r)!=signal.ndim:
            delta_r_copy=np.ones(signal.ndim)
        else:
            delta_r_copy=np.array(delta_r)
            
        if not periodic:
            delta_r_copy*=2
        
        # make grid of r magnitudes
        r=r_grid(signal_copy.shape,delta_r_copy)
        
        # set kernel
        kernel_copy=kernel(r,*kernel_args)
        
    else:

        # crop/pad kernel to signal_copy.shape      
        kernel_copy=crop_pad(kernel,signal_copy.shape,centred=True)
        
    
    # normalise kernel to one
    if normalise_kernel:
        kernel_copy/=kernel_copy.sum()
        
    # shift kernel copy for fft
    kernel_copy=np.fft.ifftshift(kernel_copy)
            
    # multiply spectra of signal and kernel
    if correlate:
        convolution=np.fft.fftn(signal_copy)*np.fft.fftn(kernel_copy).conj()
    else:
        convolution=np.fft.fftn(signal_copy)*np.fft.fftn(kernel_copy)
    
    if not return_fft:
         
        # perform convolution
        convolution=np.fft.ifftn(convolution).real
    
        # trim convolution back to original size
        if not periodic:
            if trim:
                # trim convolution
                convolution=crop_pad(convolution,signal.shape)
            else:
                # centre full convolution
                convolution=np.roll(convolution,tuple([np.ceil(i/2).astype(np.int) for i in signal.shape]),axis=tuple(np.arange(signal.ndim)))
    
    
    return convolution
            
            
        
        
        
            
        
        
        
        
        
            
            