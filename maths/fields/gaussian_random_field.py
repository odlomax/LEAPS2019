import numpy as np
from scipy.special import erfc
from maths.fields.potential_field import dft_potential
from maths.fields.grid import u_r_grid
from maths.fields.convolve import nd_convolve

class scalar_grf:
    
    """
    Class for building nd gaussian random scalar field
    
    Data
    ----
    
    spectrum[...]: float
        spectrum of gaussian random field
    
    signal[...]: float
        signal of gaussian random field
    
    """
        
        
    def __init__(self,n_shape_in,beta,real_field=True,generate_signal=True):
        
        """
        Subroutine: generate a nd scalar gaussian random field with a power spectrum
        
        Arguments
        ---------
        
        n_shape_in(,): tuple, int
            shape of field
            
        beta: float
            power spectrum exponent
        
        real_field: boolean
            is field real?
            
        generate_signal: boolean
            only generate spectrum if set to false
        
        """
        

        
        # make n_shape array-like, if scalar
        if isinstance(n_shape_in,(tuple,list,np.ndarray)):
            n_shape=n_shape_in
        else:
            n_shape=(n_shape_in,)
        
        
        # make k coordinate arrays
        k_list=[]
        for i in n_shape:
            k_list.append(np.arange(-i//2+np.mod(i,2),i//2+1))

        # generate array of k values
        if len(n_shape)>1:
            k=np.zeros(np.array(n_shape)+np.mod(np.array(n_shape)+1,2))
            for i in np.meshgrid(*k_list,indexing="ij"):
                k+=i**2
        else:
            k=k_list[0]**2
            
        k=np.sqrt(k)
            
        # create amplitude array a(k)
        a_k=np.where(k>0,k**(-0.5*beta),0.)
        a_k/=np.sqrt((a_k**2).sum())
        
        # create phase array phi(k)
        phi_k=np.random.random(k.shape)*2.*np.pi
        
        # make spetrum Hermitian if signal is real
        if real_field:
            # make array of phi(-k)
            phi_k_minus=phi_k[tuple([slice(None,None,-1)]*len(n_shape))]
            phi_k=phi_k-phi_k_minus
        
        # create complex spectrum
        self.spectrum=np.zeros(k.shape,dtype=np.complex)
        self.spectrum.real=a_k*np.cos(phi_k)
        self.spectrum.imag=a_k*np.sin(phi_k)
        
        # resize array (if size is even, n//2 frequency is superposition of n//2 and -n//2)
        for i in range(len(n_shape)):
            if np.mod(n_shape[i],2)==0:
                i_plus=[slice(None)]*len(n_shape)
                i_plus[i]=-1
                i_minus=[slice(None)]*len(n_shape)
                i_minus[i]=0
                self.spectrum[tuple(i_minus)]+=self.spectrum[tuple(i_plus)]
                indices=[slice(None)]*len(n_shape)
                indices[i]=slice(None,-1)
                self.spectrum=self.spectrum[tuple(indices)]
                
        # shift array
        offset=(np.array(n_shape)+np.mod(np.array(n_shape),2))//2
        axis=np.arange(len(n_shape))
        self.spectrum=np.roll(self.spectrum,offset,axis)
            
        # perform FFT
        if generate_signal:
            self.signal=self.spectrum.size*np.fft.ifftn(self.spectrum)
        
        return
        
        
    def normalise(self,sigma=1.,exponentiate=False,exp_base=np.e,power_law=False,alpha=2.):
        
        """
        
        Subroutine: normalise signal to a standard deviation (Assumes signal has normal distribution)
        
        Arguments
        ---------
        
        sigma: float
            standard deviation
            
        exponentiate: boolean
            exponentiate signal?
            
        exp_base: float
            base of exponential
            
        power_law: boolean
            convert to power law? (mutually exclusive with boolean)
        
        """
        
        self.exponentiated=exponentiate
        self.signal*=sigma/np.std(self.signal)
        
        if exponentiate:
            # lognormal distribution
            self.signal=np.exp(self.signal*np.log(exp_base))
        elif power_law:
            # f(rho) ~ rho**(-alpha)
            self.signal=(0.5*erfc(self.signal/np.sqrt(2.)))**(1./(1.-alpha))
            
            
        
            
        # update spectrum
        self.spectrum=np.fft.fftn(self.signal)
        
        return
        
        
    def pot_shift(self,shift_in=None):
        
        """
        
        Subroutine: shifts signal to global potential minimum (signal should be positive for all x)
        
        Arguments
        ---------
        
        shift_in: int, tuple
            pre-determined shifts
            
        Result
        ------
        
        shift: int, tuple
            array shift corresponding to periodic centre of mass
        
        """
        
        
        if shift_in==None:
            # generate potential
            potential=dft_potential(self.signal.real,periodic=True)
            # lowest value of potential
            shift=np.argwhere(potential==potential.min())[0,:]
                
            # set shifts relative to array size
            shift=tuple(np.array(self.signal.shape)//2-shift)
        else:
            shift=shift_in
            
        # roll signal array
        self.signal=np.roll(self.signal,shift,tuple(range(len(shift))))
            
        # update spectrum
        self.spectrum=np.fft.fftn(self.signal)
        
        return shift
    
    
    def com_shift(self,shift_in=None):
        
        """
        
        Subroutine: shifts signal to periodic centre of mass (signal should be positive for all x)
        
        Arguments
        ---------
        
        shift_in: int, tuple
            pre-determined shifts
            
        Result
        ------
        
        shift: int, tuple
            array shift corresponding to periodic centre of mass
        
        """
        
        
        if shift_in==None:
            # set theta arrays
            theta_i=[]
            for i in range(len(self.signal.shape)):
                theta_i.append(np.arange(self.signal.shape[i],dtype=np.double)*2.*np.pi/(self.signal.shape[i]))
                
            # convert to grid format
            theta=np.meshgrid(*theta_i,indexing='ij')
            
            # loop over axes
            
            shift=np.zeros(len(theta),dtype=np.int)
            for i in range(len(theta)):
                
                # calculate shift
                xi=np.cos(theta[i])*np.abs(self.signal.real)
                zeta=np.sin(theta[i])*np.abs(self.signal.real)
                theta_bar=np.arctan2(-zeta.mean(),-xi.mean())+np.pi
                shift[i]=np.int((self.signal.shape[i])*0.5*theta_bar/np.pi)
                
                
                
            # set shifts relative to array size
            shift=tuple(np.array(self.signal.shape)//2-shift)
        else:
            shift=shift_in
            
        # roll signal array
        self.signal=np.roll(self.signal,shift,tuple(range(len(shift))))
            
        # update spectrum
        self.spectrum=np.fft.fftn(self.signal)
        
        return shift
    
    
        
    def gauss_conv(self,sigma):
        
        """
        
        Subroutine: applies Gaussian convolution to signal
        
        Arguments
        ---------
        
        sigma: float
            width of Gaussian (pixels)
            
        """ 
            
        # make list of axis coodinates
        axes=[]
        for i in self.signal.shape:
            axes.append(np.arange(i)-(i-1)/2.)
                        
        # make r squared array
        if len(self.signal.shape)>1:
            r_sqd=np.zeros(self.signal.shape)
            for j in np.meshgrid(*axes,indexing="ij"):
                r_sqd+=j**2
        else:
            r_sqd=axes[0]**2
        
        # generate kernel function
        kernel=np.zeros(self.signal.shape,dtype=np.complex)
        kernel.real=np.exp(-r_sqd/(2.*sigma**2))
        
        # normalise and shift kernel
        kernel.real/=np.sum(kernel.real)
        offset=(np.array(kernel.shape)+np.mod(np.array(kernel.shape),2))//2
        axis=np.arange(len(kernel.shape))
        kernel=np.roll(kernel,offset,axis)
        
        # convolve signal with kernel
        self.spectrum=np.fft.fftn(self.signal)
        self.spectrum*=np.fft.fftn(kernel)
        self.signal=np.fft.ifftn(self.spectrum)
        
        return
        
    def power_spectrum(self,normalise=False):
        
        """
        Function: returns radially averaged power spectrum
        
        Result
        ------
        
        k_bins[:]: float
            k values
        
        power_spec[:]: float
            power spectrum
        
        """
        
        # make k coordinate arrays
        k_list=[]
        for i in self.spectrum.shape:
            k_list.append(np.arange(-i//2+np.mod(i,2),(i-1)//2+1))
            print(np.arange(-i//2+np.mod(i,2),(i-1)//2+1).shape)

        # generate array of k values
        if len(self.spectrum.shape)>1:
            k=np.zeros(self.spectrum.shape)
            for i in np.meshgrid(*k_list,indexing="ij"):
                k+=i**2
        else:
            k=k_list[0]**2
        k=np.sqrt(k)
        
        # roll spectrum to align with k values
        offset=(np.array(self.spectrum.shape)+np.mod(np.array(self.spectrum.shape),2))//2
        axis=np.arange(len(self.spectrum.shape))
        rolled_spectrum=np.roll(self.spectrum,-offset,axis)
        
        # flatten arrays
        rolled_spectrum=np.ndarray.flatten(rolled_spectrum)
        k=np.ndarray.flatten(k)
        
        k_bins=np.arange(1,np.int(k.max()))
        print(k_bins.shape)
        
        power_spec=np.histogram(k,k_bins,weights=np.abs(rolled_spectrum)**2)
    
        power_spec=power_spec[0]/k_bins[:-1]**(len(self.spectrum.shape)-1)
        
        if normalise:
            power_spec/=np.sum(power_spec)
        
        return k_bins[:-1], power_spec

class vector_grf:
    
    """
    Class for building nd gaussian random vector field
    
    
    Data
    ----
    
    spectrum[...]: float
        spectrum of gaussian random field
    
    signal[...]: float
        signal of gaussian random field
    
    """
        
        
    def __init__(self,n_shape_in,beta,real_field=True,field_type="isotropic"):
        
        """
        Subroutine: generate a nd vector gaussian random field with a power spectrum
        
        Arguments
        ---------
        
        n_shape_in(,): tuple, int
            shape of field
            
        beta: float
            power spectrum exponent
        
        real_field: boolean
            is field real?
            
        field_type: string
            isotropic (defualt):    ndim independent scalar fields
            thermal:                thermal mix of solenoidal and compressive modes
            solenoidal:             purely solenoidal (div free) field
            compressive:            purely compressive field (curl free) field
        
        """
        
        # check for valid field_type
        field_type_list=("isotropic","thermal","solenoidal","compressive")
        if all([not x==field_type for x in field_type_list]):
            print("Unknown field_type:",field_type)
            print("Setting field type to isotropic")
            field_type="isotropic"
        
            
        
        # make n_shape array-like, if scalar
        if isinstance(n_shape_in,(tuple,list,np.ndarray)):
            n_shape=n_shape_in
        else:
            n_shape=(n_shape_in,)
            
            
        # make spectrum array
        self.spectrum=np.zeros((*(n_shape),len(n_shape)),dtype=np.complex)
        for i in range(len(n_shape)):
            self.spectrum[...,i]=scalar_grf(n_shape,beta,real_field,generate_signal=False).spectrum
            
        # make signal array
        self.signal=np.zeros(self.spectrum.shape,dtype=np.complex)
            
        if field_type=="isotropic":
            
            for i in range(len(n_shape)):
                self.signal[...,i]=self.spectrum[...,i].size*np.fft.ifftn(self.spectrum[...,i])
                
        else:
            
            # make amplitide array
            A_k=np.random.normal(size=self.spectrum.shape)
            
            if field_type!="thermal":
                
                # generate k-vector direction array
                # make k coordinate arrays            
                k_list=[]
                for i in range(len(n_shape)):
                    k_list.append(np.ceil(np.arange(-n_shape[i]/2,n_shape[i]/2)))
                    k_list[i]=np.roll(k_list[i],np.ceil(-n_shape[i]/2).astype(np.int))
                    
                # generate array of k values
                k=np.zeros(self.spectrum.shape)
                k_grid=np.meshgrid(*k_list,indexing="ij")
                for i in range(len(k_grid)):
                    k[...,i]=k_grid[i]
                    
                # Normalise wavevectors
                k_mag=np.sqrt((k**2).sum(axis=k.ndim-1))
                # avoid div0
                k_mag[tuple([0]*k_mag.ndim)]=1.
                k/=k_mag.reshape((*k_mag.shape,1))
                
                
                if field_type=="compressive":
                    
                    A_dot_k=(A_k*k).sum(axis=A_k.ndim-1)
                    A_k=k*A_dot_k.reshape((*A_dot_k.shape,1))
                    
                if field_type=="solenoidal":
                    
                    A_dot_k=(A_k*k).sum(axis=A_k.ndim-1)
                    A_k=A_k-k*A_dot_k.reshape((*A_dot_k.shape,1))
                    
            # Normalise amplitude array
            A_k_mag=np.sqrt((A_k**2).sum(axis=A_k.ndim-1))
            # avoid div0
            A_k_mag[tuple([0]*A_k_mag.ndim)]=1.
            A_k/=A_k_mag.reshape((*A_k_mag.shape,1))
            self.spectrum*=A_k
            
            for i in range(len(n_shape)):
                if real_field:
                    self.signal[...,i]+=self.spectrum[...,i].size*np.fft.irfftn(self.spectrum[...,i],s=self.spectrum[...,i].shape)
                else:
                    self.signal[...,i]=self.spectrum[...,i].size*np.fft.ifftn(self.spectrum[...,i])
                
                
        return
    
    
    def conv_shift(self,weight=None):
        
        """
        
        Subroutine: shifts field to make it maximally convergent on centre
        
        Arguments
        ---------
        
        weight[...]: float
            array of weights 
            
        
        """
        
        # set weights
        if weight is None:
            weight=np.ones(self.signal.shape[:-1])
        
        # make array of radial direction vectors and multiply by weights
        u_r=-1.*u_r_grid(self.signal.shape[:-1])*weight[...,np.newaxis]
        
        # caluclate total radial momentum towards each point of grid
        tot_mom=np.zeros(self.signal.shape[:-1])
        for i in range(self.signal.ndim-1):
            tot_mom+=nd_convolve(self.signal.real[...,i],u_r[...,i],normalise_kernel=False,periodic=True,correlate=True)
            
        # highest convergent momentum value
        shift=np.argwhere(tot_mom==tot_mom.max())[0,:]
            
        # set shifts relative to array size
        shift=tuple(np.array(self.signal.shape[:-1])//2-shift)
        
        # roll signal array
        self.signal=np.roll(self.signal,shift,tuple(range(len(shift))))

            
        return
