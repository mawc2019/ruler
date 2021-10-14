import numpy as np
# Some code is copied or adapted from the Meep code at https://github.com/smartalecH/meep/blob/jax_rebase/python/adjoint/filters.py 

class morph:
    def __init__(self,Lx,Ly,proj_strength=10**6):
        
        self.Lx = Lx
        self.Ly = Ly
        self.proj_strength = proj_strength
        
    def cylindrical_filter(self,arr,radius):
        
        (Nx,Ny) = arr.shape
        # Formulate grid over entire design region
        xv, yv = np.meshgrid(np.linspace(-self.Lx/2,self.Lx/2,Nx), np.linspace(-self.Ly/2,self.Ly/2,Ny), sparse=True, indexing='ij')

        # Calculate kernel
        kernel = np.where(np.abs(xv ** 2 + yv ** 2) <= radius**2,1,0)#.T

        # Normalize kernel
        kernel = kernel / np.sum(kernel.flatten()) # Normalize the filter

        # Filter the response
        arr_out = simple_2d_filter(arr,kernel,Nx,Ny)

        return arr_out
    
    def heaviside_erosion(self,arr,radius):
        
        (Nx,Ny) = arr.shape
        beta = self.proj_strength
        arr_hat = self.cylindrical_filter(arr,radius)#.flatten()
        
        return np.exp(-beta*(1-arr_hat)) + np.exp(-beta)*(1-arr_hat)

    def heaviside_dilation(self,arr,radius):
        
        (Nx,Ny) = arr.shape
        beta = self.proj_strength
        arr_hat = self.cylindrical_filter(arr,radius)#.flatten()
        
        return 1 - np.exp(-beta*arr_hat) + np.exp(-beta)*arr_hat
    
    def open_operator(self,arr,radius):
        
        he = self.heaviside_erosion(arr,radius)
        hdhe = self.heaviside_dilation(he,radius)
        
        return hdhe
    
    def close_operator(self,arr,radius):
        
        hd = self.heaviside_dilation(arr,radius)
        hehd = self.heaviside_erosion(hd,radius)
        
        return hehd
    
    def minimum_length(self,arr,len_arr): # search the minimum length scale of an image "arr" within a length array "len_arr"
        
        radius_list = sorted(list(np.abs(len_arr)/2))
        for radius in radius_list:
            diff_image = np.abs(self.open_operator(arr,radius)-self.close_operator(arr,radius))
            pixel_ex,pixel_in = pixel_count(diff_image,threshold=0.5)
            if pixel_in>0:
                print("The minimum length scale is ",radius*2)
                return radius*2
            
        print("The minimum length scale is not within this array of lengths.")
        return


def simple_2d_filter(arr,kernel,Nx,Ny):
    # Get 2d parameter space shape
    (kx,ky) = kernel.shape

    # Ensure the input is 2D
    arr = arr.reshape(Nx,Ny)

    # pad the kernel and input to avoid circular convolution and
    # to ensure boundary conditions are met.
    kernel = _zero_pad(kernel,((kx,kx),(ky,ky)))
    arr = _edge_pad(arr,((kx,kx),(ky,ky)))
    
    # Transform to frequency domain for fast convolution
    K = np.fft.fft2(kernel)
    A = np.fft.fft2(arr)
    
    # Convolution (multiplication in frequency domain)
    KA = K * A
    
    # We need to fftshift since we padded both sides if each dimension of our input and kernel.
    arr_out = np.fft.fftshift(np.real(np.fft.ifft2(KA)))
    
    # Remove all the extra padding
    arr_out = _centered(arr_out,(kx,ky))
    return arr_out

def _centered(arr, newshape):
    '''Helper function that reformats the padded array of the fft filter operation.

    Borrowed from scipy:
    https://github.com/scipy/scipy/blob/v1.4.1/scipy/signal/signaltools.py#L263-L270
    '''
    # Return the center newshape portion of the array.
    newshape = np.asarray(newshape)
    currshape = np.array(arr.shape)
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]

def _edge_pad(arr, pad):
    
    # fill sides
    left = np.tile(arr[0,:],(pad[0][0],1)) # left side
    right = np.tile(arr[-1,:],(pad[0][1],1)) # right side
    top = np.tile(arr[:,0],(pad[1][0],1)).transpose() # top side
    bottom = np.tile(arr[:,-1],(pad[1][1],1)).transpose() # bottom side)
    
    # fill corners
    top_left = np.tile(arr[0,0], (pad[0][0],pad[1][0])) # top left
    top_right = np.tile(arr[-1,0], (pad[0][1],pad[1][0])) # top right
    bottom_left = np.tile(arr[0,-1], (pad[0][0],pad[1][1])) # bottom left
    bottom_right = np.tile(arr[-1,-1], (pad[0][1],pad[1][1])) # bottom right
    
    out = np.concatenate((
        np.concatenate((top_left,top,top_right)),
        np.concatenate((left,arr,right)),
        np.concatenate((bottom_left,bottom,bottom_right))    
    ),axis=1)
    
    return out

def _zero_pad(arr, pad):
    
    # fill sides
    left = np.tile(0,(pad[0][0],arr.shape[1])) # left side
    right = np.tile(0,(pad[0][1],arr.shape[1])) # right side
    top = np.tile(0,(arr.shape[0],pad[1][0])) # top side
    bottom = np.tile(0,(arr.shape[0],pad[1][1])) # bottom side
    
    # fill corners
    top_left = np.tile(0, (pad[0][0],pad[1][0])) # top left
    top_right = np.tile(0, (pad[0][1],pad[1][0])) # top right
    bottom_left = np.tile(0, (pad[0][0],pad[1][1])) # bottom left
    bottom_right = np.tile(0, (pad[0][1],pad[1][1])) # bottom right
    
    out = np.concatenate((
        np.concatenate((top_left,top,top_right)),
        np.concatenate((left,arr,right)),
        np.concatenate((bottom_left,bottom,bottom_right))    
    ),axis=1)
    
    return out

def pixel_count(arr,threshold):
    pixel_tot,pixel_int = 0,0 # total number of pixels with values above the threshold, and the number of interior pixels
    for ii in range(arr.shape[0]):
        for jj in range(arr.shape[1]):
            if arr[ii,jj]>threshold:
                pixel_tot += 1
                if ii>0 and ii<arr.shape[0]-1 and jj>0 and jj<arr.shape[1]-1:
                    if arr[ii-1,jj]>threshold and arr[ii+1,jj]>threshold and arr[ii,jj-1]>threshold and arr[ii,jj+1]>threshold:
                        pixel_int += 1

    return pixel_tot-pixel_int,pixel_int # numbers of exterior and interior pixels
