import numpy as np

def crop_pad(old_field,new_shape,pad_value=0.,centred=False):
    
    """
    
    Fucntion: crops/pads a field to a new shape
    
    Arguments
    ---------
    
    old_field[...]: float
        ndarray of field
        
    new_shape[]: int
        shape of new field. len(new_shape)=old_field.ndim
        
    pad_value: float
        pad unused values of new field with this number
        
    centred: boolean
        make sure element N//2 of old field is index N//2 of new field
        
    Result
    ------
    
    new_field[...]: float
        ndarray of field with shape=new_shape
    
    """
    
    # initialise new field
    new_field=np.full(new_shape,pad_value)
    
    # make slice objects
    old_indices=[slice(None)]*old_field.ndim
    new_indices=[slice(None)]*new_field.ndim
    
    # loop over axes and adjust array bounds
    for i in range(old_field.ndim):
        
        if centred:
            # work out displacement between two central indices
            displacement=np.abs(old_field.shape[i]//2-new_field.shape[i]//2)
        else:
            displacement=0
        
        if old_field.shape[i]>new_field.shape[i]:
            # old field must be cropped
            old_indices[i]=slice(displacement,displacement+new_field.shape[i])
        elif old_field.shape[i]<new_field.shape[i]:
            # old field must be padded
            new_indices[i]=slice(displacement,displacement+old_field.shape[i])
            
    # copy old field to new new_field
    new_field[new_indices]=old_field[old_indices]
    
    return new_field
        
    
    
    
    
    
    