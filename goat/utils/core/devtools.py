def print_tensor_shape(inputs):
    if isinstance(inputs,list) or isinstance(inputs,tuple):
        for value in inputs:
            print(value.shape)
    else:
        print(inputs.shape)
        
def Convert_IMGTensor_To_Numpy(tensor):
    assert tensor.shape[1]==3
    return tensor.squeeze(0).permute(1,2,0).cpu().numpy()