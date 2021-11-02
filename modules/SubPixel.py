import torch.nn as nn
import torch.nn.functional as F

class SubPixel1D(nn.Module):
    """
        Implementation of 1D subpixel convolution interacing part
    """
    def __init__(self, scale_factor):
        super(SubPixel1D, self).__init__()
        self.n = scale_factor
        
    def forward(self, data):
        """ in_data: [batch_size, channel*n, num_samples]              
            out_data: [batch_size, channel, num_samples*n] 
        """           
                                                                        # [B, C*n, T]
        r = data.transpose(2, 1)                                        # [B, T, C*n]
        r = r.reshape(-1, data.size(2), data.size(1)//self.n, self.n)   # [B, T, C, n]
        r = r.transpose(3, 2)                                           # [B, T, n, C]
        r = r.reshape(-1, data.size(2)*self.n, data.size(1)//self.n)    # [B, T*n, C]
        r = r.transpose(2, 1)                                           # [B, C, T*n]
        return r
    
class Interacing1D(nn.Module):
    """
        Implementation of 1D interacing (from subpixel convolution)
        API contains both forward and reverse operation
    """
    def __init__(self, scale_factor):
        super(Interacing1D, self).__init__()
        self.sf = scale_factor
        
    def forward(self, data):
        """ in_data: [batch_size, channel*n, num_samples]              
            out_data: [batch_size, channel, num_samples*n] 
        """
        if self.sf == 1:
            return data
        dshape = data.shape                                              # [B, C*n, T]
        r = data.reshape(-1, dshape[1]//self.sf, self.sf, dshape[2])     # [B, C, n, T]
        r = r.transpose(2, 3)                                            # [B, C, T, n]
        r = r.reshape(-1, dshape[1]//self.sf, dshape[2]*self.sf)         # [B, C, T*n]
        return r
    
    def reverse(self, data):
        """ in_data: [batch_size, channel, num_samples*n]
            out_data: [batch_size, channel*n, num_samples]
        """
        if self.sf == 1:
            return data
        dshape = data.shape                                              # [B, C, T*n]
        r = data.reshape(-1, dshape[1], dshape[2]//self.sf, self.sf)     # [B, C, T, n]
        r = r.transpose(2, 3)                                            # [B, C, n, T]
        r = r.reshape(-1, dshape[1]*self.sf, dshape[2]//self.sf)         # [B, C*n, T]
        return r
    
    def cx_forward(self, data):
        """ in_data: [batch_size, n*channel, num_samples]
            out_data: [batch_size, channel, num_samples*n]
        """
        if self.sf == 1:
            return data
        dshape = data.shape                                              # [B, n*C, T]
        r = data.reshape(-1, self.sf, dshape[1]//self.sf, dshape[2])     # [B, n, C, T]
        r = r.permute(0, 2, 3, 1)                                        # [B, C, T, n]
        r = r.reshape(-1, dshape[1]//self.sf, dshape[2]*self.sf)         # [B, C, T*n]
        return r
    
    def cx_reverse(self, data):
        """ in_data: [batch_size, channel, num_samples*n]
            out_data: [batch_size, n*channel, num_samples]
        """
        if self.sf == 1:
            return data
        dshape = data.shape                                              # [B, C, T*n]
        r = data.reshape(-1, dshape[1], dshape[2]//self.sf, self.sf)     # [B, C, T, n]
        r = r.permute(0, 3, 1, 2)                                        # [B, n, C, T]
        r = r.reshape(-1, self.sf*dshape[1], dshape[2]//self.sf)         # [B, n*C, T]
        return r