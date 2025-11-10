import torch
from torch import nn
import torch.nn.functional as F
import math
from layers.RevIN import RevIN
from layers.Layer import Flatten_Head
import numpy as np



class FourierPolyMask(nn.Module):
    def __init__(self, input_dim, output_dim, fourier_degree, poly_degree):
        super(FourierPolyMask, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fourier_degree = fourier_degree
        self.poly_degree = poly_degree
        
        # Define learnable coefficients
        self.cos_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, self.fourier_degree + 1))
        self.sin_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, self.fourier_degree))
        self.poly_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, poly_degree + 1))


        # Initialize coefficients
        nn.init.normal_(self.cos_coeffs, mean=0.0, std=1 / (input_dim * (self.fourier_degree + 1)))
        nn.init.normal_(self.sin_coeffs, mean=0.0, std=1 / (input_dim * self.fourier_degree))
        nn.init.normal_(self.poly_coeffs, mean=0.0, std=1 / input_dim *(poly_degree + 1))
        
        # Initialize coefficients
        self.register_buffer("k_cos", torch.arange(0, self.fourier_degree + 1, 1))  
        self.register_buffer("k_sin", torch.arange(1, self.fourier_degree + 1, 1))  
        self.register_buffer("k_poly", torch.arange(0, poly_degree + 1, 1))
        
        self.interaction = nn.Linear(input_dim, output_dim)
        self.bias = nn.Parameter(torch.zeros(1, output_dim))

    def forward(self, x):
        x_cos = x.unsqueeze(-1) * self.k_cos * torch.pi  # [batch_size, input_dim, degree + 1]
        x_cos = torch.cos(x_cos)
        
        x_sin = x.unsqueeze(-1) * self.k_sin * torch.pi  # [batch_size, input_dim, degree]
        x_sin = torch.sin(x_sin)

        x_p = x.unsqueeze(-1)
        x_poly = x_p.pow(self.k_poly)
        
        y_cos = torch.einsum("bid,iod->bo", x_cos, self.cos_coeffs)  
        y_sin = torch.einsum("bid,iod->bo", x_sin, self.sin_coeffs)  
        y_poly = torch.einsum("bid,iod->bo", x_poly, self.poly_coeffs)
        
        y = y_cos + y_sin + y_poly

        y = y +self.bias
        y = self.interaction(y)

        return y

class AdaFE(nn.Module):
    def __init__(self, dmodel,degree,p_degree, kernel_size, padding=0):

        super(AdaFE, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.mask = FourierPolyMask(dmodel,dmodel, degree,p_degree)

    def forward(self, x):

        Mask = self.mask(x)
        threshold = Mask.mean()
        
        # Generate binary mask based on the set threshold
        active = (Mask > threshold).float()  
       
        # Pad the input and activation mask to ensure the convolution window covers the boundaries
        x_pad = F.pad(x, (self.padding, self.padding))
        active_pad = F.pad(active, (self.padding, self.padding))

        # Use unfold to extract sliding windows      
        x_unfold = x_pad.unfold(dimension=1, size=self.kernel_size, step=1)
        active_unfold = active_pad.unfold(dimension=1, size=self.kernel_size, step=1)     
        x_mask = x_unfold * active_unfold 
       
        # Calculate convolution for all positions
        out = torch.einsum('blk,k->bl', x_mask, self.weight)  
        
        return out
    


class DSFFN(nn.Module):
    def __init__(self, dmodel, ffr,drop=0.1):

        super(DSFFN, self).__init__()
        dff = dmodel * ffr
        self.pw1 = nn.Conv1d(in_channels=dmodel, out_channels=dff, kernel_size=1, stride=1,
                                 padding=0, dilation=1)
        self.act = nn.GELU()
        self.combine = nn.Conv1d(dff + dmodel, dff, kernel_size=1)
        self.pw2 = nn.Conv1d(in_channels=dff, out_channels=dmodel, kernel_size=1, stride=1,
                                 padding=0, dilation=1)
        self.drop = nn.Dropout(drop)

    def forward(self,x):

        B, M, D, N  = x.shape
        x = x.reshape(B*M,D,N)

        local_features = self.act(self.pw1(x))
        global_features = torch.mean(x, dim=2, keepdim=True).expand(-1, -1, x.size(2))
        
        # Combine local and global features
        combined = torch.cat([local_features, global_features], dim=1)        
        x = self.combine(combined)

        x = self.drop(self.pw2(x))   
        out = x.reshape(B, M,D, N)
        
        return out



class Model(nn.Module):
    def __init__(self,configs):
        super(Model, self).__init__()

        self.ffn_ratio = configs.ffn_ratio
        self.d_model = configs.d_model

        self.nvars = configs.enc_in
        self.drop_backbone = configs.dropout
        self.drop_head = configs.head_dropout
        self.revin = configs.revin
        self.affine = configs.affine
        self.seq_len = configs.seq_len
        self.individual = configs.individual
        self.target_window = configs.pred_len

        self.patch_size = configs.patch_size
        self.patch_stride = configs.patch_stride
        self.degree = configs.degree
        self.p_degree = configs.p_degree
        self.subtract_last = configs.subtract_last

        if self.revin:
            self.revin_layer = RevIN(self.nvars, affine=self.affine, subtract_last=self.subtract_last)
        
        self.patch_num = self.seq_len // self.patch_stride
        self.head_nf = self.d_model * self.patch_num
        self.head = Flatten_Head(self.individual, self.nvars, self.head_nf, self.target_window,
                                     head_dropout=self.drop_head)

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.patch_stride))      
        self.emb = nn.Linear(self.patch_size,self.d_model)   
        self.layernorm = nn.LayerNorm(self.d_model)
        self.AdaFE = AdaFE(self.d_model,self.degree,self.p_degree, kernel_size=5, padding=2)
        self.dsffn = DSFFN(self.d_model,self.ffn_ratio,self.drop_backbone)


    def forward(self, x):

        x = x.permute(0, 2, 1)

        if self.revin:
            x = x.permute(0, 2, 1)
            x = self.revin_layer(x, 'norm')
            x = x.permute(0, 2, 1)

        # Patch Embedding       
        if self.patch_size != self.patch_stride:
            x = self.padding_patch_layer(x)

        x = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)
        x = self.emb(x).permute(0,1,3,2)

        # Process
        res = x
        B, M, D, N  = x.shape

        x = x.permute(0, 1, 3, 2) 
        x = x.reshape(B * M * N, D)
        x = self.AdaFE(x)            
        x = self.layernorm(x)
        x = x.reshape(B, M, N, D)
        x = x.permute(0, 1, 3, 2) 
        x = self.dsffn(x)
        x = x + res
        x = self.head(x)
        
        if self.revin:
            x = x.permute(0, 2, 1)
            x = self.revin_layer(x, 'denorm')
            x = x.permute(0, 2, 1)

        x = x.permute(0, 2, 1)
        return x

