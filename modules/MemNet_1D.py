import torch
import torch.nn as nn
import torch.nn.functional as F

class MemNet_1D(nn.Module):
    """    
        References:
            https://github.com/wutianyiRosun/MemNet/blob/master/memnet1.py

    """
    def __init__(self, in_channels, channels, kernel_size, num_memblock, num_recursion):
        super(MemNet_1D, self).__init__()
        self.feature_extractor = BNReLUConv(in_channels=in_channels, out_channels=channels, kernel_size=kernel_size)
        self.dense_memory = nn.ModuleList(
            [MemoryBlock(channels=channels, kernel_size=kernel_size, num_recursion=num_recursion, memblock_idx=i)
             for i in range(num_memblock)]
        )
        self.reconstructor = BNReLUConv(in_channels=channels, out_channels=in_channels, kernel_size=kernel_size)
        # multi-supervision weight
        self.weights = nn.Parameter((torch.ones(num_memblock)/num_memblock), requires_grad=True)
        
    def forward(self, x):
        residual = x
        out = self.feature_extractor(x)
        w_sum = self.weights.sum()
        mid_features = []
        long_term_memory = [out]
        for memory_block in self.dense_memory:
            out = memory_block(out, long_term_memory)
            mid_features.append(out)
        # weight averaging for final output
        sub_pred = []
        for i in range(len(mid_features)):
            sub_pred.append(self.reconstructor(mid_features[i]) + residual)
        final_pred = 0
        for i in range(len(mid_features)):
            final_pred = final_pred + sub_pred[i] * self.weights[i] / w_sum

        return final_pred, sub_pred

class MemoryBlock(nn.Module):
    def __init__(self, channels, kernel_size, num_recursion, memblock_idx):
        super(MemoryBlock, self).__init__()
        self.recursive_unit = nn.ModuleList(
            [ResidualBlock(channels=channels, kernel_size=kernel_size) for i in range(num_recursion)]
        )
        self.gate_unit = BNReLUConv(in_channels=(num_recursion + memblock_idx + 1) * channels,
                                    out_channels=channels, kernel_size=1)
        
    def forward(self, out, long_term_memory):
        """
            out is output of previous memory block
            xs means s
        """
        short_term_memory = []
        for layer in self.recursive_unit:
            out = layer(out)
            short_term_memory.append(out)
            
        gate_out = self.gate_unit(torch.cat(short_term_memory+long_term_memory, 1))
        long_term_memory.append(gate_out)
        return gate_out

class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size):
        super(ResidualBlock, self).__init__()
        padding = kernel_size // 2
        self.relu_conv1 = BNReLUConv(in_channels=channels, out_channels=channels, kernel_size=kernel_size)
        self.relu_conv2 = BNReLUConv(in_channels=channels, out_channels=channels, kernel_size=kernel_size)
        
    def forward(self, x):
        residual = x
        out = self.relu_conv1(x)
        out = self.relu_conv2(out)
        out = out + residual
        return out
        
        
class BNReLUConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(BNReLUConv, self).__init__()
        padding = kernel_size // 2
        self.bn=nn.BatchNorm1d(in_channels)
        self.relu=nn.ReLU()
        self.conv=nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=kernel_size, stride=1, padding=padding)
        
    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        return x
################################################################################################## 
