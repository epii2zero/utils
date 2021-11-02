import torch
import torch.nn as nn
import torch.nn.functional as F
from .vector_quantizer import VectorQuantizer, VectorQuantizerEMA

class Multi_stage_VQ(nn.Module):
    """    
        Args:
            embedding_dim: integer representing the dimensionality of the tensors in the
              quantized space. Inputs to the modules must be in this format as well.
            num_embeddings: integer, the number of vectors in the quantized space.
            commitment_cost: scalar which controls the weighting of the loss terms(Beta in paper).
            
        Input: 
            data: Tensor (Batch, *, embedding_Dim) to be quantized.
        Output: 
            quantized: Tensor (Batch, *, embedding_Dim) containing quantized data.
            encodings: Tensor (Batch, *, Num_embeddings) containing encodings with one-hot vector.
            loss: Tensor containing loss to optimize.
            perplexity: Tensor containing the perplexity of the encodings.
    """
    def __init__(self, embedding_dim, commitment_cost, embedding_structure,
                 mode='normal', decay=0.99):
        super(Multi_stage_VQ, self).__init__()
        
        self._mode = mode
        self._embedding_dim = embedding_dim
        self._commitment_cost = commitment_cost # or Beta
        self._embedding_structure = embedding_structure
        # Structing
        self.layers = nn.ModuleList()
        for stage in embedding_structure:
            # Single VQ
            if type(stage) == type(1):
                if self._mode == 'normal':
                    self.layers.append(VectorQuantizer(num_embeddings=stage,
                                                       embedding_dim=self._embedding_dim,
                                                       commitment_cost=self._commitment_cost))
                elif self._mode == 'EMA':
                    pass
                                    
                else:
                    raise NotImplementedError("")
                
            # Split-VQ
            elif type(stage) == type([]):
                splits = len(stage)
                if embedding_dim % splits:
                    raise ValueError('Can not split embedding dimension')
                if self._mode == 'normal':
                    sub_layer = nn.ModuleList()
                    for sub in stage:
                        sub_layer.append(VectorQuantizer(num_embeddings=sub,
                                                         embedding_dim=self._embedding_dim//splits,
                                                         commitment_cost=self._commitment_cost))
                    self.layers.append(sub_layer)
        
    def entropy_forward(self, data, using=0):
        # check input shape
        # Expected: (Batch, *, embedding_Dim)
        input_shape = data.shape
        if input_shape[-1] != self._embedding_dim:
            raise ValueError("Expected input last dimension {}, but got {}"\
                             .format(self._embedding_dim, input_shape[-1]))
        
        flat_input = data.reshape(-1, self._embedding_dim) # (B x *, D)
        stage_cnt = 0
        total_encodings = []
        total_loss = 0
        total_entropy = 0
        
        quantized = torch.zeros(flat_input.shape, device=flat_input.device)
        for stage in self.layers:
            # Single VQ
            if type(stage) != type(self.layers):
                sub_quantized, encodings, loss, entropy = stage.entropy_forward(flat_input - quantized)
                quantized += sub_quantized
                total_encodings.append(encodings)
                total_loss += loss
                total_entropy += entropy
#                 print('quantized embedding:', quantized)
#                 print('one-hat encodings:', encodings)
#                 print('indices: ', torch.argmax(encodings, dim=1))
#                 print('loss:', loss)
#                 print('total_loss:', total_loss)
#                 print('entropy:', entropy)
                stage_cnt += 1
            # Split-VQ
            else:
                splits = len(stage)
                split_dim = self._embedding_dim // splits
                split_encodings = []
                for i, split in enumerate(stage):
                    split_quantized, split_encoding, split_loss, split_entropy = \
                        split.entropy_forward(flat_input[:, split_dim*i:split_dim*(i+1)] - quantized[:, split_dim*i:split_dim*(i+1)])
                    quantized[:, split_dim*i:split_dim*(i+1)] += split_quantized
                    split_encodings.append(split_encoding)
                    total_loss += split_loss
                    total_entropy += split_entropy
                stage_cnt += 1
                    
            if stage_cnt == using:
                break
                
        quantized = quantized.reshape(input_shape)
        return quantized, total_encodings, total_loss, total_entropy
    
    def forward(self, data, using=0):
        return self.entropy_forward(data, using)
    
    def extra_repr(self):
        return str(self._embedding_structure) + " codebook size" 