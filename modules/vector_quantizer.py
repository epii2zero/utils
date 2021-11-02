import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
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
            
        References:
            https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/nets/vqvae.py
            https://github.com/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb

    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
    
        
        # embedding structure
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        
        # embedding layer
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        # uniform initialization of latent
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost # or Beta
        
    def forward(self, data):
        # check input shape
        # Expected: (Batch, *, embedding_Dim)
        input_shape = data.shape
        if input_shape[-1] != self._embedding_dim:
            raise ValueError("Expected input last dimension {}, but got {}"\
                             .format(self._embedding_dim, input_shape[-1]))
        
        flat_input = data.reshape(-1, self._embedding_dim) # (B x *, D)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # encode
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=data.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).reshape(input_shape) # (B, *, D)
        
        # Loss
        q_latent_loss = F.mse_loss(quantized, data.detach()) # VQ objective
        e_latent_loss = F.mse_loss(quantized.detach(), data) # commitment loss
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        # pass decoder gradient to encoder
        quantized = data + (quantized - data).detach()
        
        # Evaluate codebook usage (perplexity)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # Reshape encodings (Batch, *, Num_embeddings) to return
        encodings = encodings.reshape(list(input_shape[:-1]) + [self._num_embeddings])
        
        return quantized, encodings, loss, perplexity
    
    def entropy_forward(self, data):
        # check input shape
        # Expected: (Batch, *, embedding_Dim)
        input_shape = data.shape
        if input_shape[-1] != self._embedding_dim:
            raise ValueError("Expected input last dimension {}, but got {}"\
                             .format(self._embedding_dim, input_shape[-1]))
        
        flat_input = data.reshape(-1, self._embedding_dim) # (B x *, D)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # encode
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=data.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).reshape(input_shape) # (B, *, D)
        
        # Loss
        q_latent_loss = F.mse_loss(quantized, data.detach()) # VQ objective
        e_latent_loss = F.mse_loss(quantized.detach(), data) # commitment loss
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        # pass decoder gradient to encoder
        quantized = data + (quantized - data).detach()
        
        # Evaluate entropy
        avg_probs = torch.mean(encodings, dim=0)
        entropy = -torch.sum(avg_probs * torch.log2(avg_probs + 1e-10))
        
        # Reshape encodings (Batch, *, Num_embeddings) to return
        encodings = encodings.reshape(list(input_shape[:-1]) + [self._num_embeddings])
        
        return quantized, encodings, loss, entropy
    
    @property
    def embedding(self):
        return self._embedding
        

################################################################################################## 
# moving average method  
class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), entropy, encodings