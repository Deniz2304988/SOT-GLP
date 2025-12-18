from typing import Type, List

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss
from torch import Tensor

from sot_glp.models.tools.topk_reduce import topk_reduce

NoneType = Type[None]


def global_weighter(global_logits,local_logits):
    #global_min, _ = global_logits.min(dim=1, keepdim=True)
    #global_max, _ = global_logits.max(dim=1, keepdim=True)
    #global_weights = (global_logits - global_min) / (global_max - global_min + 1e-8)
    global_weights = global_logits.mean(dim=-1)
    return global_weights.unsqueeze(1).unsqueeze(-1).repeat(1,local_logits.shape[1],1,local_logits.shape[-1])

def local_weighter(local_logit):
    local_min, _ = local_logit.min(dim=2, keepdim=True)
    local_max, _ = local_logit.max(dim=2, keepdim=True)
    # Local Weight dimension is B,Token_Size,Num_Classes
    local_weights = (local_logit - local_min) / (local_max - local_min + 1e-8)
    return local_weights
    
    


class GLSotLoss(_WeightedLoss):

    def __init__(
        self,
        use_global_loss: bool = True,
        use_local_loss: bool = True,
        topk: List[int] = [5],
        global_dropout_p: float = 0.75,
    ) -> NoneType:
        super().__init__()

        self.use_global_loss = use_global_loss
        self.use_local_loss = use_local_loss
        self.topk = topk
        self.global_dropout_p = global_dropout_p
        self.eps = 0.1
        self.max_iter = 100
        self.reg_m = 100
    
    def Sinkhorn(self, K, u, v):
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = 1e-2
        for i in range(self.max_iter):
            r0 = r
            r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
            c = v / torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1)
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break

        T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K

        return T
    
    def sinkhorn_unbalanced(self, K, u, v):
        """
        The Unbalanced Sinkhorn algorithm.

        Args:
            K (torch.Tensor): The Gibbs kernel, K = exp(-C/eps).
            u (torch.Tensor): The marginals of the first distribution.
            v (torch.Tensor): The marginals of the second distribution.

        Returns:
            torch.Tensor: The computed transport plan T.
        """
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        
        # Pre-compute the exponent for the update steps.
        # This is the core modification for unbalanced transport.
        power = self.reg_m / (self.reg_m + self.eps)
        
        thresh = 1e-2
        for _ in range(self.max_iter):
            r0 = r
            
            # --- Modified Sinkhorn Updates ---
            # Update r based on c
            r = (u / (K @ c.unsqueeze(-1)).squeeze(-1))**power
            
            # Update c based on the new r
            # K.transpose(-2, -1) is equivalent to K.permute(0, 2, 1) for 3D tensors
            c = (v / (K.transpose(-2, -1) @ r.unsqueeze(-1)).squeeze(-1))**power
            
            # Check for convergence
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break

        # Compute the final transport plan T
        T = r.unsqueeze(-1) * K * c.unsqueeze(-2)
        return T

    def forward(
        self,
        global_logits: Tensor,
        local_logits: Tensor,
        targets: Tensor,
        logit_scale: float,
    ) -> Tensor:
        """
        global_logits is a Tensor of shape (b, k, 1) or (b, k, n)
        local_logits is a Tensor of shape (b, p, k, 1) or (b, p, k, m)
        """
        global_loss = local_loss = 0.

        if self.use_local_loss and local_logits is not None:
            maxk = max(self.topk)
            B, N, C, CH = local_logits.shape  # CH should be 4

            # 1) Mean over the last dim (only for ranking)
            mean_for_ranking = local_logits.mean(dim=-1)        # (B, N, C)

            # 2) Top-k along N using the mean
            maxk = int(max(self.topk))                          # or just your desired k
            mean_topk_vals, topk_idx = torch.topk(
                mean_for_ranking, k=maxk, dim=1, largest=True, sorted=True
            )                                                    # shapes: (B, K, C), (B, K, C)

            # 3) Use the SAME indices to gather from the original 4-channel tensor
            idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, -1, CH)  # (B, K, C, 4)
            local_logits = torch.gather(local_logits, dim=1, index=idx_expanded)  # (B, K, C, 4)

            #local_logits = local_logits.topk(dim=1, k=maxk)[0]
            #global_weights = global_weighter(global_logits,local_logits)
            #local_logits = (local_logits + global_weights) / 2
            
            local_logits = local_logits.permute(1,3,0,2).contiguous() 
            M = local_logits.shape[0]
            N = local_logits.shape[1]
            n_cls = local_logits.shape[3]
            b = local_logits.shape[2]
            ### Local Logits shape token_size, number of text tokens 4, batch_size, number of class
            local_logits = local_logits.view(local_logits.shape[0],local_logits.shape[1], local_logits.shape[2] * local_logits.shape[3])
            local_logits = local_logits.permute(2,0,1)
            
            wdist = 1.0 - local_logits
            xx=torch.zeros(b*n_cls, M, dtype=local_logits.dtype, device=local_logits.device).fill_(1. / M)
            yy=torch.zeros(b*n_cls, N, dtype=local_logits.dtype, device=local_logits.device).fill_(1. / N)
            with torch.no_grad():
                KK = torch.exp(-wdist / self.eps)
                T = self.Sinkhorn(KK,xx,yy)
                #T = self.sinkhorn_unbalanced(KK,xx,yy)
           

            #local_loss = torch.sum(T * local_logits, dim=(1,2)) 
            local_loss = torch.sum(T * local_logits, dim=(1))
       
            local_loss = local_loss.mean(1)
       
            local_loss = local_loss.contiguous().view(b,n_cls)

            local_loss = F.cross_entropy(logit_scale  * local_loss, targets)
        

        if self.use_global_loss:
            # Dropout:
            keep_number = max(global_logits.size(-1) - int(self.global_dropout_p * global_logits.size(-1)), 1)
            index = torch.randint(global_logits.size(-1), (global_logits.size(0), 1, keep_number), device=global_logits.device).expand(-1, global_logits.size(1), -1)
            global_logits = global_logits.gather(-1, index).mean(-1)

            if global_logits.ndim == 2:
                global_loss = F.cross_entropy(logit_scale * global_logits, targets)
            elif global_logits.ndim == 3:
                global_loss = F.cross_entropy(logit_scale * global_logits, targets.unsqueeze(-1).expand(-1, global_logits.size(-1)))
            else:
                raise ValueError(f"Global logits must have 2 or 3 dimensions, but got {global_logits.ndim}.")
        


        return global_loss + local_loss, global_loss, local_loss
