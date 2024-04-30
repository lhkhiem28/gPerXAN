import os, sys
from libs import *

class eXplicitOptimizedNorm2d(nn.Module):
    def __init__(self, 
        num_features, 
    ):
        super(eXplicitOptimizedNorm2d, self).__init__()
        self.batchnorm = nn.BatchNorm2d(num_features, affine = True)
        self.score_batchnorm = nn.Parameter(torch.tensor([1.]))
        self.instancenorm = nn.InstanceNorm2d(num_features, affine = True)
        self.score_instancenorm = nn.Parameter(torch.tensor([1.]))

    def forward(self, 
        input, 
    ):
        output_batchnorm = self.batchnorm(input)
        output_instancenorm = self.instancenorm(input)
        scores = F.softmax(
            torch.stack(
                [
                    self.score_batchnorm, 
                    self.score_instancenorm, 
                ], 
                dim = 1, 
            )[0], 
            dim = 0, 
        )
        scores = scores[..., None, None, None, None]

        output = scores*torch.stack(
            [
                output_batchnorm, 
                output_instancenorm, 
            ]
        )
        output = torch.sum(output, dim = 0)

        return output