import torch
import torch.nn as nn


class GDL(nn.Module):   
    def __init__(self, pNorm=2):
        
        super(GDL, self).__init__()
        self.convX = nn.Conv2d(3, 3, kernel_size=(1, 2), stride=1, padding=(0, 1), bias=False)
        self.convY = nn.Conv2d(3, 3, kernel_size=(2, 1), stride=1, padding=(1, 0), bias=False)
        filterX = torch.Tensor(torch.FloatTensor([[[[-1, 1]], [[-1, 1]], [[-1, 1]]]]))
        filterY = torch.Tensor(torch.FloatTensor([[[[1], [-1]],[[1], [-1]], [[1], [-1]]]]))

        self.convX.weight = torch.nn.Parameter(filterX.cuda(), requires_grad=False)
        self.convY.weight = torch.nn.Parameter(filterY.cuda(), requires_grad=False)
        self.pNorm = pNorm

    def forward(self, pred, gt):
        assert not gt.requires_grad
        assert pred.dim() == 4
        assert gt.dim() == 4
        assert pred.size() == gt.size(), "{0} vs {1} ".format(pred.size(), gt.size())
        
        pred_dx = torch.abs(self.convX(pred))
        pred_dy = torch.abs(self.convY(pred))
        gt_dx = torch.abs(self.convX(gt))
        gt_dy = torch.abs(self.convY(gt))
        
        grad_diff_x = torch.abs(gt_dx - pred_dx)
        grad_diff_y = torch.abs(gt_dy - pred_dy)
        
        
        mat_loss_x = grad_diff_x ** self.pNorm
        
        mat_loss_y = grad_diff_y ** self.pNorm  # Batch x Channel x width x height
        
        shape = gt.shape

        mean_loss = (torch.sum(mat_loss_x) + torch.sum(mat_loss_y)) / (shape[0] * shape[1] * shape[2] * shape[3]) 
               
        return mean_loss
