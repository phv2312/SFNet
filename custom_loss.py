import torch
import torch.nn as nn

class loss_functions(nn.Module):
    def __init__(self, args):
        super(loss_functions, self).__init__()
        self.loss_function = loss_function(args)

    def forward(self, output, GT_src_masks, GT_tgt_masks):
        n_mask = len(GT_src_masks)

        losses = []
        l1s = []
        l2s = []
        l3s = []
        for mask_id in range(n_mask):
            k = 1.
            if mask_id == (n_mask - 1):
                k = 3.

            _output = {
                'est_src_mask': output['est_src_mask'][mask_id],
                'smoothness_S2T': output['smoothness_S2T'][mask_id],
                'grid_S2T': output['grid_S2T'],
                'est_tgt_mask': output['est_tgt_mask'][mask_id],
                'smoothness_T2S': output['smoothness_T2S'][mask_id],
                'grid_T2S': output['grid_T2S'],
                'flow_S2T': output['flow_S2T'][mask_id],
                'flow_T2S': output['flow_T2S'][mask_id],
                'warped_flow_S2T': output['warped_flow_S2T'][mask_id],
                'warped_flow_T2S': output['warped_flow_T2S'][mask_id],
                'corr_T2S': output['corr_T2S'],
            }

            GT_src_mask = GT_src_masks[mask_id]
            GT_tgt_mask = GT_tgt_masks[mask_id]

            loss, l1, l2, l3 = self.loss_function(_output, GT_src_mask, GT_tgt_mask)
            losses += [loss * k]
            l1s += [l1 * k]
            l2s += [l2 * k]
            l3s += [l3 * k]

        return torch.stack(losses).mean(),torch.stack(l1s).mean(), torch.stack(l2s).mean(), torch.stack(l3s).mean()

class loss_function(nn.Module):
    def __init__(self, args):
        super(loss_function, self).__init__()
        self.lossfn = nn.MSELoss(reduction='sum')
        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2
        self.lambda3 = args.lambda3
    
    def lossfn_two_var(self, target1, target2, num_px = None):
        if num_px is None:
            return torch.sum(torch.pow((target1 - target2),2))
        else:
            return torch.sum(torch.pow((target1 - target2),2) / num_px)

    def forward(self, output, GT_src_mask, GT_tgt_mask):
        eps = 1

        _GT_src_mask = GT_src_mask
        _GT_tgt_mask = GT_tgt_mask

        b, _, h, w = _GT_src_mask.size()
        src_num_fgnd = _GT_src_mask.sum(dim=3, keepdim=True).sum(dim=2, keepdim=True) + eps
        tgt_num_fgnd = _GT_tgt_mask.sum(dim=3, keepdim=True).sum(dim=2, keepdim=True) + eps

        L1 = self.lossfn(output['est_src_mask'], _GT_src_mask) / (h * w) + self.lossfn(output['est_tgt_mask'],
                                                                                       _GT_tgt_mask) / (
                         h * w)  # mask consistency
        L2 = self.lossfn_two_var(output['flow_S2T'], output['warped_flow_S2T'], src_num_fgnd) \
             + self.lossfn_two_var(output['flow_T2S'], output['warped_flow_T2S'], tgt_num_fgnd)  # flow consistency
        L3 = torch.sum(output['smoothness_S2T'] / src_num_fgnd) + torch.sum(
            output['smoothness_T2S'] / tgt_num_fgnd)  # smoothness

        return (self.lambda1 * L1 + self.lambda2 * L2 + self.lambda3 * L3) / _GT_src_mask.size(0), \
               L1 * self.lambda1 / _GT_src_mask.size(0), \
               L2 * self.lambda2 / _GT_src_mask.size(0), \
               L3 * self.lambda3 / _GT_src_mask.size(0)
