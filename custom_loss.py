import torch
import torch.nn as nn


class KanLossFunction(nn.Module):
    def __init__(self, args):
        super(KanLossFunction, self).__init__()
        self.loss_function = LossFunction(args)

    def forward(self, output, gt_src_masks, gt_tgt_masks):
        n_mask = len(gt_src_masks)

        losses = []
        l1s = []
        l2s = []
        l3s = []

        for mask_id in range(n_mask):
            k = 1.
            if mask_id == (n_mask - 1):
                k = 3.

            _output = {
                "est_src_mask": output["est_src_mask"][mask_id],
                "smoothness_S2T": output["smoothness_S2T"][mask_id],
                "grid_S2T": output["grid_S2T"],
                "est_tgt_mask": output["est_tgt_mask"][mask_id],
                "smoothness_T2S": output["smoothness_T2S"][mask_id],
                "grid_T2S": output["grid_T2S"],
                "flow_S2T": output["flow_S2T"][mask_id],
                "flow_T2S": output["flow_T2S"][mask_id],
                "warped_flow_S2T": output["warped_flow_S2T"][mask_id],
                "warped_flow_T2S": output["warped_flow_T2S"][mask_id],
                "corr_T2S": output["corr_T2S"],
            }

            gt_src_mask = gt_src_masks[mask_id]
            gt_tgt_mask = gt_tgt_masks[mask_id]

            loss, l1, l2, l3 = self.loss_function(_output, gt_src_mask, gt_tgt_mask)
            losses += [loss * k]
            l1s += [l1 * k]
            l2s += [l2 * k]
            l3s += [l3 * k]

        return (
            torch.stack(losses).mean(),
            torch.stack(l1s).mean(),
            torch.stack(l2s).mean(),
            torch.stack(l3s).mean(),
        )


class LossFunction(nn.Module):
    def __init__(self, args):
        super(LossFunction, self).__init__()
        self.loss_fn = nn.MSELoss(reduction="sum")
        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2
        self.lambda3 = args.lambda3

    def lossfn_two_var(self, target1, target2, num_px=None):
        if num_px is None:
            return torch.sum(torch.pow((target1 - target2), 2))
        return torch.sum(torch.pow((target1 - target2), 2) / num_px)

    def forward(self, output, gt_src_mask, gt_tgt_mask):
        eps = 1

        b, _, h, w = gt_src_mask.shape
        src_num_fgnd = gt_src_mask[:, -1:, ...].sum(dim=3, keepdim=True).sum(dim=2, keepdim=True) + eps
        tgt_num_fgnd = gt_tgt_mask[:, -1:, ...].sum(dim=3, keepdim=True).sum(dim=2, keepdim=True) + eps

        # mask consistency
        l1 = self.loss_fn(output["est_src_mask"], gt_src_mask) / (h * w) +\
            self.loss_fn(output["est_tgt_mask"], gt_tgt_mask) / (h * w)
        # flow consistency
        l2 = self.lossfn_two_var(output["flow_S2T"], output["warped_flow_S2T"], src_num_fgnd) +\
            self.lossfn_two_var(output["flow_T2S"], output["warped_flow_T2S"], tgt_num_fgnd)
        # smoothness
        l3 = torch.sum(output["smoothness_S2T"] / src_num_fgnd) +\
            torch.sum(output["smoothness_T2S"] / tgt_num_fgnd)

        return (
            (self.lambda1 * l1 + self.lambda2 * l2 + self.lambda3 * l3) / b,
            l1 * self.lambda1 / b,
            l2 * self.lambda2 / b,
            l3 * self.lambda3 / b,
        )
