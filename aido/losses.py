import torch
from torch import nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

class HungarianMatching(nn.Module):
    def __init__(
        self,
        feature_dict,
        loss_factors,
        fake_penalty,
        missing_penalty,
        classification,
        regression,
    ):
        super().__init__()

        self.feature_dict = feature_dict
        self.loss_factors = loss_factors
        self.fake_penalty = fake_penalty
        self.missing_penalty = missing_penalty
        self.classification = classification
        self.regression = regression
        assert set(self.loss_factors.keys()) == set(self.classification+self.regression), f'Mismatch between features {list(self.loss_factors.keys())} and classification ({self.classification}) + regression ({self.regression})'

    @torch.no_grad()
    def pairwise_cost(self,true,reco):
        cost = torch.zeros((true.shape[0],reco.shape[0])).to(true.device)
        for feature, idxs in self.feature_dict.items():
            t = true[...,idxs]
            y = reco[...,idxs]
            if feature in self.regression:
                cost += self.loss_factors[feature] * ((t[:,None,:]-y[None,:,:])**2).sum(dim=-1).sqrt()
            elif feature in self.classification:
                for i in range(t.shape[0]):
                    cost[i] = self.loss_factors[feature] * F.cross_entropy(
                        y,
                        t[i].argmax().repeat(y.shape[0]),
                        reduction = 'none',
                    )
            else:
                raise RuntimeError
        return cost

    def get_losses(
        self,
        true,
        reco,
        row_ind,
        col_ind,
    ):
        losses = {'total' : torch.tensor([0.]).to(true.device)}
        # Get loss for assignment #
        if len(row_ind) > 0:
            for feature, idxs in self.feature_dict.items():
                t = true[...,idxs]
                y = reco[...,idxs]
                if feature in self.regression:
                    loss = ((t[row_ind]-y[col_ind])**2).sum(dim=-1).sqrt().mean()
                if feature in self.classification:
                    loss = F.cross_entropy(y[col_ind],t[row_ind])
                losses[feature] = loss
                losses['total'] = losses['total'] + self.loss_factors[feature] * loss
        else:
            for feature in self.feature_dict.keys():
                losses[feature] = torch.tensor(0.)
        # Penalties for missing particles #
        all_true_idx = torch.arange(true.shape[0])
        matched_true = torch.tensor(row_ind)
        missing_idx = torch.tensor(list(set(all_true_idx.tolist()) - set(matched_true.tolist())))
        losses['missing'] = torch.tensor(len(missing_idx))
        losses['total'] = losses['total'] + self.missing_penalty * len(missing_idx)
        # Penalties for fake particles #
        all_reco_idx = torch.arange(reco.shape[0])
        matched_reco = torch.tensor(col_ind)
        fake_idx = torch.tensor(list(set(all_reco_idx.tolist()) - set(matched_reco.tolist())))
        losses['fake'] = torch.tensor(len(fake_idx))
        losses['total'] = losses['total'] + self.fake_penalty * len(fake_idx)
        # Return #
        return losses

    def forward(
        self,
        true_part,
        true_mask,
        reco_part,
        reco_mask,
    ):
        all_losses = {key:[] for key in list(self.feature_dict.keys())+['fake','missing','total']}
        for i in range(true_part.shape[0]):
            # Just take real particles (assume ordered as real, the rest being missing) #
            true = true_part[i][true_mask[i]].clone()
            reco = reco_part[i][reco_mask[i]].clone()
            # Compute matrix cost #
            cost_matrix = self.pairwise_cost(true,reco)
            # Get row and cold idx that minimize the cost matrix #
            row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().numpy())
            # Get losses #
            losses = self.get_losses(true,reco,row_ind,col_ind)
            # Record #
            for key,val in losses.items():
                assert key in all_losses.keys()
                all_losses[key].append(val)
        return {
            key : torch.stack(values)
            for key,values in all_losses.items()
        }


if __name__ == '__main__':
    import sys
    from aido.config import AIDOConfig
    from aido.surrogate import SurrogateDataset

    config = AIDOConfig()
    dataset = SurrogateDataset(config.surrogate)
    dataset.load(sys.argv[1])

    loss = HungarianMatching(
        feature_dict = dataset.feature_dict,
        loss_factors = config.loss.loss_factors,
        fake_penalty = config.loss.fake_penalty,
        missing_penalty = config.loss.missing_penalty,
        classification = config.surrogate.classification,
        regression = config.surrogate.regression,
    )

    losses = loss(
        dataset.true_part,
        dataset.true_mask,
        dataset.reco_part,
        dataset.reco_mask,
    )

    from IPython import embed; embed()
