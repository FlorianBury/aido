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

    def pairwise_cost(self,true,reco):
        cost = torch.zeros((true.shape[0],reco.shape[0]))
        for feature, idxs in self.feature_dict.items():
            t = true[...,idxs]
            y = reco[...,idxs]
            print (t.device,y.device)
            print (self.loss_factors[feature])
            print (type(self.loss_factors[feature]))
            if feature in self.regression:
                cost += self.loss_factors[feature] * (((t[:,None,:]-y[None,:,:])/(t[:,None,:]+1))**2).mean(dim=-1)
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

    def forward(
        self,
        true_part,
        true_mask,
        reco_part,
        reco_mask,
    ):
        losses = torch.zeros(true_part.shape[0]).to(true_part.device)
        for i in range(true_part.shape[0]):
            # Just take real particles (assume ordered as real, the rest being missing) #
            true = true_part[i][true_mask[i]]
            reco = reco_part[i][reco_mask[i]]
            # Compute matrix cost #
            cost_matrix = self.pairwise_cost(true,reco)
            # Get row and cold idx that minimize the cost matrix #
            row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().numpy())
            # Get loss for assignment #
            loss = torch.tensor([0.])
            for feature, idxs in self.feature_dict.items():
                t = true[...,idxs]
                y = reco[...,idxs]
                if feature in self.regression:
                    loss += (((t[row_ind]-y[col_ind])/(t[row_ind]+1))**2).mean()
                    # average per #feature and #particles #
                if feature in self.classification:
                    loss += self.loss_factors[feature] * F.cross_entropy(y[col_ind],t[row_ind])
            # Penalties for missing or fake particles #
            all_true_idx = torch.arange(true.shape[0])
            matched_true = torch.tensor(row_ind)
            missing_idx = torch.tensor(list(set(all_true_idx.tolist()) - set(matched_true.tolist())))
            loss += self.missing_penalty * len(missing_idx)
            all_reco_idx = torch.arange(reco.shape[0])
            matched_reco = torch.tensor(col_ind)
            fake_idx = torch.tensor(list(set(all_reco_idx.tolist()) - set(matched_reco.tolist())))
            loss += self.fake_penalty * len(fake_idx)
            # Record #
            losses[i] = loss
        return losses


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
