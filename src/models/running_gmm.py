import torch
from torch.distributions import Normal, OneHotCategorical, MixtureSameFamily, Independent


class RunningGMM(torch.nn.Module):
    def __init__(self, features_dim, n_classes, momentum=0.1):
        super(RunningGMM, self).__init__()
        self.features_dim = features_dim
        self.n_classes = n_classes

        self.momentum = momentum

        self.register_buffer('running_means', torch.zeros(self.n_classes, self.features_dim))
        self.register_buffer('running_vars', torch.ones(self.n_classes, self.features_dim))

    def update_running(self, feature_map, targets):
        # current_mean = torch.zeros(self.n_classes, self.features_dim).type_as(feature_map)
        # current_var = torch.ones(self.n_classes, self.features_dim).type_as(feature_map)
        for i, cls in enumerate(sorted(targets.unique())):
            if (targets == cls).int().sum() > 3:

                current_mean = feature_map[targets == cls].mean(0)
                current_var = feature_map[targets == cls].std(0) ** 2

                self.running_means[i] = (1 - self.momentum) * self.running_means[i] + self.momentum * current_mean
                self.running_vars[i] = (1 - self.momentum) * self.running_vars[i] + self.momentum * current_var

    def log_prob(self, feature_map):
        normal_components = Normal(self.running_means, torch.sqrt(self.running_vars))
        categorical_dist = OneHotCategorical(probs=(torch.ones(self.n_classes) / self.n_classes).type_as(feature_map))
        gmm_dist = MixtureSameFamily(categorical_dist._categorical, Independent(normal_components, 1))
        return gmm_dist.log_prob(feature_map).squeeze()



