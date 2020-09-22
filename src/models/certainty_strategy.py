import sys
from abc import ABC
from typing import Tuple

import torch


def entropy(logits: torch.Tensor) -> torch.Tensor:
    epsilon = sys.float_info.min
    return -torch.sum(logits * torch.log(logits + epsilon), dim=-1)


def mutual_information(logits: torch.Tensor) -> torch.Tensor:
    epsilon = sys.float_info.min
    mi = entropy(torch.mean(logits, dim=0)) - torch.mean(torch.sum(-logits * torch.log(logits + epsilon), dim=-1),
                                                         dim=0)
    return mi


class AbstractStrategy(ABC):
    @staticmethod
    def is_ensemble():
        return True

    def get_certainty_and_label(self, logits_t_n_c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This function gets some predictions and returns a certainty value and the predicted target class.
        :param logits_t_n_c: tensor containing the predictions. If strategy is of type ensemble,
        the tensor contains T times the predictions with different outputs. If strategy is not of type ensemble, the
        tensor only contains one value for each sample.
        """
        raise NotImplementedError()


class Entropy(AbstractStrategy):
    @staticmethod
    def is_ensemble():
        return False

    def get_certainty_and_label(self, logits_t_n_c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        max_probs, targets_u = torch.max(logits_t_n_c, dim=-1)
        raw_entropy = entropy(logits_t_n_c)
        certainty = 1 / (1 + raw_entropy)
        return certainty, targets_u


class Margin(AbstractStrategy):
    """
    Classical active learning strategy.
    The larger the difference between first and second highest label the more certain.
    """

    @staticmethod
    def is_ensemble():
        return False

    def get_certainty_and_label(self, logits_t_n_c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        prob_dist, targets_u = torch.sort(logits_t_n_c, dim=-1, descending=True)
        difference = (prob_dist[:, 0] - prob_dist[:, 1])  # difference between top two props
        return difference, targets_u[:, 0]


class SoftMax(AbstractStrategy):
    """ Standard FixMatch metod from the paper. The output of softmax is used as certainty. """

    @staticmethod
    def is_ensemble():
        return False

    def get_certainty_and_label(self, logits_t_n_c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        max_probs, targets_u = torch.max(logits_t_n_c, dim=-1)
        # prob of max label and label index
        return max_probs, targets_u


class MeanSoftmax(AbstractStrategy):
    """ Monte Carlo dropout, mean softmax output is used as certainty. """

    def get_certainty_and_label(self, logits_t_n_c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        max_probs, targets_u = torch.max(torch.mean(logits_t_n_c, dim=0), dim=-1)
        return max_probs, targets_u


class BALDCertainty(AbstractStrategy):
    """
    Based on BALD (Bayesian Active Learning by Disagreement).
    Reminder: BALD measures model confidence, not predictive uncertainty.
    Therefore maybe one threshold is not enough,
    maybe we need a prediction_threshold and a confidence_threshold which both need to be passed.
    """

    def get_certainty_and_label(self, logits_t_n_c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        max_probs, targets_u = torch.max(torch.mean(logits_t_n_c, dim=0), dim=-1)
        mut_inf = mutual_information(logits_t_n_c)
        certainty = 1 / (1 + mut_inf)
        return certainty, targets_u


class PECertainty(AbstractStrategy):
    def get_certainty_and_label(self, logits_t_n_c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = torch.mean(logits_t_n_c, dim=0)
        max_probs, targets_u = torch.max(torch.mean(logits_t_n_c, dim=0), dim=-1)

        mean_entropy = entropy(mean)
        certainty = 1 / (1 + mean_entropy)
        return certainty, targets_u
