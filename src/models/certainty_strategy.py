from abc import ABC
from typing import Tuple

import torch


def entropy_log(values: torch.Tensor) -> torch.Tensor:
    """
    Calculate log of pytorch tensor used for entropy calculation.
    0 results in 0 as they add zero information to entropy.
    :param values: input values.
    :return: prepared log of values.
    """
    _log = torch.log(values)
    mask = torch.isinf(_log)
    return _log.masked_fill(mask, 0)


def entropy(values: torch.Tensor) -> torch.Tensor:
    return -torch.sum(values * entropy_log(values), dim=-1)


def mutual_information(values: torch.Tensor) -> torch.Tensor:
    mi = entropy(values.mean(0)) - torch.mean(torch.sum(-values * entropy_log(values), -1), 0)
    return mi


class AbstractStrategy(ABC):
    is_ensemble = True

    def get_certainty_and_label(self, softmax_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This function gets some predictions and returns a certainty value and the predicted target class.
        :param softmax_outputs: tensor containing the predictions (softmax outputs). If strategy is of type ensemble,
        the tensor contains T times the predictions with different outputs. If strategy is not of type ensemble, the
        tensor only contains one value for each sample.
        """
        raise NotImplementedError()


class Entropy(AbstractStrategy):
    is_ensemble = False

    def get_certainty_and_label(self, softmax_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        max_probs, targets_u = torch.max(softmax_outputs, dim=-1)
        raw_entropy = entropy(softmax_outputs)
        certainty = 1 / (1 + raw_entropy)
        return certainty, targets_u


class Margin(AbstractStrategy):
    """
    Classical active learning strategy.
    The larger the difference between first and second highest label the more certain.
    """
    is_ensemble = False

    def get_certainty_and_label(self, softmax_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        prob_dist, targets_u = torch.sort(softmax_outputs, dim=-1, descending=True)
        difference = (prob_dist[:, 0] - prob_dist[:, 1])  # difference between top two props
        return difference, targets_u[:, 0]


class SoftMax(AbstractStrategy):
    """ Standard FixMatch metod from the paper. The output of softmax is used as certainty. """
    is_ensemble = False

    def get_certainty_and_label(self, softmax_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        max_probs, targets_u = torch.max(softmax_outputs, dim=-1)
        # prob of max label and label index
        return max_probs, targets_u


class MeanSoftmax(AbstractStrategy):
    """ Monte Carlo dropout, mean softmax output is used as certainty. """

    def get_certainty_and_label(self, softmax_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        max_probs, targets_u = torch.max(torch.mean(softmax_outputs, dim=0), dim=-1)
        return max_probs, targets_u


class BALDCertainty(AbstractStrategy):
    """
    Based on BALD (Bayesian Active Learning by Disagreement).
    Reminder: BALD measures model confidence, not predictive uncertainty.
    Therefore maybe one threshold is not enough,
    maybe we need a prediction_threshold and a confidence_threshold which both need to be passed.
    """

    def get_certainty_and_label(self, softmax_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        max_probs, targets_u = torch.max(torch.mean(softmax_outputs, dim=0), dim=-1)
        return max_probs, targets_u

    def get_model_confidence(self, softmax_outputs: torch.Tensor) -> torch.Tensor:
        mut_inf = mutual_information(softmax_outputs)
        confidence = 1 / (1 + mut_inf)
        return confidence


class PECertainty(AbstractStrategy):
    def get_certainty_and_label(self, softmax_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = torch.mean(softmax_outputs, dim=0)
        max_probs, targets_u = torch.max(mean, dim=-1)

        mean_entropy = entropy(mean)
        certainty = 1 / (1 + mean_entropy)
        return certainty, targets_u


class MultiStrategies(AbstractStrategy):
    def __init__(self):
        self.strategies = [Entropy(), Margin(), SoftMax(), MeanSoftmax(), BALDCertainty(), PECertainty()]
        self.strategies = {k: v for (k, v) in zip(map(lambda s: s.__class__.__name__, self.strategies), self.strategies)}

    def get_certainty_and_label(self, softmax_outputs: torch.Tensor, strategy_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.strategies[strategy_name].get_certainty_and_label(softmax_outputs)
