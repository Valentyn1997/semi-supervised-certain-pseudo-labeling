import torch

from src.models.certainty_strategy import BALDCertainty, PECertainty, Entropy


def test_certainty():
    """ small test utility to check if BALDCertainty and PECertainty can handle entries equal 0. """
    # create test data with shape (T, n, c) where T = iterations, n = samples, c = classes
    data = torch.as_tensor([[[0.05, 0.9, 0.05], [0.8, 0.1, 0.1], [0, 1, 0], [0, 0, 1]],
                            [[0.05, 0.05, 0.9], [0.9, 0.01, 0.09], [0, 1, 0], [0, 0, 1]],
                            [[0.9, 0.05, 0.05], [0.86, 0.04, 0.1], [0, 1, 0], [0, 0, 1]],
                            [[0.05, 0.05, 0.9], [0.99, 0.01, 0], [0, 1, 0], [0, 0, 1]],
                            [[0.98, 0.01, 0.01], [0.89, 0.01, 0.1], [0, 1, 0], [0, 0, 1]]], dtype=torch.float)
    assert all(torch.sum(data, -1).reshape(-1) == 1)  # assert test prediction data adds up to 1

    for strategy in [PECertainty(), BALDCertainty(), Entropy()]:  # strategies using entropy / mutual information
        if strategy.is_ensemble:
            scores, labels = strategy.get_certainty_and_label(softmax_outputs=data)
        else:
            scores, labels = strategy.get_certainty_and_label(softmax_outputs=data[2])
        assert not any(torch.isnan(scores))  # scores should not contain nan
        assert torch.eq(torch.as_tensor([0, 0, 1, 2]), labels).all()  # check labels
        assert scores.shape[0] == 4  # check shape of scores
