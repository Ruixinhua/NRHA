import torch


class CategoricalLoss(torch.nn.Module):

    def __init__(self):
        super(CategoricalLoss, self).__init__()

    def forward(self, predictions, targets, epsilon=1e-12):
        """
        Computes cross entropy between targets (encoded as one-hot vectors)
        and predictions.
        Input: predictions (N, k) ndarray
               targets (N, k) ndarray
        Returns: scalar
        """
        predictions, targets = predictions.float(), targets.float()
        predictions = torch.clamp(predictions, epsilon, 1. - epsilon)
        return -torch.sum(targets * torch.log(predictions + 1e-9)) / predictions.shape[0]