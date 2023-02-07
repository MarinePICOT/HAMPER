import torch
from tqdm import tqdm
import torch.nn.functional as trchfnctnl
import numpy as np

def compute_accuracy(predictions: torch.tensor, targets: torch.tensor):
    """
    compute the model's accuracy
    :param predictions: tensor containing the predicted labels
    :param targets: tensor containing the target labels
    :return: the accuracy in [0, 1]
    """
    accuracy = torch.div(torch.sum(predictions == targets), len(targets))
    return accuracy


def compute_logits_return_labels_and_predictions(model, X, device=torch.device("cpu"), batch_size=500, *args, **kwargs):
    """
    compute the logits given input data loader and model
    :param model: model utilized for the logits computation
    :param X: training data
    :param device: device used for computation
    :return: logits and targets
    """
    logits = []
    labels = []
    predictions = []
    if isinstance(X, np.ndarray):
        from utils.utils_data import from_numpy_to_dataloader
        assert 'y' in kwargs, "X is of type numpy.ndarray, pass y in kwargs"
        y = kwargs['y']
        if len(y.shape) > 1 or y.shape[1] > 1:
            y = np.argmax(y, axis=1)
        dataloader = from_numpy_to_dataloader(X, y, batch_size=batch_size)
    else:
        dataloader = X
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(dataloader, ascii=True, ncols=50, colour='red')):
            preds_logit = model(data.to(device))
            logits.append(preds_logit.detach().cpu())

            soft_prob = trchfnctnl.softmax(preds_logit, dim=1)
            if 'print_sp' in kwargs:
                if kwargs['print_sp']:
                    print(soft_prob)
            preds = torch.argmax(soft_prob, dim=1)
            predictions.append(preds.detach().cpu().reshape(-1, 1))

            labels.append(target.detach().cpu().reshape(-1, 1))

    logits = torch.vstack(logits)
    labels = torch.vstack(labels).reshape(-1)
    predictions = torch.vstack(predictions).reshape(-1)
    return logits, labels, predictions
