from torch.nn import Softmax
import torch

def most_confident(model, x, edge_index, threshold_confidence_value = None, need_softmax=True, task='node_classification'):
    if task=='node_classification':
        with torch.no_grad():
            model.eval()
            y = model(x, edge_index)
    elif task == 'link_prediction':
        with torch.no_grad():
            model.eval()
            z = model.encode(x, edge_index)
            y = model.decode(z, edge_index).view(-1).sigmoid()
    else:
         print("Task ", task, " not known")
         exit(1)
    if need_softmax:
        softmax = Softmax(dim=-1)
        softed_y = softmax(y)
        most_confident_class, _ = torch.max(softed_y, axis=-1)
    else:
        most_confident_class = y

    higher_than_threshold = most_confident_class > threshold_confidence_value
    most_confident_samples = torch.nonzero(higher_than_threshold)
    if most_confident_samples.shape[0] == 0:

            return None, None
    confidence_values = most_confident_class[most_confident_samples]

    return most_confident_samples, confidence_values
