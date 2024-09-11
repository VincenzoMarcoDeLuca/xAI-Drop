from torch.nn import Softmax
import torch

def most_confident(model, x, edge_index, percentage, threshold_confidence_value = None, confidence_strategy = 'ranking', need_softmax=True, task='node_classification', criterion = 'Most', mask = None):
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
        ###HERE I GET THE MOST CERTAIN NODE
        softed_y = softmax(y)
        most_confident_class, _ = torch.max(softed_y, axis=-1)
    else:
        most_confident_class = y

    mean_confidence = torch.mean(most_confident_class)
    std_confidence = torch.std(most_confident_class)
    if confidence_strategy == 'ranking':
        ranked_y = torch.argsort(most_confident_class)
        most_confident_samples = ranked_y[-int(y.shape[0]*percentage):]
        confidence_values = most_confident_class[most_confident_samples]
    elif confidence_strategy == 'threshold':
        if criterion == 'Most':
            higher_than_threshold = most_confident_class > threshold_confidence_value
        elif criterion == 'Least':
            higher_than_threshold = most_confident_class < threshold_confidence_value
        else:
            print("Criterion for sampling nodes: ", criterion, " based on confidence actually not available")
        most_confident_samples = torch.nonzero(higher_than_threshold)
        if most_confident_samples.shape[0] == 0:

            return None, None, mean_confidence, std_confidence
        confidence_values = most_confident_class[most_confident_samples]
    else:
        print("Confidence strategy: ", confidence_strategy)
        exit(1)
    return most_confident_samples, confidence_values, mean_confidence.item(), std_confidence.item()

