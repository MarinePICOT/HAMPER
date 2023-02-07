import os
import logging
import numpy as np
from tqdm import tqdm
from depth.data_depth import sampled_sphere, DataDepth


def load_depth_layer_if_exists(layer_name, c, dataset, attack_name, ROOT):
    file_depth = ROOT + f"{layer_name}/probs_{dataset}_{attack_name}_{c}.npy"

    if not os.path.exists(file_depth):
        return None, file_depth

    return np.load(file_depth), file_depth

def merge_layers_depths(num_classes, num_samples, layers_names, dataset, attack, ROOT):

    depths = np.ones((num_samples, len(layers_names), num_classes))

    for i in range(len(layers_names)):
        layer = layers_names[i]
        for c in range(num_classes):
            depth, _ = load_depth_layer_if_exists(layer, c, dataset, attack, ROOT)
            depths[:, i, c] = depth

    return depths

def depth_by_class(depth, X_train, X_test, y_train, c, layer, U=None):
    X_train_c = X_train[np.where(np.argmax(y_train, axis=1) == c)[0]]
    res = depth.halfspace_mass(X=X_train_c, X_test=X_test, U=U, layer=layer, num_class=c)
    return res

def depth_from_dict(model, dict_train, y_train, dict_test, K, layers, num_classes, dataset):
    from collections import defaultdict
    features = defaultdict(list)
    depth = DataDepth(K, dataset)
    depth_res = {}
    
    for i in range(len(layers)):
        layer = layers[i]
        depth_res[layer] = []
        print('Layer', layer)
        for c in tqdm(range(num_classes), ascii=True, ncols=70, colour='yellow'):
            X_test = dict_test[layer]
            logging.info(X_test.shape)
            X_train = dict_train[layer]
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_test = X_test.reshape(X_test.shape[0], -1)

            _, dim = X_train.shape
            path = 'results/depth/U_{}.npy'.format(dim)
            if os.path.exists(path):
                U = np.load(path)
            else:
                from pathlib import Path
                Path('results/depth/').mkdir(parents=True, exist_ok=True)
                U = sampled_sphere(K, dim)
                np.save(path, U)
            res = depth_by_class(depth=depth, X_train=X_train, X_test=X_test, y_train=y_train, c=c, U=U, layer=layer)
            depth_res[layer].append(res)
        logging.info(layer)
    return depth_res


