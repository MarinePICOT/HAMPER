from models.detector import Extract_Depth, extraction_resnet
from utils.utils_depth import load_depth_layer_if_exists, merge_layers_depths
from tqdm import tqdm
import numpy as np
import pickle
import os

def check_depth_files(layers, dataset, num_classes, attack, ROOT):
    missing_layers = {}
    load_layers = []

    for layer in tqdm(layers, ascii=True, ncols=70, desc="Checking depth files..."):
        classes = []
        for c in range(num_classes):
            _, file_depth = load_depth_layer_if_exists(layer, c, dataset, attack, ROOT)

            if not os.path.exists(file_depth):
                classes.append(c)
            else:
                load_layers.append(file_depth)
        missing_layers[layer] = classes
    return missing_layers, load_layers

def extract_depth_features_by_layers(X_nat_train_classifier, y_nat_train_classifier, X_train_detector_nat, X_train_detector_adv,
                                     classifier, dataset, num_classes, attack_name, layers, ROOT, batch_size=500):
    # ---------------------------------------------------------------
    # Extract depth features from the data training for the detector
    # ---------------------------------------------------------------

    # Check if the depth features have been already extracted
    layers, load_layers = check_depth_files(layers=layers, dataset=dataset, num_classes=num_classes, attack=attack_name, ROOT=ROOT)

    if np.sum([len(v) for _, v in layers.items()]) > 0:
        print("!!! Extract depth features !!!")

        print(f"Missing layers-classes: {layers}")

        dict_train_file = ROOT + f'regressor/dict_train_{dataset}.pkl'
        if os.path.exists(dict_train_file):
            print(f"dict_train_depth train already exists {dict_train_file}")
            dict_train_depth = pickle.load(open(dict_train_file, 'rb'))
        else:
            dict_train_depth = extraction_resnet(X_nat_train_classifier, classifier, bs=batch_size, dataset=dataset)
            dict_train_depth = dict((k, dict_train_depth[k]) for k in layers if k in dict_train_depth)
            os.makedirs(ROOT + f'regressor', exist_ok=True)
            pickle.dump(dict_train_depth, open(dict_train_file, 'wb'))

        missing_layers = dict((k, layers[k]) for k in layers if len(layers[k]) > 0)
        depth_extractor = Extract_Depth(model=classifier,
                                        dataset=dataset,
                                        layers_dict_train=dict_train_depth,
                                        y_train=y_nat_train_classifier,
                                        layers=np.sort(list(missing_layers.keys())).tolist(),
                                        num_classes=num_classes, bs=batch_size)
        print('NATURAL')
        depth_features_train_nat_detector = depth_extractor(X_train_detector_nat)

        print('ADVERSARIAL')
        depth_features_train_adv_detector = depth_extractor(X_train_detector_adv)

        for layer, classes in layers.items():
            for c in tqdm(classes, ascii=True):
                depth_layer_c_nat = depth_features_train_nat_detector[layer][c]
                depth_layer_c_adv = depth_features_train_adv_detector[layer][c]
                depth_layer_c = np.concatenate((depth_layer_c_nat, depth_layer_c_adv), axis=0)

                _, file_depth = load_depth_layer_if_exists(layer, c, dataset, attack_name, ROOT)
                os.makedirs(ROOT + f"{layer}/", exist_ok=True)
                np.save(arr=depth_layer_c, file=file_depth)
                load_layers.append(file_depth)

    depth_features = merge_layers_depths(num_classes=num_classes,
                                         num_samples=len(X_train_detector_nat) + len(X_train_detector_adv),
                                         layers_names=np.sort(list(layers.keys())).tolist(),
                                         dataset=dataset,
                                         attack=attack_name,
                                         ROOT=ROOT)

    depth_features = depth_features.reshape((depth_features.shape[0], depth_features.shape[1] * depth_features.shape[2]))
    true_labels = np.concatenate((np.ones(len(X_train_detector_nat), ), np.zeros(len(X_train_detector_adv), )))  # 1 is the depth score for natural samples, 0 for adversarial

    return depth_features, true_labels