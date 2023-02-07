from sklearn import preprocessing
from sklearn.linear_model import Ridge
import numpy as np
import torch
import pickle
import os

def regressor(depth_train, true_labels, ROOT, detector_name):
    detector_path = detector_name

    if os.path.exists(detector_path):
        print(f"Load detector {detector_path}")
        detector = pickle.load(open(detector_path, 'rb'))
    else:
        print(f"Train detector {detector_path}")
        os.makedirs(ROOT + "/regressor/", exist_ok=True)
        depth_train = preprocessing.normalize(depth_train)

        detector = Ridge(alpha=0)
        detector.fit(depth_train, true_labels)
        pickle.dump(detector, open(detector_path, 'wb'))

    return detector

def test_hamper_detector(hamper_detector, depth_test, dataset, attack_name, ROOT):
    hamper_scores = ROOT + f"regressor/probs_{dataset}_{attack_name}.npy"

    from sklearn import preprocessing
    depth_test = preprocessing.normalize(depth_test)

    with torch.no_grad():
        scores = hamper_detector.predict(depth_test)
        np.save(arr=scores, file=hamper_scores)

    true_labels_detector = np.concatenate((np.zeros(len(scores) // 2, ), np.ones(len(scores) // 2, )))

    return scores, true_labels_detector