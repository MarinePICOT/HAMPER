from utils.utils_data import load_adv_data, get_attack_name_to_load
from depth.extract_depth_features import extract_depth_features_by_layers
import numpy as np
import os

def create_dataset_hamper_ba(dataset, X_nat_train_classifier, y_nat_train_classifier, X_train_detector_nat,
                             classifier, num_classes, layers, ROOT, batch_size, num_samples=1000):

    depth_features_train_filename = ROOT + "/regressor/hamper_ba_train_depths.npy"

    if os.path.exists(depth_features_train_filename):
        depth_features_train = np.load(depth_features_train_filename)
    else:
        from setup import ATTACKS_CIFAR10_SVHN, ATTACKS_CIFAR100
        ATTACKS = ATTACKS_CIFAR10_SVHN if dataset != 'cifar100' else ATTACKS_CIFAR100

        depth_features_train_nat = np.zeros((len(ATTACKS), num_samples, len(layers) * num_classes))
        depth_features_train_adv = np.zeros((len(ATTACKS), num_samples, len(layers) * num_classes))

        for i in range(len(ATTACKS)):
            attack_epsilon = ATTACKS[i]
            attack = attack_epsilon.split('_')[0]
            epsilon = attack_epsilon.split('_')[1]
            X_adv, attack_name = load_adv_data(dataset_name=dataset, attack=attack, epsilon=epsilon)

            depth_features_train_, true_labels_train_ = extract_depth_features_by_layers(
                X_nat_train_classifier, y_nat_train_classifier, X_train_detector_nat, X_adv[:num_samples],
                classifier, dataset, num_classes, attack_name, layers, ROOT, batch_size
            )
            depth_features_train_nat[i, :, :] = depth_features_train_[:num_samples, :]
            depth_features_train_adv[i, :, :] = depth_features_train_[num_samples:, :]

        depth_features_train = np.concatenate((depth_features_train_nat, depth_features_train_adv))
        depth_features_train = depth_features_train.reshape((depth_features_train.shape[0] * depth_features_train.shape[1], depth_features_train.shape[2]))
        np.save(arr=depth_features_train, file=depth_features_train_filename)

    true_labels_train = np.concatenate((np.ones(depth_features_train.shape[0] // 2, ), np.zeros(depth_features_train.shape[0] // 2, )))

    return depth_features_train, true_labels_train


def hamper_ba_detector(classifier, X_nat_train_classifier, y_nat_train_classifier, X_nat_test_classifier, y_nat_test_classifier,
                       attack, dataset, device, layers, num_classes, ROOT, epsilon=None, batch_size=500, num_samples=1000):

    print(f"Attack: {get_attack_name_to_load(attack, epsilon)}")
    detector_name = ROOT + f'regressor/hamper_ba.pt'

    # --------------------------
    # Load adversarial examples
    # --------------------------

    from utils.utils_ml import compute_accuracy, compute_logits_return_labels_and_predictions
    X_adv, attack_name = load_adv_data(dataset_name=dataset, attack=attack, epsilon=epsilon)

    _, labels_test_class, predictions_test_adv_class = compute_logits_return_labels_and_predictions(model=classifier, X=X_adv, y=y_nat_test_classifier, device=device)
    print(f"Accuracy on adversarial testing samples: {compute_accuracy(predictions=predictions_test_adv_class, targets=labels_test_class)}")

    # --------------
    # Split the data
    # --------------

    X_train_detector_nat = X_nat_test_classifier
    X_test_detector_nat, X_test_detector_adv = X_nat_test_classifier, X_adv

    # ---------------------------------------------------------------
    # Extract depth features from the data training for the detector
    # ---------------------------------------------------------------

    if not os.path.exists(detector_name):
        depth_features_train, true_labels_train = create_dataset_hamper_ba(
            dataset, X_nat_train_classifier, y_nat_train_classifier, X_train_detector_nat,
            classifier, num_classes, layers, ROOT, batch_size, num_samples=num_samples
        )
    else:
        depth_features_train = None
        true_labels_train = None

    depth_features, true_labels = extract_depth_features_by_layers(
        X_nat_train_classifier, y_nat_train_classifier, X_test_detector_nat, X_test_detector_adv,
        classifier, dataset, num_classes, attack_name, layers, ROOT, batch_size
    )

    depth_features_test = np.concatenate((depth_features[num_samples:depth_features.shape[0] // 2, :],
                                          depth_features[num_samples + (depth_features.shape[0] // 2):, :]))
    true_labels_test = np.concatenate((true_labels[num_samples:true_labels.shape[0] // 2], true_labels[num_samples + (true_labels.shape[0] // 2):]))

    from hamper_detectors.hamper import regressor
    ROOT_ = ROOT + f'/hamper_ba/'
    detector = regressor(depth_train=depth_features_train, true_labels=true_labels_train, ROOT=ROOT_, detector_name=detector_name)

    return detector, depth_features_test, true_labels_test, predictions_test_adv_class[num_samples:], labels_test_class[num_samples:]



