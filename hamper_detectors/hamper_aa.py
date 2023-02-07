from utils.utils_data import get_attack_name_to_load
import numpy as np
import os


def hamper_aa_detector(classifier, X_nat_train_classifier, y_nat_train_classifier, X_nat_test_classifier, y_nat_test_classifier,
                       attack, dataset, device, layers, num_classes, ROOT, epsilon=None, batch_size=500, num_samples=1000):
    print(f"Attack: {get_attack_name_to_load(attack, epsilon)}")
    detector_name = ROOT + f'regressor/{get_attack_name_to_load(attack, epsilon)}.pt'

    # --------------------------
    # Load adversarial examples
    # --------------------------

    from utils.utils_data import load_adv_data
    from utils.utils_ml import compute_accuracy, compute_logits_return_labels_and_predictions
    X_adv, attack_name = load_adv_data(dataset_name=dataset, attack=attack, epsilon=epsilon)

    _, labels_test_class, predictions_test_adv_class = compute_logits_return_labels_and_predictions(model=classifier, X=X_adv, y=y_nat_test_classifier, device=device)
    print(f"Accuracy on adversarial testing samples: {compute_accuracy(predictions=predictions_test_adv_class, targets=labels_test_class)}")

    # --------------
    # Split the data
    # --------------
    X_test_detector_nat, X_test_detector_adv = X_nat_test_classifier, X_adv

    # ---------------------------------------------------------------
    # Extract depth features from the data training for the detector
    # ---------------------------------------------------------------
    from depth.extract_depth_features import extract_depth_features_by_layers

    depth_features, true_labels = extract_depth_features_by_layers(
        X_nat_train_classifier, y_nat_train_classifier, X_test_detector_nat, X_test_detector_adv,
        classifier, dataset, num_classes, attack_name, layers, ROOT, batch_size
    )

    depth_features_train = np.concatenate((depth_features[:num_samples, :],
                                           depth_features[depth_features.shape[0] // 2: num_samples+(depth_features.shape[0] // 2), :]))
    true_labels_train = np.concatenate((true_labels[:num_samples],
                                        true_labels[true_labels.shape[0] // 2: num_samples+(true_labels.shape[0] // 2)]))

    depth_features_test = np.concatenate((depth_features[num_samples:depth_features.shape[0] // 2, :],
                                          depth_features[num_samples + (depth_features.shape[0] // 2):, :]))
    true_labels_test = np.concatenate((true_labels[num_samples:true_labels.shape[0] // 2], true_labels[num_samples + (true_labels.shape[0] // 2):]))

    from hamper_detectors.hamper import regressor
    ROOT_ = ROOT + f'/hamper_aa/'
    detector = regressor(depth_train=depth_features_train, true_labels=true_labels_train, ROOT=ROOT_, detector_name=detector_name)

    return detector, depth_features_test, true_labels_test, predictions_test_adv_class[num_samples:], labels_test_class[num_samples:]
