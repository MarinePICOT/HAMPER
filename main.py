from setup import results_dir, checkpoints_dir, LAYERS_D_RES, LAYERS_D_RES_CIFAR100
from utils.utils_data import load_data, get_attack_name_to_load
from utils.utils_general import assign_to_device
from utils.utils_models import load_model
from utils.utils_ml import compute_accuracy, compute_logits_return_labels_and_predictions
import numpy as np

ROOT = results_dir


def main(args):
    attack = args.attack
    batch_size = args.batch_size
    dataset = args.dataset_name
    device = assign_to_device(args.device)
    epsilon = args.epsilon
    hamper = args.hamper_type
    checkpoints_dir_ = '{}/{}/'.format(checkpoints_dir, dataset)

    assert dataset in ['cifar10', 'svhn', 'cifar100'], "'cifar10', 'svhn', 'cifar100'"

    # --------------------
    # Load the classifier
    # --------------------
    classifier = load_model(dataset, checkpoints_dir_, device)
    classifier.eval()

    # --------------------
    # Load the dataset
    # --------------------
    x_train_, y_train_, x_test_, y_test_, num_classes = load_data(dataset)
    LAYERS = LAYERS_D_RES if dataset != 'cifar100' else LAYERS_D_RES_CIFAR100
    LAYERS = np.sort(LAYERS).tolist()

    _, labels_train, predictions_train = compute_logits_return_labels_and_predictions(model=classifier, X=x_train_, y=y_train_, device=device)
    _, labels_test, predictions_test = compute_logits_return_labels_and_predictions(model=classifier, X=x_test_, y=y_test_, device=device)

    print(f"Dataset: {dataset}")
    print(f"Accuracy on natural training samples: {compute_accuracy(predictions=predictions_train, targets=labels_train)}")
    print(f"Accuracy on natural testing samples: {compute_accuracy(predictions=predictions_test, targets=labels_test)}")

    ROOT_ = ROOT + f'hamper/{dataset}/'

    if hamper == 'aa':
        # ---------------------------------
        # HAMPER_AA (Attack-aware)
        # ---------------------------------

        from hamper_detectors.hamper_aa import hamper_aa_detector
        hamper_detector, depth_features_test, true_labels_test, predictions_test_adv_class, labels_test_class = \
            hamper_aa_detector(
                classifier=classifier,
                X_nat_train_classifier=x_train_, y_nat_train_classifier=y_train_,
                X_nat_test_classifier=x_test_, y_nat_test_classifier=y_test_,
                attack=attack, epsilon=epsilon, device=device, dataset=dataset,
                batch_size=batch_size, layers=LAYERS, num_classes=num_classes,
                ROOT=ROOT_
            )
    else:
        # ---------------------------------
        # HAMPER_BA (Blind-to-attack)
        # ---------------------------------

        from hamper_detectors.hamper_ba import hamper_ba_detector
        hamper_detector, depth_features_test, true_labels_test, predictions_test_adv_class, labels_test_class = \
            hamper_ba_detector(
                classifier=classifier,
                X_nat_train_classifier=x_train_, y_nat_train_classifier=y_train_,
                X_nat_test_classifier=x_test_, y_nat_test_classifier=y_test_,
                attack=attack, epsilon=epsilon, device=device, dataset=dataset,
                batch_size=batch_size, layers=LAYERS, num_classes=num_classes,
                ROOT=ROOT_
            )

    from hamper_detectors.hamper import test_hamper_detector
    hamper_scores, true_labels_detector = test_hamper_detector(hamper_detector, depth_features_test, dataset, get_attack_name_to_load(attack, epsilon), ROOT_)

    from evaluation.evaluation import evaluate

    auc, fpr_at_95_tpr = evaluate(y_preds_adv=predictions_test_adv_class,
                                  y_test=labels_test_class, proba=hamper_scores, labels=true_labels_detector)

    from tabulate import tabulate
    TABLE = [
        [f"Hamper {'Attack-Aware' if hamper == 'aa' else 'Blind-to-Attack'} Average Score over natural samples", hamper_scores[:len(hamper_scores) // 2].mean()],
        [f"Hamper {'Attack-Aware' if hamper == 'aa' else 'Blind-to-Attack'} Average Score over adversarial samples", hamper_scores[-len(hamper_scores) // 2:].mean()],
        ["AUROC", auc],
        ["FPR at 95% of TPR", fpr_at_95_tpr]
    ]

    print(tabulate(TABLE))

    return hamper_scores, auc, fpr_at_95_tpr


if __name__ == "__main__":
    from parser import parse_args

    args = parse_args()

    main(args)
