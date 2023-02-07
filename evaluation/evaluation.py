import numpy as np
import sklearn.metrics as skm


def collect_decision_by_thresholds(probs, thrs_size):
    thrs = np.linspace(probs.min(), probs.max(), thrs_size)
    decision_by_thr = np.zeros((len(probs), thrs_size))

    # An example is detected as adversarial is the prediction is above the threshold, natural if not
    for i in range(thrs_size):
        thr = thrs[i]

        y_pred = np.where(probs <= thr, 1, 0)

        decision_by_thr[:, i] = y_pred

    return decision_by_thr, thrs

def get_fp_tp_tn_fn(decision_by_thr_adv, decision_by_thr_nat, correct_all, labels_tot):
    # The sample is considered as true positive iff there is at least one successful adversarial examples (i.e. correct_all > 0) and iff the detector detects all of successful adversarial examples (i.e. decision_by_thr_adv = correct_all).
    tp = np.where((decision_by_thr_adv == correct_all) & (correct_all > 0), 1, 0)
    # The sample is considered as false positive if it was a natural example (i.e. labels_tot==0) and if it detected as adversarial (the decision is above 0).
    fp = np.where((decision_by_thr_adv > 0) & (labels_tot == 0), 1, 0)
    # The sample is considered as false positive if it was a natural example (i.e. labels_tot==0) and if it detected as natural (the natural decision is 0).
    tn = np.where((decision_by_thr_nat == 0) & (labels_tot == 0), 1, 0)
    # The sample is considered as false negative iff here is at least one successful adversarial examples (i.e. correct_all > 0) and if it detected less examples than there is (decision_by_thr_adv < correct_all).
    fn = np.where((decision_by_thr_adv < correct_all) & (correct_all > 0), 1, 0)
    return tp, fp, tn, fn


def evaluate(y_preds_adv, y_test, proba, labels, thrs_size=200):

    # Compute whether the adversarial examples successfully fools the target classifier or not, and save the decision
    successful_attacks = [np.where(y_preds_adv != y_test, 1, 0)]

    # We reshape the variable that if the noisy sample is successful or not
    ca = np.ones((len(proba), thrs_size))
    for j in range(thrs_size):
        ca[len(proba) // 2:, j] = np.asarray(successful_attacks)

    # We multiply the adversarial decision by the successfulness of the attack to discard the non-adversarial samples.
    decision, thrs = collect_decision_by_thresholds(proba, thrs_size)
    decision_by_thr_adv = decision * ca
    decision_by_thr_nat = decision

    successful_attacks = np.transpose(successful_attacks)

    # We gather the true label (i.e. 0 if the sample is natural, 1 if it is not).
    labels_tot = (np.ones((thrs_size, len(labels))) * labels).transpose()

    successful_attacks = np.concatenate((np.zeros(successful_attacks.shape), successful_attacks), axis=0)
    correct_all = np.zeros(decision_by_thr_adv.shape)

    # We compute the number of times a natural sample has a successful adversarial examples.
    for i in range(decision_by_thr_adv.shape[1]):
        correct_all[:, i] = successful_attacks.sum(axis=1)

    tp, fp, tn, fn = get_fp_tp_tn_fn(decision_by_thr_adv, decision_by_thr_nat, correct_all, labels_tot)

    # We sum over all the examples.
    tpr = tp.sum(axis=0) / (tp.sum(axis=0) + fn.sum(axis=0))
    fpr = fp.sum(axis=0) / (fp.sum(axis=0) + tn.sum(axis=0))

    x = np.interp(0.95, np.sort(tpr), np.sort(fpr))
    auc, fpr_at_95_tpr = round(skm.auc(np.sort(fpr), np.sort(tpr)) * 100, 1), round(x * 100, 1)

    return auc, fpr_at_95_tpr
