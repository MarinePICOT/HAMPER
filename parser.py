import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-d', '--dataset_name',
        help="Dataset to use; either 'cifar10', 'svhn' or 'cifar100'",
        type=str, default='cifar10'
    )

    parser.add_argument(
        '-a', '--attack',
        help="Attack strategy to test",
        type=str, default='pgdi'
    )

    parser.add_argument(
        '-e', '--epsilon',
        help="Perturbation magnitude for the attack to test",
        type=str, default=0.03125
    )

    parser.add_argument(
        '-ht', '--hamper_type',
        help="Hamper version to use; either 'aa' (attack-aware) or 'ba' (blind-to-attack)",
        type=str, default='aa'
    )

    parser.add_argument(
        '-dv', '--device',
        help="GPU/CPU",
        type=int, default=0
    )

    parser.add_argument(
        '-bs', '--batch_size',
        help="Batch size",
        default=500, type=int
    )

    args = parser.parse_args()
    return args



