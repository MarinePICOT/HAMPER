import os

checkpoints_dir = 'checkpoints/classifiers/'
adv_data_dir = 'adv_data/'
results_dir = 'results/'

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# ----------------------- HAMPER CONFIGURATION -----------------------

# CIFAR10 or SVHN
LAYERS_D_RES = ['block-{}-{}-{}'.format(b_layer, n_layer, n_block)
                for b_layer in ['bn_1', 'bn_2', 'conv_1', 'conv_2']
                for n_layer in ['layer4']
                for n_block in [0, 1]]
LAYERS_D_RES += ['layer4', 'convolution_end', 'logits']

ATTACKS_CIFAR10_SVHN = [
    'fgsm_0.03125', 'fgsm_0.0625', 'fgsm_0.125', 'fgsm_0.25', 'fgsm_0.3125', 'fgsm_0.5',
    'bim_0.03125', 'bim_0.0625', 'bim_0.125', 'bim_0.25', 'bim_0.3125', 'bim_0.5',
    'pgdi_0.03125', 'pgdi_0.0625', 'pgdi_0.125', 'pgdi_0.25', 'pgdi_0.3125', 'pgdi_0.5',
    'pgd1_5', 'pgd1_10', 'pgd1_15', 'pgd1_20', 'pgd1_25', 'pgd1_30', 'pgd1_40',
    'pgd2_0.125', 'pgd2_0.25', 'pgd2_0.3125', 'pgd2_0.5', 'pgd2_1', 'pgd2_1.5', 'pgd2_2',
]

# CIFAR100
LAYERS_D_RES_CIFAR100 = ['block-{}-{}-{}'.format(b_layer, n_layer, n_block)
                         for b_layer in ['bn_1', 'bn_2', 'conv_1', 'conv_2']
                         for n_layer in ['layer3']
                         for n_block in [0, 1]]
LAYERS_D_RES_CIFAR100 += ['layer3', 'convolution_end', 'logits']

ATTACKS_CIFAR100 = [
    'fgsm_0.03125', 'fgsm_0.0625','fgsm_0.0625', 'fgsm_0.125', 'fgsm_0.25', 'fgsm_0.3125', 'fgsm_0.5',
    'bim_0.03125', 'bim_0.0625', 'bim_0.125', 'bim_0.25', 'bim_0.3125', 'bim_0.5',
    'pgdi_0.03125', 'pgdi_0.0625', 'pgdi_0.125', 'pgdi_0.25', 'pgdi_0.3125', 'pgdi_0.5',
    'pgd1_40', 'pgd1_500', 'pgd1_1000', 'pgd1_1500', 'pgd1_2000', 'pgd1_2500', 'pgd1_5000',
    'pgd2_5', 'pgd2_10', 'pgd2_15', 'pgd2_20', 'pgd2_30', 'pgd2_40', 'pgd2_50',
    'cwi_', 'df_', 'sa_', 'hop_', 'sta_', 'cw2_'
]


