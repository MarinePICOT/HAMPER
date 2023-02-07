import torch
import numpy as np
from tqdm import tqdm

def load_model(dataset_name, checkpoints_dir, device):
    if dataset_name in ['cifar10', 'svhn']:
        from models.resnet18 import ResNet18
        path = '{}rn-best.pt'.format(checkpoints_dir)
        model = ResNet18(num_classes=10)
    elif dataset_name == 'cifar100':
        from models.resnet101 import ResNet, Bottleneck
        from collections import OrderedDict
        path = '{}/resnet/model_best.pth.tar'.format(checkpoints_dir)
        model = ResNet(Bottleneck, [18, 18, 18], num_classes=100)
        checkpoint = torch.load(path)
        state_dict_ = checkpoint["state_dict"]
        state_dict = OrderedDict()
        for k, v in state_dict_.items():
            name = k[7:]
            state_dict[name] = v

    if torch.cuda.is_available():
        if dataset_name != 'cifar100':
            state_dict = torch.load(path)
        model.load_state_dict(state_dict)
        model = model.to(device)
    else:
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)

    return model

def extraction_resnet(loader, model, device="cuda", bs=500, dataset='cifar100'):
    from torch.nn import functional as F
    import math

    with torch.no_grad():
        num_batches = math.ceil(loader.shape[0] / bs)
        for i in tqdm(range(num_batches), ascii=True):
            if i == 0:
                data = torch.tensor(loader[i * bs: (i + 1) * bs]).to(device)
                all_hidden_states = {
                }
                hidden_states = data
                # Conv1
                hidden_states = model.conv1(hidden_states)
                # Bn1
                hidden_states = model.bn1(hidden_states)
                hidden_states = F.relu(hidden_states)

                # ----------------------------------------------------------------------------------------
                # --------- Layer1
                # -------- Block0
                # ------- Conv1
                inside_hidden_states = model.layer1[0].conv1(hidden_states)
                all_hidden_states['block-conv_1-layer1-0'] = inside_hidden_states.detach().cpu().numpy()
                # ------- Bn1
                inside_hidden_states = F.relu(model.layer1[0].bn1(inside_hidden_states))
                all_hidden_states['block-bn_1-layer1-0'] = inside_hidden_states.detach().cpu().numpy()
                # ------- Conv2
                inside_hidden_states = model.layer1[0].conv2(inside_hidden_states)
                all_hidden_states['block-conv_2-layer1-0'] = inside_hidden_states.detach().cpu().numpy()
                # ------- Bn2
                inside_hidden_states = model.layer1[0].bn2(inside_hidden_states)
                all_hidden_states['block-bn_2-layer1-0'] = F.relu(inside_hidden_states).detach().cpu().numpy()
                out_block0_hidden_states = model.layer1[0](hidden_states)

                # -------- Block1
                # ------- Conv1
                inside_hidden_states = model.layer1[1].conv1(out_block0_hidden_states)
                all_hidden_states['block-conv_1-layer1-1'] = inside_hidden_states.detach().cpu().numpy()
                # ------- Bn1
                inside_hidden_states = F.relu(model.layer1[1].bn1(inside_hidden_states))
                all_hidden_states['block-bn_1-layer1-1'] = inside_hidden_states.detach().cpu().numpy()
                # ------- Conv2
                inside_hidden_states = model.layer1[1].conv2(inside_hidden_states)
                all_hidden_states['block-conv_2-layer1-1'] = inside_hidden_states.detach().cpu().numpy()
                # ------- Bn2
                inside_hidden_states = model.layer1[1].bn2(inside_hidden_states)
                all_hidden_states['block-bn_2-layer1-1'] = F.relu(inside_hidden_states).detach().cpu().numpy()
                out_block1_hidden_states = model.layer1[1](out_block0_hidden_states)

                hidden_states = model.layer1(hidden_states)
                all_hidden_states['layer1'] = hidden_states.detach().cpu().numpy()

                # ----------------------------------------------------------------------------------------

                # --------- Layer2
                # -------- Block0
                # ------- Conv1
                inside_hidden_states = model.layer2[0].conv1(hidden_states)
                all_hidden_states['block-conv_1-layer2-0'] = inside_hidden_states.detach().cpu().numpy()
                # ------- Bn1
                inside_hidden_states = F.relu(model.layer2[0].bn1(inside_hidden_states))
                all_hidden_states['block-bn_1-layer2-0'] = inside_hidden_states.detach().cpu().numpy()
                # ------- Conv2
                inside_hidden_states = model.layer2[0].conv2(inside_hidden_states)
                all_hidden_states['block-conv_2-layer2-0'] = inside_hidden_states.detach().cpu().numpy()
                # ------- Bn2
                inside_hidden_states = model.layer2[0].bn2(inside_hidden_states)
                all_hidden_states['block-bn_2-layer2-0'] = F.relu(inside_hidden_states).detach().cpu().numpy()
                out_block0_hidden_states = model.layer2[0](hidden_states)

                # -------- Block1
                # ------- Conv1
                inside_hidden_states = model.layer2[1].conv1(out_block0_hidden_states)
                all_hidden_states['block-conv_1-layer2-1'] = inside_hidden_states.detach().cpu().numpy()
                # ------- Bn1
                inside_hidden_states = F.relu(model.layer2[1].bn1(inside_hidden_states))
                all_hidden_states['block-bn_1-layer2-1'] = inside_hidden_states.detach().cpu().numpy()
                # ------- Conv2
                inside_hidden_states = model.layer2[1].conv2(inside_hidden_states)
                all_hidden_states['block-conv_2-layer2-1'] = inside_hidden_states.detach().cpu().numpy()
                # ------- Bn2
                inside_hidden_states = model.layer2[1].bn2(inside_hidden_states)
                all_hidden_states['block-bn_2-layer2-1'] = F.relu(inside_hidden_states).detach().cpu().numpy()
                out_block1_hidden_states = model.layer2[1](out_block0_hidden_states)

                hidden_states = model.layer2(hidden_states)
                all_hidden_states['layer2'] = hidden_states.detach().cpu().numpy()

                # ----------------------------------------------------------------------------------------

                # --------- Layer3
                # -------- Block0
                # ------- Conv1
                inside_hidden_states = model.layer3[0].conv1(hidden_states)
                all_hidden_states['block-conv_1-layer3-0'] = inside_hidden_states.detach().cpu().numpy()
                # ------- Bn1
                inside_hidden_states = F.relu(model.layer3[0].bn1(inside_hidden_states))
                all_hidden_states['block-bn_1-layer3-0'] = inside_hidden_states.detach().cpu().numpy()
                # ------- Conv2
                inside_hidden_states = model.layer3[0].conv2(inside_hidden_states)
                all_hidden_states['block-conv_2-layer3-0'] = inside_hidden_states.detach().cpu().numpy()
                # ------- Bn2
                inside_hidden_states = model.layer3[0].bn2(inside_hidden_states)
                all_hidden_states['block-bn_2-layer3-0'] = F.relu(inside_hidden_states).detach().cpu().numpy()
                out_block0_hidden_states = model.layer3[0](hidden_states)

                # -------- Block1
                # ------- Conv1
                inside_hidden_states = model.layer3[1].conv1(out_block0_hidden_states)
                all_hidden_states['block-conv_1-layer3-1'] = inside_hidden_states.detach().cpu().numpy()
                # ------- Bn1
                inside_hidden_states = F.relu(model.layer3[1].bn1(inside_hidden_states))
                all_hidden_states['block-bn_1-layer3-1'] = inside_hidden_states.detach().cpu().numpy()
                # ------- Conv2
                inside_hidden_states = model.layer3[1].conv2(inside_hidden_states)
                all_hidden_states['block-conv_2-layer3-1'] = inside_hidden_states.detach().cpu().numpy()
                # ------- Bn2
                inside_hidden_states = model.layer3[1].bn2(inside_hidden_states)
                all_hidden_states['block-bn_2-layer3-1'] = F.relu(inside_hidden_states).detach().cpu().numpy()
                out_block1_hidden_states = model.layer3[1](out_block0_hidden_states)

                hidden_states = model.layer3(hidden_states)
                all_hidden_states['layer3'] = hidden_states.detach().cpu().numpy()

                if dataset != 'cifar100':
                    # ----------------------------------------------------------------------------------------
                    # --------- Layer4
                    # -------- Block0
                    # ------- Conv1
                    inside_hidden_states = model.layer4[0].conv1(hidden_states)
                    all_hidden_states['block-conv_1-layer4-0'] = inside_hidden_states.detach().cpu().numpy()
                    # ------- Bn1
                    inside_hidden_states = F.relu(model.layer4[0].bn1(inside_hidden_states))
                    all_hidden_states['block-bn_1-layer4-0'] = inside_hidden_states.detach().cpu().numpy()
                    # ------- Conv2
                    inside_hidden_states = model.layer4[0].conv2(inside_hidden_states)
                    all_hidden_states['block-conv_2-layer4-0'] = inside_hidden_states.detach().cpu().numpy()
                    # ------- Bn2
                    inside_hidden_states = model.layer4[0].bn2(inside_hidden_states)
                    all_hidden_states['block-bn_2-layer4-0'] = F.relu(inside_hidden_states).detach().cpu().numpy()
                    out_block0_hidden_states = model.layer4[0](hidden_states)

                    # -------- Block1
                    # ------- Conv1
                    inside_hidden_states = model.layer4[1].conv1(out_block0_hidden_states)
                    all_hidden_states['block-conv_1-layer4-1'] = inside_hidden_states.detach().cpu().numpy()
                    # ------- Bn1
                    inside_hidden_states = F.relu(model.layer4[1].bn1(inside_hidden_states))
                    all_hidden_states['block-bn_1-layer4-1'] = inside_hidden_states.detach().cpu().numpy()
                    # ------- Conv2
                    inside_hidden_states = model.layer4[1].conv2(inside_hidden_states)
                    all_hidden_states['block-conv_2-layer4-1'] = inside_hidden_states.detach().cpu().numpy()
                    # ------- Bn2
                    inside_hidden_states = model.layer4[1].bn2(inside_hidden_states)
                    all_hidden_states['block-bn_2-layer4-1'] = F.relu(inside_hidden_states).detach().cpu().numpy()
                    out_block1_hidden_states = model.layer4[1](out_block0_hidden_states)

                    hidden_states = model.layer4(hidden_states)
                    all_hidden_states['layer4'] = hidden_states.detach().cpu().numpy()

                # ----------------------------------------------------------------------------------------

                hidden_states = F.avg_pool2d(hidden_states, 4).view(data.shape[0], -1)
                all_hidden_states['convolution_end'] = hidden_states.detach().cpu().numpy()

                # ----------------------------------------------------------------------------------------

                logits = model(data)
                all_hidden_states['logits'] = logits.detach().cpu().numpy()

                # ----------------------------------------------------------------------------------------

                pred = torch.nn.Softmax(dim=1)(logits)
                all_hidden_states['pred'] = pred.detach().cpu().numpy()

            else:
                data = torch.tensor(loader[i * bs: (i + 1) * bs]).to(device)
                hidden_states = data
                # Conv1
                hidden_states = model.conv1(hidden_states)
                # all_hidden_states['conv1'] = np.concatenate(( all_hidden_states['conv1'], hidden_states.detach().cpu().numpy()), axis=0)
                # Bn1
                hidden_states = model.bn1(hidden_states)
                # all_hidden_states['bn1'] = np.concatenate(( all_hidden_states['bn1'], hidden_states.detach().cpu().numpy()), axis=0)
                hidden_states = F.relu(hidden_states)

                # ----------------------------------------------------------------------------------------
                # --------- Layer1
                # -------- Block0
                # ------- Conv1
                inside_hidden_states = model.layer1[0].conv1(hidden_states)
                all_hidden_states['block-conv_1-layer1-0'] = np.concatenate((all_hidden_states['block-conv_1-layer1-0'], inside_hidden_states.detach().cpu().numpy()), axis=0)
                # ------- Bn1
                inside_hidden_states = F.relu(model.layer1[0].bn1(inside_hidden_states))
                all_hidden_states['block-bn_1-layer1-0'] = np.concatenate((all_hidden_states['block-bn_1-layer1-0'], inside_hidden_states.detach().cpu().numpy()), axis=0)
                # ------- Conv2
                inside_hidden_states = model.layer1[0].conv2(inside_hidden_states)
                all_hidden_states['block-conv_2-layer1-0'] = np.concatenate((all_hidden_states['block-conv_2-layer1-0'], inside_hidden_states.detach().cpu().numpy()), axis=0)
                # ------- Bn2
                inside_hidden_states = model.layer1[0].bn2(inside_hidden_states)
                all_hidden_states['block-bn_2-layer1-0'] = np.concatenate((all_hidden_states['block-bn_2-layer1-0'], F.relu(inside_hidden_states).detach().cpu().numpy()), axis=0)
                out_block0_hidden_states = model.layer1[0](hidden_states)

                # -------- Block1
                # ------- Conv1
                inside_hidden_states = model.layer1[1].conv1(out_block0_hidden_states)
                all_hidden_states['block-conv_1-layer1-1'] = np.concatenate((all_hidden_states['block-conv_1-layer1-1'], inside_hidden_states.detach().cpu().numpy()), axis=0)
                # ------- Bn1
                inside_hidden_states = F.relu(model.layer1[1].bn1(inside_hidden_states))
                all_hidden_states['block-bn_1-layer1-1'] = np.concatenate((all_hidden_states['block-bn_1-layer1-1'], inside_hidden_states.detach().cpu().numpy()), axis=0)
                # ------- Conv2
                inside_hidden_states = model.layer1[1].conv2(inside_hidden_states)
                all_hidden_states['block-conv_2-layer1-1'] = np.concatenate((all_hidden_states['block-conv_2-layer1-1'], inside_hidden_states.detach().cpu().numpy()), axis=0)
                # ------- Bn2
                inside_hidden_states = model.layer1[1].bn2(inside_hidden_states)
                all_hidden_states['block-bn_2-layer1-1'] = np.concatenate((all_hidden_states['block-bn_2-layer1-1'], F.relu(inside_hidden_states).detach().cpu().numpy()), axis=0)
                out_block1_hidden_states = model.layer1[1](out_block0_hidden_states)

                hidden_states = model.layer1(hidden_states)
                all_hidden_states['layer1'] = np.concatenate((all_hidden_states['layer1'], hidden_states.detach().cpu().numpy()), axis=0)

                # ----------------------------------------------------------------------------------------

                # --------- Layer2
                # -------- Block0
                # ------- Conv1
                inside_hidden_states = model.layer2[0].conv1(hidden_states)
                all_hidden_states['block-conv_1-layer2-0'] = np.concatenate((all_hidden_states['block-conv_1-layer2-0'], inside_hidden_states.detach().cpu().numpy()), axis=0)
                # ------- Bn1
                inside_hidden_states = F.relu(model.layer2[0].bn1(inside_hidden_states))
                all_hidden_states['block-bn_1-layer2-0'] = np.concatenate((all_hidden_states['block-bn_1-layer2-0'], inside_hidden_states.detach().cpu().numpy()), axis=0)
                # ------- Conv2
                inside_hidden_states = model.layer2[0].conv2(inside_hidden_states)
                all_hidden_states['block-conv_2-layer2-0'] = np.concatenate((all_hidden_states['block-conv_2-layer2-0'], inside_hidden_states.detach().cpu().numpy()), axis=0)
                # ------- Bn2
                inside_hidden_states = model.layer2[0].bn2(inside_hidden_states)
                all_hidden_states['block-bn_2-layer2-0'] = np.concatenate((all_hidden_states['block-bn_2-layer2-0'], F.relu(inside_hidden_states).detach().cpu().numpy()), axis=0)
                out_block0_hidden_states = model.layer2[0](hidden_states)

                # -------- Block1
                # ------- Conv1
                inside_hidden_states = model.layer2[1].conv1(out_block0_hidden_states)
                all_hidden_states['block-conv_1-layer2-1'] = np.concatenate((all_hidden_states['block-conv_1-layer2-1'], inside_hidden_states.detach().cpu().numpy()), axis=0)
                # ------- Bn1
                inside_hidden_states = F.relu(model.layer2[1].bn1(inside_hidden_states))
                all_hidden_states['block-bn_1-layer2-1'] = np.concatenate((all_hidden_states['block-bn_1-layer2-1'], inside_hidden_states.detach().cpu().numpy()), axis=0)
                # ------- Conv2
                inside_hidden_states = model.layer2[1].conv2(inside_hidden_states)
                all_hidden_states['block-conv_2-layer2-1'] = np.concatenate((all_hidden_states['block-conv_2-layer2-1'], inside_hidden_states.detach().cpu().numpy()), axis=0)
                # ------- Bn2
                inside_hidden_states = model.layer2[1].bn2(inside_hidden_states)
                all_hidden_states['block-bn_2-layer2-1'] = np.concatenate((all_hidden_states['block-bn_2-layer2-1'], F.relu(inside_hidden_states).detach().cpu().numpy()), axis=0)
                out_block1_hidden_states = model.layer2[1](out_block0_hidden_states)

                hidden_states = model.layer2(hidden_states)
                all_hidden_states['layer2'] = np.concatenate((all_hidden_states['layer2'], hidden_states.detach().cpu().numpy()), axis=0)

                # ----------------------------------------------------------------------------------------

                # --------- Layer3
                # -------- Block0
                # ------- Conv1
                inside_hidden_states = model.layer3[0].conv1(hidden_states)
                all_hidden_states['block-conv_1-layer3-0'] = np.concatenate((all_hidden_states['block-conv_1-layer3-0'], inside_hidden_states.detach().cpu().numpy()), axis=0)
                # ------- Bn1
                inside_hidden_states = F.relu(model.layer3[0].bn1(inside_hidden_states))
                all_hidden_states['block-bn_1-layer3-0'] = np.concatenate((all_hidden_states['block-bn_1-layer3-0'], inside_hidden_states.detach().cpu().numpy()), axis=0)
                # ------- Conv2
                inside_hidden_states = model.layer3[0].conv2(inside_hidden_states)
                all_hidden_states['block-conv_2-layer3-0'] = np.concatenate((all_hidden_states['block-conv_2-layer3-0'], inside_hidden_states.detach().cpu().numpy()), axis=0)
                # ------- Bn2
                inside_hidden_states = model.layer3[0].bn2(inside_hidden_states)
                all_hidden_states['block-bn_2-layer3-0'] = np.concatenate((all_hidden_states['block-bn_2-layer3-0'], F.relu(inside_hidden_states).detach().cpu().numpy()), axis=0)
                out_block0_hidden_states = model.layer3[0](hidden_states)

                # -------- Block1
                # ------- Conv1
                inside_hidden_states = model.layer3[1].conv1(out_block0_hidden_states)
                all_hidden_states['block-conv_1-layer3-1'] = np.concatenate((all_hidden_states['block-conv_1-layer3-1'], inside_hidden_states.detach().cpu().numpy()), axis=0)
                # ------- Bn1
                inside_hidden_states = F.relu(model.layer3[1].bn1(inside_hidden_states))
                all_hidden_states['block-bn_1-layer3-1'] = np.concatenate((all_hidden_states['block-bn_1-layer3-1'], inside_hidden_states.detach().cpu().numpy()), axis=0)
                # ------- Conv2
                inside_hidden_states = model.layer3[1].conv2(inside_hidden_states)
                all_hidden_states['block-conv_2-layer3-1'] = np.concatenate((all_hidden_states['block-conv_2-layer3-1'], inside_hidden_states.detach().cpu().numpy()), axis=0)
                # ------- Bn2
                inside_hidden_states = model.layer3[1].bn2(inside_hidden_states)
                all_hidden_states['block-bn_2-layer3-1'] = np.concatenate((all_hidden_states['block-bn_2-layer3-1'], F.relu(inside_hidden_states).detach().cpu().numpy()), axis=0)
                out_block1_hidden_states = model.layer3[1](out_block0_hidden_states)

                hidden_states = model.layer3(hidden_states)
                all_hidden_states['layer3'] = np.concatenate((all_hidden_states['layer3'], hidden_states.detach().cpu().numpy()), axis=0)

                if dataset != 'cifar100':
                    # ----------------------------------------------------------------------------------------
                    # --------- Layer4
                    # -------- Block0
                    # ------- Conv1
                    inside_hidden_states = model.layer4[0].conv1(hidden_states)
                    all_hidden_states['block-conv_1-layer4-0'] = np.concatenate((all_hidden_states['block-conv_1-layer4-0'], inside_hidden_states.detach().cpu().numpy()), axis=0)
                    # ------- Bn1
                    inside_hidden_states = F.relu(model.layer4[0].bn1(inside_hidden_states))
                    all_hidden_states['block-bn_1-layer4-0'] = np.concatenate((all_hidden_states['block-bn_1-layer4-0'], inside_hidden_states.detach().cpu().numpy()), axis=0)
                    # ------- Conv2
                    inside_hidden_states = model.layer4[0].conv2(inside_hidden_states)
                    all_hidden_states['block-conv_2-layer4-0'] = np.concatenate((all_hidden_states['block-conv_2-layer4-0'], inside_hidden_states.detach().cpu().numpy()), axis=0)
                    # ------- Bn2
                    inside_hidden_states = model.layer4[0].bn2(inside_hidden_states)
                    all_hidden_states['block-bn_2-layer4-0'] = np.concatenate((all_hidden_states['block-bn_2-layer4-0'], F.relu(inside_hidden_states).detach().cpu().numpy()), axis=0)
                    out_block0_hidden_states = model.layer4[0](hidden_states)

                    # -------- Block1
                    # ------- Conv1
                    inside_hidden_states = model.layer4[1].conv1(out_block0_hidden_states)
                    all_hidden_states['block-conv_1-layer4-1'] = np.concatenate((all_hidden_states['block-conv_1-layer4-1'], inside_hidden_states.detach().cpu().numpy()), axis=0)
                    # ------- Bn1
                    inside_hidden_states = F.relu(model.layer4[1].bn1(inside_hidden_states))
                    all_hidden_states['block-bn_1-layer4-1'] = np.concatenate((all_hidden_states['block-bn_1-layer4-1'], inside_hidden_states.detach().cpu().numpy()), axis=0)
                    # ------- Conv2
                    inside_hidden_states = model.layer4[1].conv2(inside_hidden_states)
                    all_hidden_states['block-conv_2-layer4-1'] = np.concatenate((all_hidden_states['block-conv_2-layer4-1'], inside_hidden_states.detach().cpu().numpy()), axis=0)
                    # ------- Bn2
                    inside_hidden_states = model.layer4[1].bn2(inside_hidden_states)
                    all_hidden_states['block-bn_2-layer4-1'] = np.concatenate((all_hidden_states['block-bn_2-layer4-1'], F.relu(inside_hidden_states).detach().cpu().numpy()), axis=0)
                    out_block1_hidden_states = model.layer4[1](out_block0_hidden_states)

                    hidden_states = model.layer4(hidden_states)
                    all_hidden_states['layer4'] = np.concatenate((all_hidden_states['layer4'], hidden_states.detach().cpu().numpy()), axis=0)

                hidden_states = F.avg_pool2d(hidden_states, 4).view(data.shape[0], -1)
                all_hidden_states['convolution_end'] = np.concatenate((all_hidden_states['convolution_end'], hidden_states.detach().cpu().numpy()), axis=0)

                logits = model(data)
                all_hidden_states['logits'] = np.concatenate((all_hidden_states['logits'], logits.detach().cpu().numpy()), axis=0)
                pred = torch.nn.Softmax(dim=1)(logits)
                all_hidden_states['pred'] = np.concatenate((all_hidden_states['pred'], pred.detach().cpu().numpy()), axis=0)

    return all_hidden_states
