# HAMPER: A Halfspace-Mass Depth-Based Method for Adversarial Attack Detection
HAMPER is a a new method to detect adversarial examples by leveraging the concept of data depths, a statistical
notion that provides center-outward ordering of points with respect to a probability distribution. In particular, the halfspace-mass (HM) depth exhibits attractive properties
such as computational efficiency, which makes it a natural candidate for adversarial attack detection in high-dimensional spaces. 

The adversarial examples have been created by executing this <a href="https://github.com/aldahdooh/detectors_review">code</a>. 

The repository with the model checkpoints is available at <a href="https://drive.google.com/drive/folders/13LfOUq2IS4j0vhlIV-bWO5iTLGGlREDL?usp=sharing" >link</a>.


### Current package structure
```
Package
HAMPER/
├── README.md
├── adv_data
├── checkpoints
│   ├── classifier
│   │   ├── cifar10
│   │   │   └── rn-best.pt
│   │   ├── cifar100
│   │   │   └── resnet
│   │   │       └── model_best.pth.tar
│   │   └── svhn
│   │       └── rn-best.pt
│   └── detectors
├── depth
│   ├── __init__.py
│   ├── data_depth.py
│   └── extract_depth_features.py
├── evaluation
│   ├── __init__.py
│   └── evaluation.py
├── hamper_detectors
│   ├── __init__.py
│   ├── hamper.py
│   ├── hamper_aa.py
│   └── hamper_ba.py
├── main.py
├── models
│   ├── __init__.py
│   ├── detector.py
│   ├── resnet101.py
│   └── resnet18.py
├── parser.py
├── requirements.txt
├── setup.py
└── utils
    ├── __init__.py
    ├── utils_data.py
    ├── utils_depth.py
    ├── utils_general.py
    ├── utils_ml.py
    └── utils_models.py
```

#### Usage

To execute HAMPER:
- Create the environment for HAMPER:
```console
foo@bar:~$ conda create --name hamper python==3.8.11
```
- Activate the enviroment for SALAD:
```console
foo@bar:~$ source activate hamper
```
- Install all the required packages:
```console
(hamper) foo@bar:~$ pip3 install -r requirements.txt
```
- Launch the test from CLI for CIFAR10, CIFAR100 or SVHN (see <code>parser.py</code>):
```console
(hamper) foo@bar:~$ python main.py -d cifar10 -a pgdi -e 0.03125 -ht aa
```
Output
```
----------------------------------------------------------  -----------
Hamper Attack-Aware Average Score over natural samples        0.979069
Hamper Attack-Aware Average Score over adversarial samples    0.0150045
AUROC                                                       100
FPR at 95% of TPR                                             0
----------------------------------------------------------  -----------
```
