import pathlib

import torch
import torchvision
from torchvision.transforms import transforms
from PIL import Image


# TODO: Hardcode
CELL_TYPES = {
    'erythropoiesis': {
        'Proerythroblast': 'PEB',
        'Erythroblast': 'EBO',
    },
    'lymphoid': {
        'Immature Lymphocyte': 'LYI',
        'Lymphocyte': 'LYT',
        'Plasma Cell': 'PLM',
    },
    'myeloid': {
        'myeloid_immature': {
            'Myeloblast': 'BLA',
            'Metamyelocyte': 'MMZ',
            'Myelocyte': 'MYB',
            'Promyelocyte': 'PMO',
        },
        'myeloid_mature': {
            'Neutrophil': {
                'Band Neutrophil': 'NGB',
                'Segmented Neutrophil': 'NGS',
            },
            'Basophil': 'BAS',
            'Eosinophil': 'EOS',
            'Monocyte': 'MON',
        },
    },
    'abnormal': {
        'Not Identifiable': 'NIF',
        'Other Cell': 'OTH',
        'Abnormal Eosinophil': 'ABE',
        'Artefact': 'ART',
        'Smudge Cell': 'KSC',
        'Faggott Cell': 'FGC'
    }
}

# TODO: Also Hardcode.
CLASSIFICATION_GRAPH = {
    0: {
        'root': {0: 'erythropoiesis', 1: 'lymphoid', 2: 'myeloid', 3: 'abnormal'}
    },
    1: {
        'erythropoiesis': {0: 'Proerythroblast', 1: 'Erythroblast'},
        'lymphoid': {0: 'Lymphocyte', 1: 'Plasma Cell'},
        'myeloid': {0: 'myeloid_immature', 1: 'myeloid_mature'},
        'abnormal': {0: 'Not Identifiable', 1: 'Artefact'}
    },
    2: {
        'myeloid_immature': {0: 'Myeloblast', 1: 'Metamyelocyte', 2: 'Myelocyte', 3: 'Promyelocyte'},
        'myeloid_mature': {0: 'Neutrophil', 1: 'Eosinophil', 2: 'Monocyte'}
    },
    3: {
        'Neutrophil': {0: 'Band Neutrophil', 1: 'Segmented Neutrophil'}
    },
}

DEPTH = 2

MODELS_PATH = pathlib.Path(__file__).parent.resolve() / 'models'

NN_NAME = 'resnet'
CASE = 'pretrained_augmentated_'


def classify(image_path):
    all_class_names = list()
    for v in CLASSIFICATION_GRAPH.values():
        all_class_names += list(v.keys())
    models_names = dict()
    for name in all_class_names:
        models_names[name] = NN_NAME + '_' + CASE + name
    class_name = 'root'
    for depth in range(DEPTH + 1):
        if class_name not in CLASSIFICATION_GRAPH[depth]:
            continue
        model = torchvision.models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, len(CLASSIFICATION_GRAPH[depth][class_name]))

        # crystal = "cuda:0" if torch.cuda.is_available() else "cpu"
        crystal = 'cpu'
        device = torch.device(crystal)
        model.to(device)
        if crystal == 'cpu':
            model.load_state_dict(torch.load(MODELS_PATH / models_names[class_name], map_location='cpu'))
        else:
            model.load_state_dict(torch.load(MODELS_PATH / models_names[class_name]))
        model.eval()

        image_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        image = Image.open(image_path)
        image = image_transforms(image).float()
        image = image.unsqueeze(0)

        output = model(image)
        _, predicted = torch.max(output.data, 1)
        class_name = CLASSIFICATION_GRAPH[depth][class_name][predicted.item()]
    return class_name
