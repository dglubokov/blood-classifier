import torch
import torchvision
from torchvision.transforms import transforms
from PIL import Image


class_names = {
    'Нормобласты': 0.0,
    'Сегментноядерный нейтрофил': 1.0,
    'Палочкоядерный нейтрофил': 2.0,
    'Миелоцит': 3.0,
    'Бласты': 4.0,
 }


def classify(image_path, model_path='model'):
    model = torchvision.models.resnet34(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, len(class_names))
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    model.load_state_dict(torch.load(model_path))
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
    inv_map = {v: k for k, v in class_names.items()}
    return inv_map[predicted.item()]
