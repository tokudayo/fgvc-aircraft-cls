import argparse

import torch
import torchvision.transforms as T
from torchvision.models import convnext_tiny

from dataset import FgvcAircraftDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    return parser.parse_args()


def test(model_path):
    model = convnext_tiny(pretrained=True)
    model.classifier[2] = torch.nn.Linear(768, 100)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)

    model.eval().cuda()

    augment_weak = T.Compose([
        T.Resize((300, 300)),
        T.ToTensor(),
        # T.Normalize(mean=[0.4796, 0.5107, 0.5341], std=[0.1957, 0.1945, 0.2162])
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_loader = torch.utils.data.DataLoader(
        FgvcAircraftDataset(
            meta_path='./fgvc-aircraft-2013b/data/images_variant_test.txt',
            image_path='./fgvc-aircraft-2013b/data/images',
            transforms=augment_weak,
        ),
        batch_size=32,
        shuffle=False,
        num_workers=4,
    )
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += len(data)
    print(f'Accuracy: {correct}/{total} ({100. * correct / total:.2f}%)')


if __name__ == '__main__':
    args = parse_args()
    test(args.model_path)