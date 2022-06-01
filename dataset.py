import os
import random

from torch.utils.data import Dataset
from typing import Optional, Sequence, Callable
from PIL import Image

from augment import RandAugment


class FgvcAircraftDataset(Dataset):
    def __init__(
        self,
        meta_path: str,
        image_path: str,
        transforms: Sequence[Callable] = None,
        seed: Optional[int] = None
    ):
        # Variable initialization
        self.meta_path = meta_path
        self.image_path = image_path
        self.num_classes = 100
        self.transform = transforms
        self.im_list = []
        labels = []

        with open(meta_path, 'r') as f:
            meta = f.readlines()
            for line in meta:
                deli_pos = line.find(' ')
                im_id = line[:deli_pos]
                label = line[deli_pos + 1:]
                if label not in labels:
                    labels.append(label)
                self.im_list.append((im_id, labels.index(label)))
        self.labels = labels[1:]

        if seed is not None:
            rd = random.Random(seed)
            rd.shuffle(self.im_list)
        else:
            random.shuffle(self.im_list)

    def __len__(self):
        return len(self.im_list)
    
    def __getitem__(self, idx: int):
        # Get image path
        im_name, class_idx = self.im_list[idx]
        im_path = os.path.join(
            self.image_path, im_name + '.jpg'
        )

        # Get image
        im = Image.open(im_path)

        # Transform image
        if self.transform is not None:
            im = self.transform(im)

        return im, class_idx


class FgvcAircraftSimCLR(Dataset):
    def __init__(
        self,
        meta_path: str,
        image_path: str,
        transforms: Sequence[Callable],
        seed: Optional[int] = None
    ):
        # Variable initialization
        self.meta_path = meta_path
        self.image_path = image_path
        self.transform = transforms
        self.im_list = []\

        with open(meta_path, 'r') as f:
            meta = f.readlines()
            for line in meta:
                deli_pos = line.find(' ')
                im_id = line[:deli_pos]
                self.im_list.append(im_id)

        if seed is not None:
            rd = random.Random(seed)
            rd.shuffle(self.im_list)
        else:
            random.shuffle(self.im_list)

    def __len__(self):
        return len(self.im_list)
    
    def __getitem__(self, idx: int):
        # Get image path
        im_name = self.im_list[idx]
        im_path = os.path.join(
            self.image_path, im_name + '.jpg'
        )

        # Get image
        im = Image.open(im_path)

        # Transform image
        im1 = self.transform(im)
        im2 = self.transform(im)

        return im1, im2


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import torchvision.transforms as T
    ds = FgvcAircraftDataset(
        meta_path='./fgvc-aircraft-2013b/data/images_family_train.txt',
        image_path='./fgvc-aircraft-2013b/data/images',
        transforms=T.Compose([
            T.Resize(320),
            RandAugment(2, 9),
            T.RandomResizedCrop(224),
        ])
    )
    plt.imshow(ds[0][0])
    plt.axis('off')
    plt.title(ds.labels[ds[0][1]])
    plt.show()
