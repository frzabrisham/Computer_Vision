import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CustomImageDataset(Dataset):

    def __init__(self, mode):
        base_path = 'Data/' + mode
        self.dataset = []
        labels = CustomImageDataset.get_labels()
        self.n_labels = len(labels)
        for label, label_number in labels.items():
            for image in os.listdir('{}/{}'.format(base_path, label)):
                self.dataset.append({
                    'image_path': '{}/{}/{}'.format(base_path, label, image),
                    'label': label_number,
                })

        self.trans = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.flip = transforms.RandomHorizontalFlip(1.1)
        self.blur = transforms.GaussianBlur((3, 3))
        self.sharp = transforms.RandomAdjustSharpness(2, 1.1)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = Image.open(self.dataset[idx]['image_path']).convert("RGB")
        image = self.trans(transforms.ToTensor()(image))
        label = self.dataset[idx]['label']

        return image, label

    @staticmethod
    def __get_immediate_subdirectories(a_dir):
        return [name for name in os.listdir(a_dir)
                if os.path.isdir(os.path.join(a_dir, name))]

    @staticmethod
    def get_labels():
        labels = sorted(CustomImageDataset.__get_immediate_subdirectories('Data/Train'))
        labels = {label: i for i, label in enumerate(labels)}
        return labels
