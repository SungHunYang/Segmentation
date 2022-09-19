import os
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, dir_path, normalize=False):
        self.dir_path = dir_path
        self.all_imgs = os.listdir(f'{dir_path}/origin')

        self.normalize = normalize
        self.normalizer = T.Normalize(mean=(0.4021, 0.3812, 0.3748), std=(0.2262, 0.2185, 0.2071))

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        origin_img_loc = os.path.join(f'{self.dir_path}/origin', self.all_imgs[idx])
        truth_img_loc = os.path.join(f'{self.dir_path}/truth', self.all_imgs[idx])

        image = Image.open(origin_img_loc)
        label = Image.open(truth_img_loc)

        image = T.ToTensor()(image)
        label = T.ToTensor()(label)

        if self.normalize:
            image = self.normalizer(image)

        label = T.Grayscale(num_output_channels=1)(label)

        return image, label
