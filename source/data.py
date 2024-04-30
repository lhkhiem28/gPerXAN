import os, sys
from libs import *

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, 
        data_dir, 
        augment = False, 
    ):
        self.image_paths = glob.glob(data_dir + "*/*")
        if augment:
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p = 0.5), 
                    A.ColorJitter(p = 0.5), 
                    A.Resize(
                        height = 224, width = 224, 
                    ), 
                    A.Normalize(), AT.ToTensorV2(), 
                ]
            )
        else:
            self.transform = A.Compose(
                [
                    A.Resize(
                        height = 224, width = 224, 
                    ), 
                    A.Normalize(), AT.ToTensorV2(), 
                ]
            )

    def __len__(self, 
    ):
        return len(self.image_paths)

    def __getitem__(self, 
        index, 
    ):
        image_path = self.image_paths[index]
        label = int(image_path.split("/")[-2])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(
            image, 
            code = cv2.COLOR_BGR2RGB, 
        )
        image = self.transform(image = image)["image"]

        return image, label