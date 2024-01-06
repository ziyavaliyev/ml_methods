import torch
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
#from torchvision.transforms import v2
import torch

transform = transforms.ToTensor()

image = "/Users/ziya03/Downloads/360_F_29828143_RkHCM5hFK8ZcuT35xrAOYVAsNFIQ6MHN.jpg"

#image = Path(path)

image_pil = Image.open(image)

# Apply the transformation to the image
#image_tensor = transform(image_pil)

train_dataset_transform = transforms.Compose([transforms.Resize(64), transforms.ToTensor(),
                                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                std=[0.229, 0.224, 0.225])])
val_dataset_transform = transforms.Compose([transforms.Resize(64), transforms.ToTensor(),
                                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                std=[0.229, 0.224, 0.225])])
test_dataset_transform = transforms.Compose([transforms.Resize(64), transforms.ToTensor(),
                                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                std=[0.229, 0.224, 0.225])])

print(train_dataset_transform == val_dataset_transform)
transformed = train_dataset_transform(image_pil)
#count = int(torch.sum(image_tensor > 0).item()/3)



print(torch.cuda.is_available())
print(transformed)

