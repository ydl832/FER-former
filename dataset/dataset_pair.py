import torch.utils.data as data
from PIL import Image, ImageFile
import torch

ImageFile.LOAD_TRUNCATED_IAMGES = True

# https://github.com/pytorch/vision/issues/81

def PIL_loader(path):
    try:
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')
    except IOError:
        print('Cannot load image ' + path)
        
#..............................................................................................
def default_reader(fileList):
    imgList = []
    with open(fileList, 'r') as file:
        for line in file.readlines():
            path, a, f = line.strip().split()
            imgList.append((path, str(a), int(f)))
    return imgList


class ImageList(data.Dataset):
    def __init__(self, fileList, transform=None, list_reader=default_reader, loader=PIL_loader):
        self.imgList = list_reader(fileList)
        self.transform = transform
        self.loader = loader
        
    def __getitem__(self, index):
        imgPath, label, label1 = self.imgList[index]

        img_name = imgPath.split('/')[-1]
        # img_name = imgPath.split('/')[-1][5:-12]
        img = self.loader(imgPath)

        if self.transform is not None:
            img = self.transform(img)
        return img, label, label1, imgPath, img_name

    def __len__(self):
        return len(self.imgList)

