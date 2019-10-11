from torch.utils.data import Dataset, DataLoader
from PIL import Image

class TrainingDataset(Dataset):
    def __init__(self, img, transform=None):
        self.img= img
        self.transform=transform
    
    def __getitem__(self, index):
        gt_path=self.img[index][3]
        im_path=[]
        img=[]
        for i in range(3):
            im_path.append(self.img[index][i])
        for im in im_path:
            im_opened=Image.open(im).convert('RGB')
            if self.transform is not None:
                img.append(self.transform(im_opened))
        gt = Image.open(gt_path).convert('RGB')
        if self.transform is not None:
            gt = self.transform(gt)
        return (img,gt)

    def __len__(self):
        return len(self.img)