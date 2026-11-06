from PIL import Image
from torch.utils.data import Dataset
from torchvision.models import ResNet18_Weights as weights


transforms = weights.DEFAULT.transforms(antialias=True)
class FoodDataset(Dataset):
    def __init__(self,df):
        self.df = df
        self.n_samples = len(self.df)
        self.shape = self.df.shape

    def __len__(self):
        return self.n_samples

    def __getitem__(self,idx):
        image = Image.open(self.df.at[idx,'images']).convert('RGB')
        image = transforms(image)
        label = self.df.at[idx,'labels']
        return image,label