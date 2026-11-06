from .model import Model
from .dataset import FoodDataset
import seaborn as sns 
from torch.utils.data import DataLoader
from torchvision import models



model = Model(models.resnet18(weights='IMAGENET1K_V1')).to(device)
model.train(epochs=3)

# Utility Function to classify any fruit 
def classify_fruit(fruit_name):
    train_loader = DataLoader(FoodDataset(train_df.loc[train_df['fruits']==fruit_name].reset_index()),batch_size=BATCH_SIZE,shuffle=True)
    test_loader = DataLoader(FoodDataset(test_df.loc[test_df['fruits']==fruit_name].reset_index()),batch_size=BATCH_SIZE,shuffle=True)
    inputs,classes = next(iter(train_loader))
    imshow(make_grid(inputs))
    model.train()
    fig = plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.title("Loss")
    plt.plot(model.history['train_loss'],label="Training")
    plt.plot(model.history['val_loss'],label="Validation")
    plt.legend()
    plt.subplot(1,2,2)
    plt.title("Accuracy")
    plt.plot(model.history['train_acc'],label="Training")
    plt.plot(model.history['val_acc'],label="Validation")
    plt.legend()
    plt.show()
    y_true=[]
    y_pred=[]
    with torch.no_grad():
        for X,y in tqdm(test_loader):
            pred = model(X.to(device))
            pred = torch.argmax(pred,axis=1).detach().cpu().numpy()
            y_true.extend(y)
            y_pred.extend(pred)
    sns.heatmap(confusion_matrix(y_true,y_pred,normalize='true'),cmap='Blues',cbar=True,annot=True)
