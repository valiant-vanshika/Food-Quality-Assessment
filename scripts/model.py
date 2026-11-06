import math
import torch
import torchmetrics
import torch.nn as nn
from tqdm import tqdm 


class Model(nn.Module):
    def __init__(self,base):
        super().__init__()
        self.alpha = 0.7
        self.base = base
        for param in list(self.base.parameters())[:-15]:
            param.requires_grad=False
        self.base.classifier = nn.Sequential()
        self.base.fc = nn.Sequential()
        self.block1 = nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256,128),
        )
        self.block2 = nn.Sequential(
            nn.Linear(128,32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32,2),
        )
        self.optimizer1 = torch.optim.Adam([
            {"params":self.base.parameters(),'lr':1e-5},
            {"params":self.block1.parameters(),'lr':3e-4}
        ])
        self.optimizer2 = torch.optim.Adam(self.block2.parameters(),lr=3e-4)
        self.loss_fxn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task='multiclass',num_classes=2)
        self.TRAIN_BATCHES = math.ceil(len(train)/BATCH_SIZE)
        self.VAL_BATCHES = math.ceil(len(test)/BATCH_SIZE)
        self.history = {'train_loss':[],'val_loss':[],'train_acc':[],'val_acc':[]}

    def forward(self,x):
        x = self.base(x)
        x = self.block1(x)
        y = self.block2(x)
        return y

    def train_step(self,x,y):
        pred = self.forward(x)
        l = self.loss_fxn(pred,y)
        self.optimizer2.zero_grad()
        self.optimizer1.zero_grad()
        l.backward(retain_graph=True)
        loss = self.alpha*l
        loss.backward()
        self.optimizer2.step()
        self.optimizer1.step()
        accuracy = self.accuracy(torch.argmax(pred,axis=1),y)
        return loss,accuracy                 

    def val_step(self,x,y):
        with torch.no_grad():
            pred = self.forward(x)
            loss = self.alpha*self.loss_fxn(pred,y)
            accuracy = self.accuracy(torch.argmax(pred,axis=1),y)
            return loss,accuracy

    def update_history(self,train_loss,val_loss,train_acc,val_acc):
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['train_acc'].append(train_acc)        
        self.history['val_acc'].append(val_acc)

    def train(self,epochs=5):
        for epoch in tqdm(range(epochs)):
            train_loss,train_acc = 0,0
            val_loss,val_acc = 0,0
            for X,y in tqdm(train_loader):
                X,y = [v.to(device) for v in (X,y)]
                loss,acc = self.train_step(X,y)
                train_loss+=loss.item()
                train_acc+=acc.item()
            for X,y in tqdm(test_loader):
                X,y = [v.to(device) for v in (X,y)]
                loss,acc = self.val_step(X,y)
                val_loss+=loss.item()
                val_acc+=acc.item()
            train_loss,train_acc=[x/self.TRAIN_BATCHES for x in (train_loss,train_acc)]
            val_loss,val_acc=[x/self.VAL_BATCHES for x in (val_loss,val_acc)]
            self.update_history(train_loss,val_loss,train_acc,val_acc)
            print(f'Epoch:{epoch+1}\nTraining:[loss:{train_loss:.4f},accuracy:{train_acc*100:.3f}],\n Validation:[loss:{val_loss:.4f},accuracy:{val_acc*100:.3f}]')
