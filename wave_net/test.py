import pandas as pd
import torch 
from Wavenet import WaveNet,WaveNetClassifier
from tqdm import tqdm 

df=pd.read_csv("train_clean.csv")
df['signal']=df['signal']/df['signal'].max()

class TimesereisDataset(torch.utils.data.Dataset):
    def __init__(self, df:pd.Series(), seqLen=10):
        super().__init__()
        self.df=df
        self.seqLen=seqLen
    
    def __len__(self):
        return self.df.shape[0]-self.seqLen-1
    
    def __getitem__(self,index):
        x=self.df.iloc[index:index+self.seqLen,1].values
        y=self.df.iloc[index+self.seqLen-1,2] 
        return x,y 
      
seqLen=50
timeSeriesDataset = TimesereisDataset(df,seqLen)
trainNumbers=int(len(timeSeriesDataset)*0.9)
trainDataset,testDataset=torch.utils.data.random_split(timeSeriesDataset,[trainNumbers,len(timeSeriesDataset)-trainNumbers])
trainDataLoader=torch.utils.data.DataLoader(trainDataset,batch_size=8,shuffle=True)
testDataLoader=torch.utils.data.DataLoader(testDataset,batch_size=8,shuffle=True)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device="cpu"
wavenetClassifierModel=WaveNetClassifier(seqLen,df['open_channels'].max()+1)
wavenetClassifierModel.to(device)

wavenetClassifierModel.train()

optimizer=torch.optim.AdamW(wavenetClassifierModel.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
lossFunction = torch.nn.CrossEntropyLoss()

def calc_accuracy(Out,Y):
    max_vals, max_indices = torch.max(Out,1)
    train_acc = (max_indices == Y).sum().item()/max_indices.size()[0]
    return train_acc
  
epochs=1
globalStep=500

for epoch in range(epochs):
    for step, (x_train,y_train) in tqdm(enumerate(trainDataLoader),desc="Training"):
         x_train = torch.unsqueeze(x_train,dim=1).float()
         x_train.to(device)
         y_train.to(device)
         output=wavenetClassifierModel(x_train)
         output = torch.squeeze(output,dim=1)

         loss= lossFunction(output,y_train)
         optimizer.zero_grad()
         loss.backward()
         optimizer.step()
         if step%globalStep==0:
            # scheduler.step()
            # print(output.detach().numpy())
            # print(y_train.numpy())
            with torch.no_grad():
                accuracy=0
                loss=0
                for stepTest, (x_test,y_test) in tqdm(enumerate(testDataLoader),desc="Validation"):
                    x_test.to(device)
                    y_test.to(device)
                    x_test = torch.unsqueeze(x_test,dim=1).float()
                    output=wavenetClassifierModel(x_test)
                    output = torch.squeeze(output,dim=1)
                    accuracy+=calc_accuracy(output,y_test)*100
                    loss+= lossFunction(output,y_test).item()
                    if stepTest>200:
                        break
            print(f"loss for step {step} : {loss/stepTest}  :  Accuracy: {accuracy/stepTest} %")

         
    print(f"epoch {epoch}")