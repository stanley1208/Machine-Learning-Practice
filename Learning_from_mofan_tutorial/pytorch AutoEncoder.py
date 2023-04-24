import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision

EPOCH=10
BATCH_SIZE=64
LR=0.005
DOWMLOAD_MNIST=False
N_TEST_IMG=5

train_data=torchvision.datasets.MNIST(root='./mnist',
                                      train=True,
                                      transform=torchvision.transforms.ToTensor(),
                                      download=DOWMLOAD_MNIST)

train_loader=Data.DataLoader(dataset=train_data
                             ,batch_size=BATCH_SIZE,
                             shuffle=True)

# print(train_data.train_data.size())
# print(train_data.test_labels.size())
# plt.imshow(train_data.train_data[2].numpy(),cmap='gray')
# plt.title('%i'%train_data.test_labels[2])
# plt.show()

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder=nn.Sequential(
            nn.Linear(28*28,128),
            nn.Tanh(),
            nn.Linear(128,64),
            nn.Tanh(),
            nn.Linear(64, 12),
            nn.Tanh(),
            nn.Linear(12, 3),
        )
        self.decoder=nn.Sequential(
            nn.Linear(3, 12),
            nn.Tanh(),
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 28*28),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded=self.encoder(x)
        decoded=self.decoder(encoded)
        return encoded,decoded

autoencoder=AutoEncoder()
optimizer=torch.optim.Adam(autoencoder.parameters(),lr=LR)
loss_func=nn.MSELoss()

f,a=plt.subplots(2,N_TEST_IMG,figsize=(5,2))


view_data=train_data.test_data[:N_TEST_IMG].view(-1,28*28).type(torch.FloatTensor)
for i in range(N_TEST_IMG):
    a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)), cmap='gray');
    a[0][i].set_xticks(());
    a[0][i].set_yticks(());

for epoch in range(EPOCH):
    for step,(x,y) in enumerate(train_loader):
        b_x=x.view(-1,28*28)
        b_y=x.view(-1,28*28)


        encoded,decoded=autoencoder(b_x)

        loss=loss_func(decoded,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step%100==0:
            print("Epoch: ",epoch,"| train loss: %.4f"%loss.data.numpy())
            _,decoded_data = autoencoder(view_data)

            for i in range(N_TEST_IMG):
                a[1][i].clear()
                a[1][i].imshow(np.reshape(decoded_data.data.numpy()[i], (28, 28)), cmap='gray')
                a[1][i].set_xticks(());
                a[1][i].set_yticks(());
            plt.draw();
            plt.pause(0.00005)


plt.show()