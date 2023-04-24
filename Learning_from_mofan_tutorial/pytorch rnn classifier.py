import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms

EPOCH = 1
BATCH_SIZE = 64
TIME_STEP=28
INPUT_SIZE=28
LR = 0.001
DOWNLOAD_MNIST = True

train_data =dsets.MNIST(
    root='./mnist',
    train=True,
    transform=transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)

train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)

test_data = dsets.MNIST(root='./mnist/',
                        train=False,
                        transform=transforms.ToTensor())
test_x=test_data.test_data.type(torch.FloatTensor)[:2000]/255
test_y=test_data.test_labels.numpy()[:2000]

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn=nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=1,
            batch_first=True,

        )
        self.out = nn.Linear(64, 10)   # fully connected layer, output 10 classes
    def forward(self, x):
        r_out,(h_n,h_c)=self.rnn(x,None)
        out=self.out(r_out[:,-1,:])
        return out # return x for visualization


rnn=RNN()
# print(rnn)
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step,(b_x,b_y) in enumerate(train_loader):
        b_x=b_x.view(-1,28,28)

        output=rnn(b_x)
        loss=loss_func(output,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            test_output = rnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y==test_y).astype(int).sum())/float(test_y.size)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
            epoch += 1


    # print 10 predictions from test data
test_output = rnn(test_x[:10].view(-1,28,28))
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y[:10], 'real number')