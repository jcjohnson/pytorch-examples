import torch
from torch.autograd import Variable

N, D_in, H, D_out = 64, 1000, 100, 10

x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out))

w1 = Variable(torch.randn(D_in, H), requires_grad=True)
w2 = Variable(torch.randn(H, D_out), requires_grad=True)

learning_rate = 1e-6
for t in range(500):
  y_pred = x.mm(w1).clamp(min=0).mm(w2)
  loss = (y_pred - y).pow(2).sum()

  w1.grad.data.zero_()
  w2.grad.data.zero_()
  loss.backward()

  print(t, loss.data[0])
  w1.data -= learning_rate * w1.grad.data
  w2.data -= learning_rate * w2.grad.data
