import torch
from torch.autograd import Variable


class RecurrentNet(torch.nn.Module):
  def __init__(self, D_in, H, D_out):
    super(RecurrentNet, self).__init__()
    self.linear1 = torch.nn.Linear(D_in, H)
    self.recurrent_linear = torch.nn.Linear(H, H)
    self.linear2 = torch.nn.Linear(H, D_out)

  def forward(self, x):
    h_relu = self.linear1(x).clamp(min=0)
    for _ in range(3):
      h_relu = self.recurrent_linear(h_relu).clamp(min=0)
    y_pred = self.linear2(h_relu)
    return y_pred


N, D_in, H, D_out = 64, 1000, 100, 10

x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)

model = RecurrentNet(D_in, H, D_out)
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
for t in range(500):
  y_pred = model(x)
  loss = criterion(y_pred, y)
  print(t, loss.data[0])

  model.zero_grad()
  loss.backward()
  optimizer.step()

