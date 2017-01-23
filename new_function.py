import torch
from torch.autograd import Variable


class MulTwo(torch.autograd.Function):
  def forward(self, input):
    return 2.0 * input

  def backward(self, grad_output):
    return 2.0 * grad_output


x = Variable(torch.randn(3, 4), requires_grad=True)
y = MulTwo()(x)
s = y.sum()

s.backward()
print(x.grad.data)

