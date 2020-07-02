import torch as t

K = 10
class _IntRound(t.autograd.Function):

  @staticmethod
  def forward(ctx,x):
    ctx.save_for_backward(x)
    return t.round(x)

  @staticmethod
  def backward(ctx, dy):
    x = ctx.saved_tensors
    rx = t.round(x)
    delta = t.abs(x-rx)
    return (K**(4*delta-1)) * dy

IntRound = _IntRound.apply
