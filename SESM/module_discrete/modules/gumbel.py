from torch import where, rand_like, zeros_like, log, sigmoid
from torch.nn import Module


class GumbelSigmoid(Module):
    def forward(self, x, t=0.1, eps=1e-20, hard=True):
        assert t != 0, "temperature must not 0"

        if not self.training:
            return where(x > 0, 1.0, 0.0)

        uniform1 = rand_like(x)
        uniform2 = rand_like(x)

        noise = log(log(uniform2 + eps) / log(uniform1 + eps) + eps)
        y = sigmoid((x + noise) / t)

        if not hard:
            return y

        y_hard = zeros_like(y)
        y_hard[y > 0] = 1
        y_hard = (y_hard - y).detach() + y
        return y_hard
