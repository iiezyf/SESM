from torch.nn import Module


class PoolingAllOut(Module):
    def forward(self, x):
        out, _ = x
        return out.mean(1)


class RNNLastOut(Module):
    def forward(self, x):
        out, _ = x
        return out[:, -1, :]


class RNNAllOut(Module):
    def forward(self, x):
        out, _ = x
        return out


class RNNHidden(Module):
    def __init__(self, rnn_type="lstm"):
        super().__init__()
        self.rnn_type = rnn_type

    def forward(self, x):
        if self.rnn_type == "lstm":
            out, (h, c) = x
        else:
            out, h = x
        return h.mean(0)
