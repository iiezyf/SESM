import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsnooper


class Prototypes(nn.Module):
    """
    The 'Prototypes Layer' as a tf.keras Layer.
    """

    def __init__(self, K, D, dmin=1.0, Ld=0.01, Lc=0.01, Le=0.1, **kwargs):
        """
        Parameters
        ----------
        k : int
            Number of prototype vectors to create.
        dmin : float, optional
            Threshold to determine whether two prototypes are close, default=1.0.
            For "diversity" regularization. See paper section 3.2 for details.
        Ld : float, optional
            Weight for "diversity" regularization loss, default=0.01.
        Lc : float, optional
            Weight for "clustering" regularization loss, default=0.01.
        Le : float, optional
            Weight for "evidence" regularization loss, default=0.1.
        **kwargs
            Additional arguments for base `Layer` constructor (name, etc.)
        """
        super(Prototypes, self).__init__(**kwargs)
        self.K = K
        self.D = D
        self.dmin = dmin
        self.Ld, self.Lc, self.Le = Ld, Lc, Le

        # Makes sense to use same `initializer` as LSTM ?
        self.prototypes = nn.Parameter(torch.empty((1, K, D)))
        nn.init.kaiming_uniform_(self.prototypes)
        # print(self.prototypes)

    # @property
    # def prototypes(self):
    #     self.__prototypes = self.__prototypes.clamp(-1.0, 1.0)
    #     return self.__prototypes

    # @torchsnooper.snoop()
    def forward(self, x, training=None):
        """Forward pass."""

        # L2 distances b/t encodings and prototypes
        x = x.unsqueeze(-2)
        d2 = torch.norm(x - self.prototypes.to(x.device), p=2, dim=-1)
        # print(x.shape, self.prototypes.shape, d2.shape)
        # Losses only computed `if training`
        if training:
            dLoss = self._diversity_term()
            cLoss = d2.min(dim=0)[0].sum()
            eLoss = d2.min(dim=1)[0].sum()
        else:
            dLoss, cLoss, eLoss = 0.0, 0.0, 0.0

        # self.add_loss(dLoss)
        # self.add_loss(cLoss, inputs=True)
        # self.add_loss(eLoss, inputs=True)

        # Return exponentially squashed distances
        return torch.exp(-d2), (dLoss, cLoss, eLoss)

    def _diversity_term(self):
        """Compute the "diversity" loss,
        which penalizes prototypes that are close to each other

        NOTE: Computes full distance matrix, which is redudant, but `prototypes`
              is usually a small-ish tensor and performance is acceptable,
              so I'm not going to worry about it.
        """
        dist = torch.cdist(self.prototypes, self.prototypes)

        Rd = F.relu(-dist + self.dmin)

        # Zero the diagonal elements
        zero_diag = torch.ones_like(Rd) - torch.eye(self.K).to(Rd.device)

        return ((Rd * zero_diag) ** 2).sum() / 2.0
