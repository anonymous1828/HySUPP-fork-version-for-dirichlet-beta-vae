"""
DirVAE PyTorch implementation
"""

import logging
import time

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from src.model.blind.base import BlindUnmixingModel
from src.utils.constraint import NonNegConstraint
from src.utils.tf_rms_prop import RMSpropTF
from src.utils.metrics import GammaKL

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DirVAE(nn.Module, BlindUnmixingModel):
    """
        Based on 
        "Dirichlet Variational Autoencoder"
        of Weonyoung Joo, Wonsung Lee, Sungrae Park, Il-Chul Moon
    """

    def __init__(
        self,
        epochs = 300,
        lr = 1e-3,
        batch_size = 200,
        beta: float = 1.,
        hidden_dims: list = None,
        encoder_activation_fn: callable = nn.LeakyReLU(negative_slope=0.01),
        encoder_batch_norm: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu",
        )

        self.beta = beta
        self.encoder_batch_norm = encoder_batch_norm
        self.encoder_activation_fn = encoder_activation_fn

        self.one_over_beta = None
        self.hidden_dims = [i for i in reversed(range(1, 4))] + [1]

        self.lr         = lr
        self.batch_size = batch_size
        self.epochs     = epochs

        reg_factor = kwargs.get("reg_factor", None); assert reg_factor is not None
        self.reg_factor = reg_factor

        self.alpha_target = None
        self.n_bands      = None
        self.n_ems        = None
        self.n_samples    = None
        self.encoder      = None
        self.decoder      = None

    def to(self, device):
        super().to(device)
        self.alpha_target = torch.from_numpy(self.alpha_target).to(device)
        self.one_over_beta = self.one_over_beta.to(device)
        return self

    def _build_encoder(self, encoder_activation_fn):

        torch.manual_seed(self.seed)

        encoder = []

        for i, h_dim in enumerate(self.hidden_dims):
            # dense and batch norm layers
            if i == 0:
                encoder.append(nn.Linear(self.n_bands, 
                                         self.n_ems * self.hidden_dims[i]))
                if self.encoder_batch_norm:
                    encoder.append(nn.BatchNorm1d(self.n_ems * self.hidden_dims[i]))
            else:
                encoder.append(nn.Linear(self.n_ems * self.hidden_dims[i - 1], 
                                         self.n_ems * self.hidden_dims[i]))
                if self.encoder_batch_norm:
                    encoder.append(nn.BatchNorm1d(self.n_ems * self.hidden_dims[i]))

            # act fn
            if i < len(self.hidden_dims) - 1:
                if self.encoder_activation_fn is not None:
                    encoder.append(encoder_activation_fn)

        # constrain values to be positive
        encoder.append(nn.Softplus())
        return nn.Sequential(*encoder)


    def _build_decoder(self):

        torch.manual_seed(self.seed)

        decoder = []
        decoder.append(nn.Linear(in_features=self.n_ems,
                       out_features=self.n_bands,
                       bias=False))
        return nn.Sequential(*decoder)


    def init_architecture(
        self,
        n_ems: int,
        n_bands: int,
        n_samples: int,
        seed: int,
        *args,
        **kwargs
    ):
        self.n_ems   = n_ems
        self.n_bands = n_bands
        self.n_samples = n_samples
        # beta is 1 so 1 / beta = 1
        self.one_over_beta = torch.ones((n_ems))

        self.hidden_dims = [i for i in reversed(range(1, 4))] + [1]

        self.seed = seed
        self.encoder = self._build_encoder(self.encoder_activation_fn)
        self.decoder = self._build_decoder()


    def init_decoder(self, 
                     init_mode:str, 
                     Y:torch.Tensor = None):

        if init_mode == "he":
            init_M = torch.nn.init.kaiming_normal_(self.decoder[-1].weight)

        elif init_mode == "random":
            assert Y.shape[0] == self.n_samples and Y.shape[1] == self.n_bands 

            random_indices = np.random.choice(self.n_samples, self.n_ems, replace=False)
            init_M         = torch.tensor(Y[random_indices, :].T)
            
        else:
            raise ValueError(f"Invalid init_mode, got {init_mode}")

        with torch.no_grad():  
            last_decoder_layer = self.decoder[-1]  
            last_decoder_layer.weight.copy_(init_M)


    def reparameterize(self, alphas: torch.Tensor) -> torch.Tensor:
        """
            - u \sim U(0,1)
            - v \sim multigamma(\alpha, \beta \mathbb{1}_K)

            inverse CDF of the multigamma distribution is
            v = CDF^{-1}(u ; \alpha, \beta \mathbb{1}_K) = 
                          \beta^{-1}(u * \alpha * \Gamma(\alpha))^{1/\alpha}
        """
        u = torch.rand_like(alphas)
        
        clamped_alphas = torch.clamp(alphas, max=30) # clamped to avoid NaNs 

        int1 = 1 / torch.max(clamped_alphas, 1e-8 * torch.ones_like(clamped_alphas))
        int2 = clamped_alphas.lgamma()
        int3 = int2.exp()
        int4 = int3 * u + 1e-12 # 1e-12 to avoid NaNs 

        v_latent = self.one_over_beta * (int4 * clamped_alphas) ** int1

        return v_latent


    def process_latent(self, alphas: torch.Tensor, eps=1e-6) -> torch.Tensor:
        """
            Input
            - alpha: parameters of the Dirichlet distribution
            - z_latent \sim Dir(\alpha)
        """
        v_latent = self.reparameterize(alphas)
        sum_v    = torch.sum(v_latent, dim=1, keepdim=True)
        z_latent = v_latent / torch.max(sum_v, torch.ones_like(sum_v) * 1e-8)

        return z_latent
    

    def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, ...]:
        
        alphas   = self.encoder(input)
        z_latent = self.process_latent(alphas)
        output   = self.decoder(z_latent)

        return output, z_latent, alphas


    @staticmethod
    def loss(target, output):
        """
            Data fidelity loss (SAD). 
        """
        assert target.shape == output.shape

        dot_product = (target * output).sum(dim=1)
    
        target_norm = target.norm(dim=1)
        output_norm = output.norm(dim=1)
        norm_factor = target_norm * output_norm

        cos = dot_product / torch.max(norm_factor, 1e-6 * torch.ones_like(norm_factor))        
        clamped_cos = torch.clamp(cos, -1 + 1e-6, 1 - 1e-6)

        sad_score = clamped_cos.acos()
        sad_score_mean = sad_score.mean()

        return sad_score_mean


    def reg_loss(self, alpha_hat):
        """
            Regularization loss 
            -> Dkl between 2 gammas distributions with 
               shared lambda params is Dkl between 
               2 Dirichlet distributions.
        """
        assert self.alpha_target is not None
        reg_loss = GammaKL(self.alpha_target, reduction="mean")
        return reg_loss(alpha_hat)
        

    def compute_endmembers_and_abundances(self, 
                                          Y, 
                                          n_ems, 
                                          seed=0, 
                                          *args, 
                                          **kwargs):

        tic = time.time()
        logger.debug("Solving started...")

        print(f"{self.encoder_batch_norm=}")
        print(f"{self.reg_factor=}")
        print(f"{self.lr=}")

        n_bands, n_samples = Y.shape
        self.alpha_target = np.array([1.0 for i in range(n_ems)]).reshape(1,-1)

        # Dataloader
        dataloader = torch.utils.data.DataLoader(
            torch.from_numpy(Y.T).float(),
            batch_size=self.batch_size,
            shuffle=True,
        )  

        self.init_architecture(n_ems,
                               n_bands,
                               n_samples,
                               seed)
        
        constraint = NonNegConstraint([self.decoder[0]])
        self.init_decoder("he", Y)
        constraint.apply()

        # Send model to GPU
        self = self.to(self.device)
        optimizer = RMSpropTF(self.parameters(), lr=self.lr)

        progress = tqdm(range(self.epochs))
        for ee in progress:

            running_loss = 0.
            run_rec_loss = 0.
            run_reg_loss = 0.

            for batch_idx, targets in enumerate(dataloader):
                
                targets = targets.to(self.device)

                optimizer.zero_grad()

                outputs, z_latent, alphas = self(targets)
        
                loss       = self.loss(targets, outputs)
                reg_loss   = self.reg_loss(alphas)
                total_loss = loss + self.reg_factor * reg_loss
                
                total_loss.backward()
                optimizer.step()

                # constraint on model's weights
                if constraint is not None:
                    constraint.apply()

                run_rec_loss += loss.item()
                run_reg_loss += self.reg_factor * reg_loss.item()
                running_loss += total_loss.item()

            progress.set_postfix_str(f"loss={running_loss:.10e}")

        # Get final abundances and endmembers
        self.eval()

        Y_eval = torch.from_numpy(Y.T).float().to(self.device)

        output, abund, alphas = self(Y_eval)

        Ahat = abund.detach().cpu().numpy()
        Ehat = self.decoder[-1].weight.data.detach().cpu().numpy()

        self.time = time.time() - tic
        logger.info(self.print_time())

        return Ehat, Ahat.T


def check_model():
    from src.data.base import HSI

    hsi = HSI("DC1")
    print(hsi.Y.shape)

    model = DirVAE(epochs = 2,
                   lr = 1e-3,
                   batch_size = 200,
                   beta = 1.,
                   hidden_dims = None,
                   #encoder_activation_fn = nn.LeakyReLU(),
                   reg_factor = 1e-2,
                  )
    model.compute_endmembers_and_abundances(hsi.Y, 
                                            hsi.p, 
                                            seed=42)

if __name__ == "__main__":
    check_model()

