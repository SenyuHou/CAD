import torch.nn as nn
import torch.nn.functional as F
from utils.diffusion_utils import *
import utils.ResNet_for_32 as resnet_s
import utils.ResNet_for_224 as resnet_l

class ConditionalLinear(nn.Module):
    """
    A conditional linear layer that combines input features with embedded time steps.
    """
    def __init__(self, num_in, num_out, n_steps):
        """
        Initialize the ConditionalLinear layer.

        Parameters:
        - num_in: Number of input features.
        - num_out: Number of output features.
        - n_steps: Number of time steps for embedding.
        """
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, t):
        """
        Forward pass for the ConditionalLinear layer.

        Parameters:
        - x: Input features.
        - t: Time step.

        Returns:
        - out: Output features combined with embedded time steps.
        """
        out = self.lin(x)
        gamma = self.embed(t)
        out = gamma.view(-1, self.num_out) * out
        return out

class ConditionalModel(nn.Module):
    """
    A conditional model that uses a U-Net structure with conditional linear layers.
    """
    def __init__(self, n_steps, y_dim=20, fp_dim=128, feature_dim=None, guidance=True):
        """
        Initialize the ConditionalModel.

        Parameters:
        - n_steps: Number of time steps for embedding.
        - y_dim: Dimension of the labels.
        - fp_dim: Dimension of the feature embeddings.
        - feature_dim: Dimension of the features.
        - guidance: Whether to use guidance.
        """
        super(ConditionalModel, self).__init__()
        n_steps = n_steps + 1
        self.y_dim = y_dim
        self.guidance = guidance
        self.norm = nn.BatchNorm1d(feature_dim)

        # U-Net structure
        if self.guidance:
            self.lin1 = ConditionalLinear(y_dim + fp_dim, feature_dim, n_steps)
        else:
            self.lin1 = ConditionalLinear(y_dim, feature_dim, n_steps)

        self.unetnorm1 = nn.BatchNorm1d(feature_dim)
        self.lin2 = ConditionalLinear(feature_dim, feature_dim, n_steps)
        self.unetnorm2 = nn.BatchNorm1d(feature_dim)
        self.lin3 = ConditionalLinear(feature_dim, feature_dim, n_steps)
        self.unetnorm3 = nn.BatchNorm1d(feature_dim)
        self.lin4 = nn.Linear(feature_dim, y_dim)

    def forward(self, x_embed, y, t, fp_x=None):
        """
        Forward pass for the ConditionalModel.

        Parameters:
        - x_embed: Resnet Embedded input features.
        - y: Labels.
        - t: Time step.
        - fp_x: Optional feature embeddings.

        Returns:
        - The output of the model.
        """
        x_embed = self.norm(x_embed)

        if self.guidance:
            y = torch.cat([y, fp_x], dim=-1)

        y = self.lin1(y, t)
        y = self.unetnorm1(y)
        y = F.softplus(y)
        y = x_embed * y
        y = self.lin2(y, t)
        y = self.unetnorm2(y)
        y = F.softplus(y)
        y = self.lin3(y, t)
        y = self.unetnorm3(y)
        y = F.softplus(y)
        return self.lin4(y)

class Diffusion(nn.Module):
    """
    A diffusion model that uses a conditional model and a feature encoder.
    """
    def __init__(self, fp_encoder, num_timesteps=1000, n_class=10, fp_dim=512, device='cuda', beta_schedule='cosine',
                 feature_dim=2048, encoder_type='resnet50_l', ddim_num_steps=10):
        """
        Initialize the Diffusion model.

        Parameters:
        - fp_encoder: The feature encoder.
        - num_timesteps: Number of time steps for the diffusion process.
        - n_class: Number of classes.
        - fp_dim: Dimension of the feature embeddings.
        - device: The device to run the model on.
        - beta_schedule: The beta schedule for the diffusion process.
        - feature_dim: Dimension of the features.
        - encoder_type: The type of encoder to use.
        - ddim_num_steps: Number of DDIM steps.
        """
        super().__init__()
        self.device = device
        self.num_timesteps = num_timesteps
        self.n_class = n_class
        betas = make_beta_schedule(schedule=beta_schedule, num_timesteps=self.num_timesteps, start=0.0001, end=0.02)
        betas = self.betas = betas.float().to(self.device)
        self.betas_sqrt = torch.sqrt(betas)
        alphas = 1.0 - betas
        self.alphas = alphas
        self.one_minus_betas_sqrt = torch.sqrt(alphas)
        self.alphas_cumprod = alphas.cumprod(dim=0)
        self.alphas_bar_sqrt = torch.sqrt(self.alphas_cumprod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_cumprod)
        alphas_cumprod_prev = torch.cat([torch.ones(1).to(self.device), self.alphas_cumprod[:-1]], dim=0)
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.posterior_mean_coeff_1 = (betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coeff_2 = (torch.sqrt(alphas) * (1 - alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        posterior_variance = (betas * (1.0 - alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_variance = posterior_variance
        self.logvar = betas.log()
        self.fp_dim = fp_dim

        self.fp_encoder = fp_encoder.eval()
        self.encoder_type = encoder_type
        if encoder_type == 'resnet34':
            self.diffusion_encoder = resnet_s.resnet34(num_input_channels=3, num_classes=feature_dim).to(self.device)
        elif encoder_type == 'resnet18':
            self.diffusion_encoder = resnet_s.resnet18(num_input_channels=3, num_classes=feature_dim).to(self.device)
        elif encoder_type == 'resnet50':
            self.diffusion_encoder = resnet_s.resnet50(num_input_channels=3, num_classes=feature_dim).to(self.device)
        elif encoder_type == 'resnet18_l':
            self.diffusion_encoder = resnet_l.resnet18(num_classes=feature_dim, pretrained=True).to(self.device)
        elif encoder_type == 'resnet34_l':
            self.diffusion_encoder = resnet_l.resnet34(num_classes=feature_dim, pretrained=True).to(self.device)
        elif encoder_type == 'resnet50_l':
            self.diffusion_encoder = resnet_l.resnet50(num_classes=feature_dim, pretrained=True).to(self.device)
        elif encoder_type == 'resnet101_l':
            self.diffusion_encoder = resnet_l.resnet101(num_classes=feature_dim, pretrained=True).to(self.device)    
        else:
            raise Exception("ResNet type should be one of [18, 34, 50]")

        self.model = ConditionalModel(self.num_timesteps, y_dim=self.n_class, fp_dim=fp_dim,
                                      feature_dim=feature_dim, guidance=True).to(self.device)

        self.ddim_num_steps = ddim_num_steps
        self.make_ddim_schedule(ddim_num_steps)

    def make_ddim_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0.):
        """
        Make the DDIM schedule for the diffusion process.

        Parameters:
        - ddim_num_steps: Number of DDIM steps.
        - ddim_discretize: The discretization method for DDIM.
        - ddim_eta: The noise scale for DDIM.
        """
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.num_timesteps)

        assert self.alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.device)

        self.register_buffer('sqrt_alphas_cumprod', to_torch(torch.sqrt(self.alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(torch.sqrt(1. - self.alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(torch.log(1. - self.alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(torch.sqrt(1. / self.alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(torch.sqrt(1. / self.alphas_cumprod - 1)))

        # DDIM sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=self.alphas_cumprod,
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', torch.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    def load_diffusion_net(self, net_state_dicts):
        """
        Load the state dictionaries for the diffusion model components.

        Parameters:
        - net_state_dicts: List of state dictionaries for the model, encoder, and optional feature encoder.
        """
        self.model.load_state_dict(net_state_dicts[0])
        self.diffusion_encoder.load_state_dict(net_state_dicts[1])

        if len(net_state_dicts) == 3:
            self.fp_encoder.load_state_dict(net_state_dicts[2])

    def forward_t(self, y_0_batch, x_batch, t, fp_x):

        x_batch = x_batch.to(self.device)

        e = torch.randn_like(y_0_batch).to(y_0_batch.device)
        y_t_batch = q_sample(y_0_batch, self.alphas_bar_sqrt,
                             self.one_minus_alphas_bar_sqrt, t, noise=e)

        x_embed_batch = self.diffusion_encoder(x_batch)
        output = self.model(x_embed_batch, y_t_batch, t, fp_x)

        return output, e

    def reverse(self, images, only_last_sample=True, stochastic=True, fp_x=None, fq_x=None):
        """
        Reverse the diffusion process to generate samples.

        Parameters:
        - images: The input images.
        - only_last_sample: Whether to return only the last sample.
        - stochastic: Whether to use stochastic sampling.
        - fp_x: Optional feature embeddings.
        - fq_x: Optional additional feature embeddings.

        Returns:
        - label_t_0: The generated labels.
        """
        images = images.to(self.device)
        with torch.no_grad():

            if fp_x is None:
                fp_x = self.fp_encoder(images)

            label_t_0 = p_sample_loop(self.model, images, fp_x,
                                      self.num_timesteps, self.alphas,
                                      self.one_minus_alphas_bar_sqrt,
                                      only_last_sample=only_last_sample, stochastic=stochastic, fq_x=fq_x)

        return label_t_0

    def reverse_ddim(self, x_batch, stochastic=True, fp_x=None, fq_x=None, only_last_sample=True):
        """
        Reverse the diffusion process using DDIM to generate samples.

        Parameters:
        - x_batch: The input images.
        - stochastic: Whether to use stochastic sampling.
        - fp_x: Optional feature embeddings.
        - fq_x: Optional additional feature embeddings.

        Returns:
        - label_t_0: The generated labels.
        """
        x_batch = x_batch.to(self.device)
        with torch.no_grad():

            if fp_x is None:
                fp_x = self.fp_encoder(x_batch)

            x_embed_batch = self.diffusion_encoder(x_batch)
            label_t_0 = ddim_sample_loop(self.model, x_embed_batch, fp_x, self.ddim_timesteps, self.n_class, self.ddim_alphas,
                                         self.ddim_alphas_prev, self.ddim_sigmas, stochastic=stochastic,
                                         fq_x=fq_x, only_last_sample=only_last_sample)

        return label_t_0
