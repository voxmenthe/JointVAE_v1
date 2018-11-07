import torch
from torch import nn, optim
from torch.nn import functional as F

EPS = 1e-12

# class Encoder_x(nn.Module):
#     def __init__(self, channels = 3, hidden_dim=256):
#         self.channels = channels
#         self.hidden_dim = hidden_dim

#         super(Encoder_x).__init__()

#         # Define all the layers
#         # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
#         self.conv1 = nn.Conv2d(self.channels, 32, (4, 4), stride=2, padding=1)
#         self.conv2 = nn.Conv2d(32, 32, (4, 4), stride=2, padding=1) # extra layer for 64
#         self.conv3 = nn.Conv2d(32, 64, (4, 4), stride=2, padding=1)
#         self.conv4 = nn.Conv2d(64, 64, (4, 4), stride=2, padding=1)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = F.relu(self.conv4(x))

#         return x

# class Decoder_x(nn.Module):
#     def __init__(self, channels = 3):
#         self.channels = channels

#         super(Decoder_x).__init__

#         # Define all the layers
#         self.convt1 = nn.ConvTranspose2d(64, 64, (4, 4), stride=2, padding=1)
#         self.convt2 = nn.ConvTranspose2d(64, 32, (4, 4), stride=2, padding=1)
#         self.convt3 = nn.ConvTranspose2d(32, 32, (4, 4), stride=2, padding=1)
#         self.convt4 = nn.ConvTranspose2d(32, channels, (4, 4), stride=2, padding=1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self,input):
#         x = F.relu(self.convt1(input))
#         x = F.relu(self.convt2(x))
#         x = F.relu(self.convt3(x))
#         x = self.sigmoid(self.convt4(x))

#         return x

class VAE(nn.Module):
    def __init__(self, img_size, latent_spec, temperature=.67, use_cuda=False):
        """
        Class which defines model and forward pass.

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).

        latent_spec : dict
            Specifies latent distribution. For example:
            {'cont': 10, 'disc': [10, 4, 3]} encodes 10 normal variables and
            3 gumbel softmax variables of dimension 10, 4 and 3. A latent spec
            can include both 'cont' and 'disc' or only 'cont' or only 'disc'.

        temperature : float
            Temperature for gumbel softmax distribution.

        use_cuda : bool
            If True moves model to GPU
        """
        super(VAE, self).__init__()
        self.use_cuda = use_cuda
        #self.img_to_features = encoder
        #self.features_to_image = decoder

        # Parameters
        self.img_size = img_size
        self.is_continuous = 'cont' in latent_spec
        self.is_discrete = 'disc' in latent_spec
        self.latent_spec = latent_spec
        self.num_pixels = img_size[1] * img_size[2]
        self.temperature = temperature
        self.hidden_dim = 256  # Hidden dimension of linear layer
        self.reshape = (64, 4, 4)  # Shape required to start transpose convs

        # Calculate dimensions of latent distribution
        self.latent_cont_dim = 0
        self.latent_disc_dim = 0
        self.num_disc_latents = 0
        if self.is_continuous:
            self.latent_cont_dim = self.latent_spec['cont']
        if self.is_discrete:
            self.latent_disc_dim += sum([dim for dim in self.latent_spec['disc']])
            self.num_disc_latents = len(self.latent_spec['disc'])
        self.latent_dim = self.latent_cont_dim + self.latent_disc_dim

        # Encoder operations
        self.conv1 = nn.Conv2d(self.img_size[0], 32, (4, 4), stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, (4, 4), stride=2, padding=1) # extra layer for 64
        self.conv3 = nn.Conv2d(32, 64, (4, 4), stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, (4, 4), stride=2, padding=1)

        # Decoder operations
        self.convt1 = nn.ConvTranspose2d(64, 64, (4, 4), stride=2, padding=1)
        self.convt2 = nn.ConvTranspose2d(64, 32, (4, 4), stride=2, padding=1)
        self.convt3 = nn.ConvTranspose2d(32, 32, (4, 4), stride=2, padding=1)
        self.convt4 = nn.ConvTranspose2d(32, self.img_size[0], (4, 4), stride=2, padding=1)

        # Map encoded features into a hidden vector which will be used to
        # encode parameters of the latent distribution
        self.features_to_hidden = nn.Sequential(
            nn.Linear(64 * 4 * 4, self.hidden_dim),
            nn.ReLU()
        )
        
        # Encode parameters of latent distribution
        if self.is_continuous:
            self.fc_mean = nn.Linear(self.hidden_dim, self.latent_cont_dim)
            self.fc_log_var = nn.Linear(self.hidden_dim, self.latent_cont_dim)
        if self.is_discrete:
            # Linear layer for each of the categorical distributions
            fc_alphas = []
            for disc_dim in self.latent_spec['disc']:
                fc_alphas.append(nn.Linear(self.hidden_dim, disc_dim))
            self.fc_alphas = nn.ModuleList(fc_alphas)

        # Map latent samples to features to be used by generative model
        self.latent_to_features = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 64 * 4 * 4),
            nn.ReLU()
        )

    def encode(self, x):
        """
        Encodes an image into parameters of a latent distribution defined in
        self.latent_spec.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data, shape (N, C, H, W)
        """
        batch_size = x.size()[0]

        # Encode image to hidden features
        #features = self.img_to_features(x)

        # define our img_to_features encoder calculations
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        features = F.relu(self.conv4(x))

        print("features shape: ", features.shape)
        print("features view shape: ", features.view(batch_size, -1).shape)
        hidden = self.features_to_hidden(features.view(batch_size, -1))

        # Output parameters of latent distribution from hidden representation
        latent_dist = {}

        if self.is_continuous:
            latent_dist['cont'] = [self.fc_mean(hidden), self.fc_log_var(hidden)]

        if self.is_discrete:
            latent_dist['disc'] = []
            for fc_alpha in self.fc_alphas:
                latent_dist['disc'].append(F.softmax(fc_alpha(hidden), dim=1))

        return latent_dist

    def reparameterize(self, latent_dist):
        """
        Samples from latent distribution using the reparameterization trick.

        Parameters
        ----------
        latent_dist : dict
            Dict with keys 'cont' or 'disc' or both, containing the parameters
            of the latent distributions as torch.Tensor instances.
        """
        latent_sample = []

        if self.is_continuous:
            mean, logvar = latent_dist['cont']
            cont_sample = self.sample_normal(mean, logvar)
            latent_sample.append(cont_sample)

        if self.is_discrete:
            for alpha in latent_dist['disc']:
                disc_sample = self.sample_gumbel_softmax(alpha)
                latent_sample.append(disc_sample)

        # Concatenate continuous and discrete samples into one large sample
        return torch.cat(latent_sample, dim=1)

    def sample_normal(self, mean, logvar):
        """
        Samples from a normal distribution using the reparameterization trick.

        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (N, D) where D is dimension
            of distribution.

        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (N, D)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.zeros(std.size()).normal_()
            if self.use_cuda:
                eps = eps.cuda()
            return mean + std * eps
        else:
            # Reconstruction mode
            return mean

    def sample_gumbel_softmax(self, alpha):
        """
        Samples from a gumbel-softmax distribution using the reparameterization
        trick.

        Parameters
        ----------
        alpha : torch.Tensor
            Parameters of the gumbel-softmax distribution. Shape (N, D)
        """
        if self.training:
            # Sample from gumbel distribution
            unif = torch.rand(alpha.size())
            if self.use_cuda:
                unif = unif.cuda()
            gumbel = -torch.log(-torch.log(unif + EPS) + EPS)
            # Reparameterize to create gumbel softmax sample
            log_alpha = torch.log(alpha + EPS)
            logit = (log_alpha + gumbel) / self.temperature
            return F.softmax(logit, dim=1)
        else:
            # In reconstruction mode, pick most likely sample
            _, max_alpha = torch.max(alpha, dim=1)
            one_hot_samples = torch.zeros(alpha.size())
            # On axis 1 of one_hot_samples, scatter the value 1 at indices
            # max_alpha. Note the view is because scatter_ only accepts 2D
            # tensors.
            one_hot_samples.scatter_(1, max_alpha.view(-1, 1).data.cpu(), 1)
            if self.use_cuda:
                one_hot_samples = one_hot_samples.cuda()
            return one_hot_samples

    def decode(self, latent_sample):
        """
        Decodes sample from latent distribution into an image.

        Parameters
        ----------
        latent_sample : torch.Tensor
            Sample from latent distribution. Shape (N, L) where L is dimension
            of latent distribution.
        """
        features = self.latent_to_features(latent_sample)
        #return self.features_to_img(features.view(-1, *self.reshape))

        # features_to_image  calculations
        x = features.view(-1, *self.reshape)
        x = F.relu(self.convt1(x))
        x = F.relu(self.convt2(x))
        x = F.relu(self.convt3(x))
        x = self.convt4(x)
        print('decoded shape before sigmoid: ', x.shape)
        return torch.sigmoid(x)
        
    def forward(self, x):
        """
        Forward pass of model.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (N, C, H, W)
        """
        latent_dist = self.encode(x)
        print("latent dist keys in decode: ", latent_dist.keys())
        latent_sample = self.reparameterize(latent_dist)
        print("latent sample shape in decode: ", latent_sample.shape)
        return self.decode(latent_sample), latent_dist
