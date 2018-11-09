        # Define encoder layers
        # Intial layer
        # nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        encoder_layers = [
            nn.Conv2d(self.img_size[0], 32, (4, 4), stride=2, padding=1),
            nn.ReLU()
        ]
        # Add additional layer if (64, 64) images
        if self.img_size[1:] == (64, 64):
            encoder_layers += [
                nn.Conv2d(32, 32, (4, 4), stride=2, padding=1),
                nn.ReLU()
            ]
        else:
            raise RuntimeError("Only (None, 64, 64) supported at the moment.")

        # Add final layers
        encoder_layers += [
            nn.Conv2d(32, 64, (4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, (4, 4), stride=2, padding=1),
            nn.ReLU()
        ]

        # Define encoder
        self.img_to_features = nn.Sequential(*encoder_layers)

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