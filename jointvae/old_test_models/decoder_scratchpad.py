
        # Define decoder
        decoder_layers = []

        # Additional decoding layer for (64, 64) images
        if self.img_size[1:] == (64, 64):
            decoder_layers += [
                nn.ConvTranspose2d(64, 64, (4, 4), stride=2, padding=1),
                nn.ReLU()
            ]

        decoder_layers += [
            nn.ConvTranspose2d(64, 32, (4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, (4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, self.img_size[0], (4, 4), stride=2, padding=1),
            nn.Sigmoid()
        ]

        # Define decoder
        self.features_to_img = nn.Sequential(*decoder_layers)

