from src.models.base_deepjscc import BaseDeepJSCC


class DeepJSCC(BaseDeepJSCC):
    def __init__(self, encoder, decoder, loss, **kwargs):
        super().__init__(encoder=encoder, decoder=decoder, loss=loss, **kwargs)
        
        if self.csi.channel == "rayleigh":
            num_af_dim = 3
        else:
            num_af_dim = 1
        
        self.encoder = encoder(num_af_dim=num_af_dim)
        self.decoder = decoder(num_af_dim=num_af_dim)

    def step(self, batch):
        x, csi = batch
        x_hat = self.forward((x, csi))

        loss = self.loss(x_hat, x)

        return loss, x_hat, x

    def forward(self, batch):
        x, csi = batch

        x = self.encoder((x, csi))

        x = self.power_constraint(x)

        x = self.channel((x, csi))

        x = self.decoder((x, csi))

        return x
