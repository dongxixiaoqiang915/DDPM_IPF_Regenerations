import torch
import torch.nn as nn
import torch.nn.functional as F


class EMA:
    def __init__(self, decay_rate):
        super().__init__()
        self.decay_rate = decay_rate
        self.step_count = 0

    def update_model_average(self, avg_model, current_model):
        # Iterate over parameters from the current and the average model to update the exponential moving average.
        for current_param, avg_param in zip(current_model.parameters(), avg_model.parameters()):
            previous_weight, new_weight = avg_param.data, current_param.data
            avg_param.data = self.update_average(previous_weight, new_weight)

    def update_average(self, previous, current):
        # Compute the updated average using the decay rate.
        if previous is None:
            return current
        return previous * self.decay_rate + (1 - self.decay_rate) * current

    def step_ema(self, ema_model, model, start_step=2000):
        # If we haven't reached the designated step, reset the EMA model parameters.
        if self.step_count < start_step:
            self.reset_parameters(ema_model, model)
            self.step_count += 1
            return
        self.update_model_average(ema_model, model)
        self.step_count += 1

    def reset_parameters(self, ema_model, model):
        # Synchronize the EMA model's parameters with those of the current model.
        ema_model.load_state_dict(model.state_dict())


"""
In practical terms, integrating self-attention within U-Net enables the model to concentrate on important regions across an image.
This is particularly useful for tasks such as segmentation or reconstruction, as it leverages contextual relationships—crucial in domains like medical or satellite imaging.
"""

class SelfAttention(nn.Module):
    def __init__(self, num_channels, spatial_dim):
        super(SelfAttention, self).__init__()
        self.num_channels = num_channels  # Number of channels in the input feature map.
        self.spatial_dim = spatial_dim    # Height/width dimension of the feature map.
        # Multi-head attention layer (with 4 heads) to capture features from various subspaces.
        self.mha = nn.MultiheadAttention(num_channels, 4, batch_first=True)
        self.layer_norm = nn.LayerNorm([num_channels])  # Normalize features to stabilize learning.
        # Feed-forward network to further process the attention outputs.
        self.feed_forward = nn.Sequential(
            nn.LayerNorm([num_channels]),
            nn.Linear(num_channels, num_channels),
            nn.GELU(),
            nn.Linear(num_channels, num_channels),
        )

    def forward(self, x):
        """
        Reshape and permute the input tensor to match MultiheadAttention's expected format.
        Input x shape: [batch_size, channels, height, width] → reshaped to [batch_size, height*width, channels].
        """
        x = x.view(-1, self.num_channels, self.spatial_dim * self.spatial_dim).swapaxes(1, 2)
        normalized = self.layer_norm(x)
        # Apply self-attention where the normalized tensor is used as query, key, and value.
        attn_output, _ = self.mha(normalized, normalized, normalized)
        attn_output = attn_output + x
        # Process through the feed-forward network with a residual connection.
        attn_output = self.feed_forward(attn_output) + attn_output
        # Restore the original tensor shape: [batch_size, channels, height, width].
        return attn_output.swapaxes(2, 1).view(-1, self.num_channels, self.spatial_dim, self.spatial_dim)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, use_residual=False):
        super().__init__()
        self.use_residual = use_residual
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.use_residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.down_block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, use_residual=True),
            DoubleConv(in_channels, out_channels),
        )
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, time_embedding):
        x = self.down_block(x)
        emb = self.emb_layer(time_embedding)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_block = nn.Sequential(
            DoubleConv(in_channels, in_channels, use_residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, skip_connection, time_embedding):
        x = self.upsample(x)
        x = torch.cat([skip_connection, x], dim=1)
        x = self.conv_block(x)
        emb = self.emb_layer(time_embedding)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels))
        pos_sin = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_cos = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_encoding = torch.cat([pos_sin, pos_cos], dim=-1)
        return pos_encoding

    def forward(self, x, t):
        t = t.unsqueeze(-1).float()
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output



class UNet_conditional(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, num_classes=None, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        if num_classes is not None:
            self.label_embedding = nn.Embedding(num_classes, time_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels))
        pos_sin = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_cos = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_encoding = torch.cat([pos_sin, pos_cos], dim=-1)
        return pos_encoding

    def forward(self, x, t, y):
        t = t.unsqueeze(-1).float()
        t = self.pos_encoding(t, self.time_dim)
        
        # Incorporate label embedding into the time-step representation if available.
        if y is not None:
            t += self.label_embedding(y)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output


if __name__ == '__main__':
    # net = UNet(device="cpu")
    net = UNet_conditional(num_classes=10, device="cpu")
    total_parameters = sum(p.numel() for p in net.parameters())
    print(total_parameters)
    sample_input = torch.randn(3, 3, 64, 64)
    time_steps = sample_input.new_tensor([500] * sample_input.shape[0]).long()
    labels = sample_input.new_tensor([1] * sample_input.shape[0]).long()
    print(net(sample_input, time_steps, labels).shape)