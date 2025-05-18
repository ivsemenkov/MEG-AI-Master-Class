import math

import einops
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F


def cart2sph(sensor_xyz):
    x, y, z = sensor_xyz[:,0], sensor_xyz[:,1], sensor_xyz[:,2]
    xy = np.linalg.norm(sensor_xyz[:,:2], axis=-1)
    r = np.linalg.norm(sensor_xyz, axis=-1)
    theta = np.arctan2(xy, z)
    phi = np.arctan2(y, x)
    return np.stack((r, theta, phi), axis=-1)


class Spatial3DAttentionLayer(nn.Module):
    def __init__(self, n_input, n_output, K, coords_xy, n_dropout, dropout_radius, sensor_path, seed=None):
        super().__init__()

        self.n_input = n_input
        self.n_dropout = n_dropout
        self.dropout_radius = dropout_radius

        coords_xy = torch.tensor(coords_xy, dtype=torch.float32, requires_grad=False)
        self.register_buffer('_coords_xy', coords_xy)

        coords_xyz = np.load(sensor_path + 'coords/sensor_xyz.npy')
        coords_sph = cart2sph(coords_xyz)
        coords_sph = torch.tensor(coords_sph, dtype=torch.float32, requires_grad=False)

        layout = self._create_layout(coords_sph, K - 1).float()
        self.register_buffer('_layout', layout)

        Z = self._create_parameters(n_input, n_output, K, seed)
        self.Z = nn.Parameter(Z)

    def _create_layout(self, coords_sph, L=8):
        n_input = coords_sph.shape[0]

        coords_theta = coords_sph[:, 1]
        coords_phi = coords_sph[:, 2]
        layout = torch.zeros((1, L + 1, L + 1, n_input), dtype=torch.float32, requires_grad=False)

        Plak_values = np.zeros((L + 1, L + 1, n_input))
        for i in range(0, n_input):
            Plak_values[:, :, i] = scipy.special.lpmn(L, L, math.cos(coords_theta[i]))[0]
        Plak_values = torch.tensor(Plak_values, dtype=torch.float64)

        def get_factor(l, m):
            result = math.log(2 * l + 1)
            result -= math.log((2 if m == 0 else 1) * 2 * math.pi)
            result += scipy.special.gammaln(l - abs(m) + 1)
            result -= scipy.special.gammaln(l + abs(m) + 1)
            result /= 2
            result = math.exp(result)
            return result

        counter = -1
        for l in range(0, L + 1):
            for m in range(-l, l + 1):
                counter += 1
                i, j = counter % (L + 1), counter // (L + 1)
                mult_left = get_factor(l, m)
                # mult_left = 1

                if m >= 0:
                    mult = torch.cos(m * coords_phi)
                elif m < 0:
                    mult = torch.sin(- m * coords_phi)
                sela = mult_left * Plak_values[abs(m), l, :] * mult

                layout[:, i, j, :] = sela
        return layout

    def _create_parameters(self, n_input, n_output, K, seed=None):
        if seed is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator = torch.Generator()
        generator.manual_seed(seed)

        Z = torch.randn(size=((n_output, K, K)), generator=generator) * 2 / (n_input + n_output)
        Z = einops.rearrange(Z, 'j k l -> j k l 1')
        return Z

    def to(self, device):
        self._coords_xy = self._coords_xy.to(device)
        self._layout = self._layout.to(device)
        return super().to(device)

    def get_spatial_filter(self):
        A = einops.reduce(self.Z * self._layout, 'j k l i -> j i', 'sum')
        ASoftmax = F.softmax(A, dim=1)
        return ASoftmax.clone().detach()

    def forward(self, x):
        A = einops.reduce(self.Z * self._layout, 'j k l i -> j i', 'sum')
        if self.training and self.n_dropout > 0:
            mask = torch.zeros((1, self.n_input), dtype=A.dtype, device=A.device)
            dropout_location = torch.rand(size=(self.n_dropout, 2), device=A.device) * 0.8 + 0.1
            for k in range(self.n_dropout):
                for i in range(self.n_input):
                    if torch.linalg.norm(self._coords_xy[i] - dropout_location[k]) <= self.dropout_radius:
                        mask[:, i] = - float('inf')
            A = A + mask
        ASoftmax = F.softmax(A, dim=1)
        SAx = torch.einsum('oi, bit -> bot', ASoftmax, x)
        return SAx


class SubjectPlusLayer(nn.Module):
    def __init__(self, n_input, n_output, n_subjects, regularize=True, bias=False, seed=None):
        super().__init__()
        self.bias = bias
        self.regularize = regularize
        if self.regularize:
            self.regularizer = None

        A, b = self._create_parameters(n_input, n_output, n_subjects)
        self.A = nn.Parameter(A)

        I = torch.zeros((1, n_output, n_input), requires_grad=False)
        self.register_buffer('I', I)

        if self.bias:
            self.b = nn.Parameter(b)
            zero = torch.zeros(size=(1, n_output, 1))
            self.register_buffer('zero', zero)

    def _create_parameters(self, n_input, n_output, n_subjects, seed=None):
        A = torch.zeros(size=(n_subjects, n_output, n_input))
        b = torch.zeros(size=(n_subjects, n_output, 1)) if self.bias else None
        with torch.no_grad():
            for subjects in range(n_subjects):
                layer = nn.Conv1d(in_channels=n_input, out_channels=n_output, kernel_size=1)
                A[subjects] = einops.rearrange(layer.weight.data, 'o i 1 -> o i')
                if self.bias:
                    b[subjects] = einops.rearrange(layer.bias.data, 'o -> o 1')
        return A, b

    def _create_regularizer(self, A, b):
        batch_size = A.shape[0]
        reg = torch.norm(A - self.I, p='fro')
        if self.bias:
            reg += torch.norm(b, p='fro')
        reg = reg / batch_size
        return reg

    def get_regularizer(self):
        regularizer = self.regularizer
        self.regularizer = None
        return regularizer

    def forward(self, x, s):
        batch_size = x.shape[0]

        A = torch.cat([self.I, self.A], dim=0)
        s[s >= A.size(0)] = 0
        A_ = A[s, :, :]
        out = torch.einsum('bji, bit -> bjt', A_, x)

        if self.bias:
            b = torch.cat([self.zero, self.b], dim=0)
            b_ = b[s, :, :]
            out = out + b_

        if self.regularize and self.training:
            self.regularizer = self._create_regularizer(A_, b_)

        return out


class ConvBlock(nn.Module):
    def __init__(self, n_input, n_output, block_index):
        super().__init__()

        self.kernel_size = 3
        self.block_index = block_index
        dilation1 = 2 ** (2 * block_index % 5)
        dilation2 = 2 ** ((2 * block_index + 1) % 5)
        dilation3 = 2

        self.conv1 = nn.Conv1d(in_channels=n_input, out_channels=n_output, kernel_size=self.kernel_size,
                               dilation=dilation1, padding='same')
        self.conv2 = nn.Conv1d(in_channels=n_output, out_channels=n_output, kernel_size=self.kernel_size,
                               dilation=dilation2, padding='same')
        self.conv3 = nn.Conv1d(in_channels=n_output, out_channels=2 * n_output, kernel_size=self.kernel_size,
                               dilation=dilation3, padding='same')

        self.batchnorm1 = nn.BatchNorm1d(n_output)
        self.batchnorm2 = nn.BatchNorm1d(n_output)

        self.activation1 = nn.GELU()
        self.activation2 = nn.GELU()
        self.activation3 = nn.GLU(dim=-2)

    def forward(self, x):
        c1x = self.conv1(x)
        res1 = c1x if self.block_index == 0 else x + c1x
        res1 = self.batchnorm1(res1)
        res1 = self.activation1(res1)

        c2x = self.conv2(res1)
        res2 = res1 + c2x
        res2 = self.batchnorm2(res2)
        res2 = self.activation2(res2)

        c3x = self.conv3(res2)
        out = self.activation3(c3x)

        return out


class ConvHead(nn.Module):
    def __init__(self, n_channels, n_features, pool, head_stride):
        super().__init__()

        if pool == 'max':
            self.pool = nn.Sequential(
                nn.MaxPool1d(kernel_size=3, stride=head_stride, padding=0 if head_stride == 2 else 1),
                nn.Conv1d(in_channels=n_channels, out_channels=2 * n_channels, kernel_size=1)
            )
        elif pool == 'conv':
            self.pool = nn.Conv1d(in_channels=n_channels, out_channels=2 * n_channels, kernel_size=3,
                                  stride=head_stride, padding=0 if head_stride == 2 else 1)

        self.conv = nn.Conv1d(in_channels=2 * n_channels, out_channels=n_features, kernel_size=1)
        self.activation = nn.GELU()
        self.batch_norm = nn.BatchNorm1d(n_features)

    def forward(self, x):
        x = self.pool(x)
        x = self.activation(x)
        x = self.conv(x)
        x = self.batch_norm(x)
        return x


class SpatialModule(nn.Module):
    def __init__(
            self,
            n_input,
            n_attention,
            n_unmix,
            use_spatial_attention,
            n_spatial_harmonics,
            coords_xy_scaled,
            spatial_dropout_number,
            spatial_dropout_radius,
            use_unmixing_layer,
            use_unmixing_bias,
            use_subject_layer,
            n_subjects,
            regularize_subject_layer,
            bias_subject_layer,
            sensor_path
    ):
        super().__init__()

        if use_spatial_attention:
            self.self_attention = Spatial3DAttentionLayer(
                n_input=n_input, n_output=n_attention, K=n_spatial_harmonics, coords_xy=coords_xy_scaled, n_dropout=spatial_dropout_number,
                dropout_radius=spatial_dropout_radius, sensor_path=sensor_path, seed=None
            )
        else:
            self.self_attention = None

        if use_unmixing_layer:
            n_attention = n_attention if self.self_attention else n_input
            self.unmixing_layer = nn.Conv1d(in_channels=n_attention, out_channels=n_attention, kernel_size=1, 
                                            bias=use_unmixing_bias)
        else:
            self.unmixing_layer = None

        if use_subject_layer:
            n_attention = n_attention if (self.self_attention or self.unmixing_layer) else n_input
            self.subject_layer = SubjectPlusLayer(
                n_attention, n_unmix, n_subjects, regularize=regularize_subject_layer, bias=bias_subject_layer
            )
        else:
            self.subject_layer = None

    def forward(self, xs):
        x, s = xs
        x = self.self_attention(x) if self.self_attention else x
        x = self.unmixing_layer(x) if self.unmixing_layer else x
        x = self.subject_layer(x, s) if self.subject_layer else x
        return x


class TemporalModule(nn.Module):
    def __init__(
            self,
            n_unmix,
            n_block,
    ):
        super().__init__()

        conv_blocks = []
        for block_index in range(0, 5):
            n_in = n_unmix if block_index == 0 else n_block
            conv_blocks.append(ConvBlock(n_in, n_block, block_index))
        self.conv_blocks = nn.ModuleList(conv_blocks)

    def forward(self, x):
        for i, module in enumerate(self.conv_blocks):
            x = module(x)
        return x


class BrainModule(nn.Module):
    def __init__(
            self,
            n_channels_input,
            n_channels_attention,
            n_channels_unmix,
            use_spatial_attention,
            n_spatial_harmonics,
            dirprocess,
            spatial_dropout_number,
            spatial_dropout_radius,
            use_unmixing_layer,
            use_subject_layer,
            n_subjects,
            regularize_subject_layer,
            bias_subject_layer,
            n_channels_block,
            n_features,
            head_pool,
            head_stride,
            meg_sr,
            use_temporal_filter,
            use_temporal_activation,
            use_unmixing_bias,
            use_temporal_bias
    ):
        super().__init__()
        self.spatial_module = SpatialModule(
            n_input=n_channels_input,
            n_attention=n_channels_attention,
            n_unmix=n_channels_unmix,
            use_spatial_attention=use_spatial_attention,
            n_spatial_harmonics=n_spatial_harmonics,
            coords_xy_scaled=np.load(dirprocess + 'coords/coords208_xy_scaled.npy'),
            spatial_dropout_number=spatial_dropout_number,
            spatial_dropout_radius=spatial_dropout_radius,
            use_unmixing_layer=use_unmixing_layer,
            use_unmixing_bias=use_unmixing_bias,
            use_subject_layer=use_subject_layer,
            n_subjects=n_subjects,
            regularize_subject_layer=regularize_subject_layer,
            bias_subject_layer=bias_subject_layer,
            sensor_path=dirprocess
        )

        if use_temporal_filter:
            if use_subject_layer:
                n_channels_out = n_channels_unmix
            elif use_spatial_attention or use_unmixing_layer:
                n_channels_out = n_channels_attention
            else:
                n_channels_out = n_channels_input
            temporal_kernel_size = int(meg_sr * 0.5)  # half second
            if (temporal_kernel_size % 2) == 0:
                temporal_kernel_size += 1

            if use_temporal_activation:
                self.depthwise_conv = nn.Sequential(
                    nn.Conv1d(
                        in_channels=n_channels_out, out_channels=n_channels_out, kernel_size=temporal_kernel_size,
                        groups=n_channels_out, padding='same', bias=use_temporal_bias
                    ),
                    nn.GELU()
                )
            else:
                self.depthwise_conv = nn.Conv1d(
                    in_channels=n_channels_out, out_channels=n_channels_out, kernel_size=temporal_kernel_size,
                    groups=n_channels_out, padding='same', bias=use_temporal_bias
                )
        self.use_temporal_filter = use_temporal_filter

        self.temporal_module = TemporalModule(
            n_unmix=n_channels_unmix,
            n_block=n_channels_block,
        )

        self.feature_projection = ConvHead(
            n_channels=n_channels_block,
            n_features=n_features,
            pool=head_pool,
            head_stride=head_stride,
        )

    def forward(self, xs):
        z = self.spatial_module(xs)
        if self.use_temporal_filter:
            z = self.depthwise_conv(z)
        y = self.temporal_module(z)
        y = self.feature_projection(y)
        return z, y
