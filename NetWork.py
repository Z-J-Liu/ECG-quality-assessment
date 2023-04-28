import torch
import torch.nn as nn
from einops import rearrange


def conv_15(in_planes, out_planes, stride=1, groups=1, dilation=1, padding=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=15, stride=stride,
                     padding=padding, groups=groups, bias=False, dilation=dilation)


def conv_1(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self.conv1 = conv_15(inplanes, planes, stride, padding=7)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_15(planes, planes, padding=7)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ResTransformerV2(nn.Module):
    def __init__(self):
        super(ResTransformerV2, self).__init__()

        self.conv1 = nn.Conv1d(12, 64, kernel_size=15, stride=4, padding=6,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool1d(kernel_size=15, stride=2, padding=7)

        self.layer1 = nn.Sequential(
            BasicBlock(inplanes=64, planes=64),
            BasicBlock(inplanes=64, planes=64),
            BasicBlock(inplanes=64, planes=64)
        )

        self.downsample1 = nn.Sequential(
            conv_1(in_planes=64, out_planes=128, stride=2),
            nn.BatchNorm1d(128))

        self.layer2 = nn.Sequential(
            BasicBlock(inplanes=64, planes=128, stride=2, downsample=self.downsample1),
            BasicBlock(inplanes=128, planes=128),
            BasicBlock(inplanes=128, planes=128)
        )

        self.downsample2 = nn.Sequential(
            conv_1(in_planes=128, out_planes=256, stride=2),
            nn.BatchNorm1d(256))

        self.layer3 = nn.Sequential(
            BasicBlock(inplanes=128, planes=256, stride=2, downsample=self.downsample2),
            BasicBlock(inplanes=256, planes=256),
            BasicBlock(inplanes=256, planes=256)
        )

        self.downsample3 = nn.Sequential(
            conv_1(in_planes=256, out_planes=512, stride=2),
            nn.BatchNorm1d(512))

        self.layer4 = nn.Sequential(
            BasicBlock(inplanes=256, planes=512, stride=2, downsample=self.downsample3),
            BasicBlock(inplanes=512, planes=512),
            BasicBlock(inplanes=512, planes=512)
        )

        self.downsample4 = nn.Sequential(
            conv_1(in_planes=512, out_planes=1024, stride=2),
            nn.BatchNorm1d(1024))

        self.layer5 = nn.Sequential(
            BasicBlock(inplanes=512, planes=1024, stride=2, downsample=self.downsample4),
            BasicBlock(inplanes=1024, planes=1024),
            BasicBlock(inplanes=1024, planes=1024)
        )

        self.avgpool = nn.AdaptiveAvgPool1d(output_size=1)

        self.transformer = Transformer(dim=1280, depth=2, heads=16, dim_head=64, mlp_dim=2048, dropout=0.5)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x1 = self.transformer(x)
        x = x + x1
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x


class NetWork(nn.Module):
    """
    resnet + transformer
    """
    def __init__(self):
        super(NetWork, self).__init__()
        self.encoder = ResTransformerV2()
        self.fc = nn.Linear(1024, 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        x = self.softmax(x)

        return x








