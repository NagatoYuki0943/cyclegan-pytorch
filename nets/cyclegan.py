import torch
import torch.nn as nn
import torch.nn.functional as F


#----------------------------------------#
#   残差快,通道宽高都不变
#   主干: 3x3Conv -> IN -> ReLU -> 3x3Conv -> IN
#   直接拼接输入的x
#   IN: 假设 [b,c,h,w], 在 hw 上计算, b*c*hw
#----------------------------------------#
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        ]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


#----------------------------------------#
#   生成器
#   传入图像,逐渐增多通道,减小宽高,再堆叠9次残差块,
#   再逐渐减小通道,增大宽高,最后形状和输入相同
#----------------------------------------#
class generator(nn.Module):
    def __init__(self, in_channel=3, out_channel=3, n_residual_blocks=9):
        super(generator, self).__init__()

        #----------------------------------#
        #   Initial convolution block
        #   7x7Conv
        #   [b, 3, h, w] -> [b, 64, h, w]
        #----------------------------------#
        self.stage_1 = nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(in_channel, 64, 7),
                nn.InstanceNorm2d(64),
                nn.ReLU(inplace=True)
        )

        #----------------------------------#
        #   3x3Conv
        #   [b, 64, h, w] -> [b, 128, h/2, w/2]
        #----------------------------------#
        self.stage_2 = nn.Sequential(
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.InstanceNorm2d(128),
                nn.ReLU(inplace=True)
        )

        #----------------------------------#
        #   3x3Conv
        #   [b, 128, h/2, w/2] -> [b, 256, h/4, w/4]
        #----------------------------------#
        self.stage_3 = nn.Sequential(
                nn.Conv2d(128, 256, 3, stride=2, padding=1),
                nn.InstanceNorm2d(256),
                nn.ReLU(inplace=True)
        )

        #----------------------------------#
        #   Residual blocks
        #   [b, 256, h/4, w/4] -> [b, 256, h/4, w/4]
        #----------------------------------#
        self.stage_4 = []
        for _ in range(n_residual_blocks):
            self.stage_4 += [ResidualBlock(256)]
        self.stage_4 = nn.Sequential(*self.stage_4)

        #----------------------------------#
        #   3x3Conv
        #   [b, 256, h/4, w/4] -> [b, 128, h/2, w/2]
        #----------------------------------#
        self.up_stage_1 = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(128),
                nn.ReLU(inplace=True)
        )

        #----------------------------------#
        #   3x3Conv
        #   [b, 128, h/2, w/2] -> [b, 64, h, w]
        #----------------------------------#
        self.up_stage_2 = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(64),
                nn.ReLU(inplace=True)
        )

        #----------------------------------#
        #   7x7Conv
        #   [b, 64, h, w] -> [b, 3, h, w]
        #----------------------------------#
        self.head = nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(64, out_channel, 7),
                nn.Tanh()
        )

        self.weight_init()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(0.1, 0.02)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        x = self.stage_1(x)     # [b, 3,   h,   w]   -> [b, 64,  h,   w]
        x = self.stage_2(x)     # [b, 64,  h,   w]   -> [b, 128, h/2, w/2]
        x = self.stage_3(x)     # [b, 128, h/2, w/2] -> [b, 256, h/4, w/4]
        x = self.stage_4(x)     # [b, 256, h/4, w/4] -> [b, 256, h/4, w/4]
        x = self.up_stage_1(x)  # [b, 256, h/4, w/4] -> [b, 128, h/2, w/2]
        x = self.up_stage_2(x)  # [b, 128, h/2, w/2] -> [b, 64, h, w]
        return self.head(x)     # [b, 64, h, w] -> [b, 3, h, w]


#----------------------------------------#
#   辨别器
#   将图像经过多次下采样和最后的pooling,得到维度为1的结果
#----------------------------------------#
class discriminator(nn.Module):
    def __init__(self, in_channel=3):
        super(discriminator, self).__init__()

        #----------------------------------#
        #   4x4Conv
        #   [b, 3, h, w] -> [b, 64, h/2, w/2]
        #----------------------------------#
        self.stage_1 = nn.Sequential(
                nn.Conv2d(in_channel, 64, 4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
        )

        #----------------------------------#
        #   4x4Conv
        #   [b, 64, h/2, w/2] -> [b, 128, h/4, w/4]
        #----------------------------------#
        self.stage_2 = nn.Sequential(
                nn.Conv2d(64, 128, 4, stride=2, padding=1),
                nn.InstanceNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True)
        )

        #----------------------------------#
        #   4x4Conv
        #   [b, 128, h/4, w/4] -> [b, 256, h/8, w/8]
        #----------------------------------#
        self.stage_3 = nn.Sequential(
                nn.Conv2d(128, 256, 4, stride=2, padding=1),
                nn.InstanceNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True)
        )

        #----------------------------------#
        #   4x4Conv
        #   [b, 256, h/8, w/8] -> [b, 512, ?, ?]
        #----------------------------------#
        self.stage_4 = nn.Sequential(
                nn.Conv2d(256, 512, 4, padding=1),
                nn.InstanceNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True)
        )

        #----------------------------------#
        #   4x4Conv
        #   [b, 512, ?, ?] -> [b, 1, 1, 1]
        #----------------------------------#
        self.head = nn.Sequential(
                nn.Conv2d(512, 1, 4, padding=1),
                nn.AdaptiveAvgPool2d([1, 1])
        )

        self.weight_init()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(0.1, 0.02)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        x = self.stage_1(x)     # [b, 3, h, w] -> [b, 64, h/2, w/2]
        x = self.stage_2(x)     # [b, 64, h/2, w/2] -> [b, 128, h/4, w/4]
        x = self.stage_3(x)     # [b, 128, h/4, w/4] -> [b, 256, h/8, w/8]
        x = self.stage_4(x)     # [b, 256, h/8, w/8] -> [b, 512, ?, ?]
        x = self.head(x)        # [b, 512, ?, ?] -> [b, 1, 1, 1]

        return x.view(x.size()[0])  # [b, 1, 1, 1] -> [b]


if __name__ == "__main__":
    x = torch.randn(1, 3, 64, 64)
    gen = generator()
    y = gen(x)
    print(y.size())     # [1, 3, 64, 64]
    dis = discriminator()
    z = dis(y)
    print(z.size())     # [1]