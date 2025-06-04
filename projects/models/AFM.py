import torch
import torch.nn as nn


class AFM(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_atten = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.conv_redu = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)

        self.conv1 = nn.Conv2d(dim, 1, kernel_size=1, stride=1, bias=True)
        self.conv2 = nn.Conv2d(dim, 1, kernel_size=1, stride=1, bias=True)
        self.nonlin = nn.Sigmoid()

    def forward(self, x, skip):
        output = torch.cat([x, skip], dim=1)

        att = self.conv_atten(self.avg_pool(output))
        output = output * att
        output = self.conv_redu(output)

        att = self.conv1(x) + self.conv2(skip)
        att = self.nonlin(att)
        output = output * att
        return output

if __name__ == '__main__':

    x = torch.randn(1, 3, 1280, 800)
    skip = torch.randn(1, 3, 1280, 800)

    block = AFM(3)

    output = block(x, skip)
    import thop
    flops, params = thop.profile(block, inputs=(x, skip))
    print('flops: ', flops / 1e9, 'G')
    # print("Input shape (x):", x.size())
    # print("Input shape (skip):", skip.size())
    # print("Output shape:", output.size())