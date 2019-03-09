# pylint: disable=E1101
import torch.nn as nn
import torch as th

class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(4, 4, kernel_size = (5,5), stride=(1,1)),
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=(3,3), stride=(1,1)),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=(3,3), stride=(1,1))
        )
    def forward(self, features):
        return self.net(features)

'''
TODO: implement resnet
'''

class FCNBasicBlock(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel):
        super(FCNBasicBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=(3,3), stride=(1,1)),
            # nn.BatchNorm2d(mid_channel),
            nn.ReLU(),
            nn.Conv2d(mid_channel, out_channel, kernel_size=(3,3), stride=(1,1)),
            # nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)
   
class FCNwPool(nn.Module):
    '''
    The model uses three different resolutions:
        - 40m ~ 50m
        - 320m ~ 400m
        - 1280m ~ 1600m
    The output is the sum of all these resolutions.

    TODO: try the weighted network as well. instead of using the adaptive avg pooling,
    learn the weights of each output feature map. size = 3499x1999
    '''
    def __init__(self, shape, pixel_res=20):
        super(FCNwPool, self).__init__()
        self.shape = shape # CxHxW
        self.pixel_res = pixel_res
        self.net = nn.Sequential(
            FCNBasicBlock(shape[0], 8, 16),
            nn.MaxPool2d(kernel_size=(4,4), stride=(4,4)),
            FCNBasicBlock(16, 32, 64),
            nn.MaxPool2d(kernel_size=(4,4), stride=(4,4)),
            FCNBasicBlock(64, 128, 256),
            nn.MaxPool2d(kernel_size=(4,4), stride=(4,4)),
        )
        self.res0 = nn.Sequential(
            nn.ConvTranspose2d(16, 4, kernel_size=(3,3), stride=(1,1)),
            nn.ConvTranspose2d(4, 1, kernel_size=(3,3), stride=(1,1)),
        )
        self.res1 = nn.Sequential(
            nn.ConvTranspose2d(64, 16, kernel_size=(3,3), stride=(16,16)),
            nn.ConvTranspose2d(16, 4, kernel_size=(3,3), stride=(1,1)),
            *[nn.ConvTranspose2d(4, 4, kernel_size=(3,3), stride=(1,1)) for i in range(16)],
            nn.ConvTranspose2d(4, 1, kernel_size=(3,3), stride=(1,1)),
        )
        self.res2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=(3,3), stride=(76, 76)),
            nn.ConvTranspose2d(64, 16, kernel_size=(3,3), stride=(1,1)),
            *[nn.ConvTranspose2d(16, 16, kernel_size=(3,3), stride=(1,1)) for i in range(2)],
            nn.ConvTranspose2d(16, 1, kernel_size=(3,3), stride=(1,1)),
        )
        # self.avgpool = nn.AdaptiveAvgPool2d((self.shape[1], self.shape[2]))
        # self.last = nn.Conv2d(3, 1, kernel_size=(1,1), stride=(1,1))
        self.last = nn.Conv2d(15, 1, kernel_size=(1,1), stride=(1,1))

    def pad_image(self, image, padding, padding_value=0):
        (h, w) = image.shape
        n_image = padding_value * th.ones(h+2*padding, w+2*padding)
        n_image[padding:h+padding, padding:w+padding] = image
        res = th.zeros(5, h, w)
        for i in range(padding, h+padding):
            for j in range(padding, w+padding):
                res[:, i-padding, j-padding] = th.stack(
                    n_image[i-padding, j],
                    n_image[i, j+padding],
                    n_image[i-padding, j],
                    n_image[i, j-padding],
                    n_image[i, j]
                )
        return res

    def get_neighbors(self, features, pixel_res):
        (b, c, h, w) = features.shape # c should be 3 because we have three different resolutions
        n_features = (b, c*5, h, w)
        for i in range(b):
            n_features[i, 0:5, :, :] = self.pad_image(features[i, 0, :, :], 20//pixel_res + 1)
            n_features[i, 5:10, :, :] = self.pad_image(features[i, 1, :, :], 160//pixel_res + 1)
            n_features[i, 10:, :, :] = self.pad_image(features[i, 2, :, :], 640//pixel_res + 1)
        return n_features

    def forward(self, x):
        out0 = self.net[0](x)
        out1 = self.net[1:4](out0)
        out2 = self.net[4:](out1)
        out = th.stack(
            (
                self.res0(out0),
                self.res1(out1),
                self.res2(out2)
            )
        )
        # fx = self.last(out.view(-1, 3, shape[1], shape[2]))
        fx = self.last(self.get_neighbors(out, self.pixel_res))
        print(fx.shape)
        return fx