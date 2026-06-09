import torch
import torch.nn as nn
from utils.highDHA_utils import initialize_weights
import torch.nn.functional as F

class FCDiscriminator(nn.Module):
    def __init__(self, num_classes):
        super(FCDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(num_classes, num_classes//2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(num_classes//2, num_classes//4, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(num_classes//4, num_classes//8, kernel_size=3, stride=1, padding=1)
        self.classifier = nn.Conv2d(num_classes//8, 1, kernel_size=3, stride=1, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()
        initialize_weights(self)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)## 4 1 64 64
        
        return x

class FCDiscriminator1(nn.Module):
    def __init__(self, num_classes):
        super(FCDiscriminator1, self).__init__()

        self.conv1 = nn.Conv2d(num_classes, num_classes//2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(num_classes//2, num_classes//4, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(num_classes//4, num_classes//8, kernel_size=3, stride=1, padding=1)
        self.classifier = nn.Conv2d(num_classes//8, 1, kernel_size=3, stride=1, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()
        initialize_weights(self)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)## 4 1 64 64
        return x
        
        
class FCDiscriminator2(nn.Module):
    def __init__(self, num_classes):
        super(FCDiscriminator2, self).__init__()

        self.conv1 = nn.Conv2d(num_classes, num_classes//2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(num_classes//2, num_classes//4, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(num_classes//4, num_classes//8, kernel_size=3, stride=1, padding=1)
        self.classifier = nn.Conv2d(num_classes//8, 1, kernel_size=3, stride=1, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()
        initialize_weights(self)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)## 4 1 64 64
        return x

class FCDiscriminator3(nn.Module):
    def __init__(self, num_classes):
        super(FCDiscriminator3, self).__init__()

        self.conv1 = nn.Conv2d(num_classes, num_classes//2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(num_classes//2, num_classes//4, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(num_classes//4, num_classes//8, kernel_size=3, stride=1, padding=1)
        self.classifier = nn.Conv2d(num_classes//8, 1, kernel_size=3, stride=1, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()
        initialize_weights(self)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)## 4 1 64 64
        return x     
        
        
class OutspaceDiscriminator(nn.Module):
    def __init__(self, num_classes, ndf = 16):
        super(OutspaceDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
        self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.leaky_relu3 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.leaky_relu4 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()
        initialize_weights(self)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)## 4 1 4 4 
        return x
           
class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=128, num_classes=1):
        super(PixelDiscriminator, self).__init__()

        # self.D = nn.Sequential(
            # nn.Conv2d(input_nc, ndf, kernel_size=3, stride=1, padding=1),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # nn.Conv2d(ndf, ndf//2, kernel_size=3, stride=1, padding=1),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True)
		# )
        self.Conv2d1 = nn.Conv2d(input_nc, ndf, kernel_size=3, stride=1, padding=1)
        self.Conv2d2 = nn.Conv2d(ndf, ndf//2, kernel_size=3, stride=1, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        self.cls1 = nn.Conv2d(ndf//2, num_classes, kernel_size=3, stride=1, padding=1)
        self.cls2 = nn.Conv2d(ndf//2, num_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, x, size=None):
        # out = self.D(x)
        out = self.Conv2d1(x)
        out = self.leaky_relu(out)
        out = self.Conv2d2(out)
        out = self.leaky_relu(out)
        src_out = self.cls1(out)
        tgt_out = self.cls2(out)
        out = torch.cat((src_out, tgt_out), dim=1)
        if size is not None:
            out = F.interpolate(out, size=size, mode='bilinear', align_corners=True)
        return out
    
def soft_label_cross_entropy(pred, soft_label, pixel_weights=None):
    N, C, H, W = pred.shape
    loss = -soft_label.float()*F.log_softmax(pred, dim=1)
    if pixel_weights is None:
        return torch.mean(torch.sum(loss, dim=1))
    return torch.mean(pixel_weights*torch.sum(loss, dim=1))
        
        
        
        
        
        
        