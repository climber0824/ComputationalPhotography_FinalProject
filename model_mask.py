import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math



class FHDR(nn.Module):
    def __init__(self, iteration_count):
        super(FHDR, self).__init__()
        print("FHDR model initialised")

        self.iteration_count = iteration_count

        self.reflect_pad = nn.ReflectionPad2d(1)
        self.feb1 = nn.Conv2d(3, 64, kernel_size=3, padding=0)
        self.feb2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        self.feedback_block = FeedbackBlock()

        self.hrb1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.hrb2 = nn.Conv2d(64, 3, kernel_size=3, padding=0)

        self.tanh = nn.Tanh()

    def forward(self, input):

        outs = []
        
        #creat input mask
        mask = torch.where(input <= 0.96, 1.0, ((1 - input) / (1 - 0.96)).type(torch.double))
        
        feb1 = F.relu(self.feb1(self.reflect_pad(input)))
        feb2 = F.relu(self.feb2(feb1))
        
        for i in range(self.iteration_count):
            fb_out = self.feedback_block(feb2)
            
            FDF = fb_out + feb1
            
            hrb1 = F.relu(self.hrb1(FDF))
            out = self.hrb2(self.reflect_pad(hrb1))
            out = self.tanh(out)
            #add mask result to feedback result
            out = out + mask * input
            
            outs.append(out)
            
        return outs


class FeedbackBlock(nn.Module):
    def __init__(self):
        super(FeedbackBlock, self).__init__()

        self.compress_in = nn.Conv2d(128, 64, kernel_size=1, padding=0)
        self.DRDB1 = DilatedResidualDenseBlock()
        self.DRDB2 = DilatedResidualDenseBlock()
        self.DRDB3 = DilatedResidualDenseBlock()
        self.last_hidden = None

        self.GFF_3x3 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
        self.should_reset = True

    def forward(self, x):
        if self.should_reset:
            self.last_hidden = torch.zeros(x.size()).cuda()
            self.last_hidden.copy_(x)
            self.should_reset = False

        out1 = torch.cat((x, self.last_hidden), dim=1)
        out2 = self.compress_in(out1)

        out3 = self.DRDB1(out2)
        out4 = self.DRDB2(out3)
        out5 = self.DRDB3(out4)

        out = F.relu(self.GFF_3x3(out5))
        self.last_hidden = out
        self.last_hidden = Variable(self.last_hidden.data)

        return out


class DilatedResidualDenseBlock(nn.Module):
    def __init__(self, nDenselayer=4, growthRate=32):
        super(DilatedResidualDenseBlock, self).__init__()

        nChannels_ = 64
        modules = []

        for i in range(nDenselayer):
            modules.append(make_dense(nChannels_, growthRate))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.should_reset = True

        self.compress = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
        self.conv_1x1 = nn.Conv2d(nChannels_, 64, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        if self.should_reset:
            self.last_hidden = torch.zeros(x.size()).cuda()
            self.last_hidden.copy_(x)
            self.should_reset = False

        cat = torch.cat((x, self.last_hidden), dim=1)

        out = self.compress(cat)
        out = self.dense_layers(out)
        out = self.conv_1x1(out)

        self.last_hidden = out
        self.last_hidden = Variable(out.data)

        return out


class make_dense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3):
        super(make_dense, self).__init__()
        self.conv = nn.Conv2d(
            nChannels,
            growthRate,
            kernel_size=kernel_size,
            padding=(kernel_size - 1),
            bias=False,
            dilation=2,
        )

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out
    
class SoftConvNotLearnedMask(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=True):
        super().__init__()

        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, 1, bias)
        self.mask_update_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, 1, False)
        self.input_conv.apply(weights_init('xavier'))
        
    def forward(self, input, mask):
        output = self.input_conv(input * mask)

        with torch.no_grad():
            self.mask_update_conv.weight = torch.nn.Parameter(self.input_conv.weight.abs())
            filters, _, _, _ = self.mask_update_conv.weight.shape
            k = self.mask_update_conv.weight.view((filters, -1)).sum(1)
            norm = k.view(1,-1,1,1).repeat(mask.shape[0],1,1,1)
            new_mask = self.mask_update_conv(mask)/(norm + 1e-6) 

        return output, new_mask
    
class PCBActiv(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, sample='none-3', activ='relu', conv_bias=False):
        super().__init__()
        if sample == 'down-5':
            self.conv = SoftConvNotLearnedMask(in_ch, out_ch, 5, 2, 2, bias=conv_bias) # Downsampling by 2
        elif sample == 'down-7':
            self.conv = SoftConvNotLearnedMask(in_ch, out_ch, 7, 2, 3, bias=conv_bias) # Downsampling by 2
        elif sample == 'down-3':
            self.conv = SoftConvNotLearnedMask(in_ch, out_ch, 3, 2, 1, bias=conv_bias) # Downsampling by 2
        else:
            self.conv = SoftConvNotLearnedMask(in_ch, out_ch, 3, 1, 1, bias=conv_bias)

        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)
            
    def forward(self, input, input_mask):
        h, h_mask = self.conv(input, input_mask)
        
        if hasattr(self, 'activation'):
            h = self.activation(h)
            
        return h, h_mask

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun