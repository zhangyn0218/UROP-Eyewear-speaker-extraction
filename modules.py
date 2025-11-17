import torch
import torch.nn as nn
import torch.nn.functional as F
import copy, math

def overlap_and_add(signal, frame_step):
    outer_dimensions = signal.size()[:-2]
    frames, frame_length = signal.size()[-2:] #the num of frames and its length
    subframe_length = math.gcd(frame_length, frame_step)  # gcd=Greatest Common Divisor
    print(outer_dimensions,frames,frame_length,subframe_length)
    subframe_step = frame_step // subframe_length # step between subframe
    subframes_per_frame = frame_length // subframe_length # the num of subframes per frame
    output_size = frame_step * (frames - 1) + frame_length # the overall output size
    output_subframes = output_size // subframe_length # the num of output frames
    print("subframes_per_frame",subframes_per_frame,"output_subframes",output_subframes,"subframe_step",subframe_step,"output_subframes",output_subframes)
    subframe_signal = signal.view(*outer_dimensions, -1, subframe_length)
    print(signal.shape,subframe_signal.shape)
    frame = torch.arange(0, output_subframes,
                         device=signal.device).unfold(0, subframes_per_frame, subframe_step)# select 2 element each time with step1

    frame = frame.long()  # signal may in GPU or CPU
    frame = frame.contiguous().view(-1)# frame is for location
    result = signal.new_zeros(*outer_dimensions, output_subframes, subframe_length)
    print(result.shape)
    #print(subframe_signal)
    #print(frame)
    result.index_add_(-2, frame, subframe_signal)
    print(result.shape)
    result = result.view(*outer_dimensions, -1)
    print(result.shape)

    return result

def _clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class ResNetLayer(nn.Module):

    """
    A ResNet layer used to build the ResNet network.
    Architecture:
    --> conv-bn-relu -> conv -> + -> bn-relu -> conv-bn-relu -> conv -> + -> bn-relu -->
     |                        |   |                                    |
     -----> downsample ------>    ------------------------------------->
    """

    def __init__(self, inplanes, outplanes, stride):
        super(ResNetLayer, self).__init__()
        self.conv1a = nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1a = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv2a = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.stride = stride
        self.downsample = nn.Conv2d(inplanes, outplanes, kernel_size=(1,1), stride=stride, bias=False)
        self.outbna = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)

        self.conv1b = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1b = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv2b = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.outbnb = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        return


    def forward(self, inputBatch):
        #print('resnet')
        #print(inputBatch.shape)
        batch = F.relu(self.bn1a(self.conv1a(inputBatch)))
        #print(batch.shape)
        batch = self.conv2a(batch)
        #print(batch.shape)
        if self.stride == 1:
            residualBatch = inputBatch
        else:
            residualBatch = self.downsample(inputBatch)
            #print('hey')
        #print('residualBatch')
        #print(residualBatch.shape)
        batch = batch + residualBatch
        intermediateBatch = batch
        batch = F.relu(self.outbna(batch))
        #print(batch.shape)
        batch = F.relu(self.bn1b(self.conv1b(batch)))
        #print(batch.shape)
        batch = self.conv2b(batch)
        #print(batch.shape)
        residualBatch = intermediateBatch
        batch = batch + residualBatch
        outputBatch = F.relu(self.outbnb(batch))
        return outputBatch


class GlobalLayerNorm(nn.Module):
    def __init__(self, channel_size):
        super(GlobalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True) #[M, 1, 1]
        var = (torch.pow(y-mean, 2)).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        gLN_y = self.gamma * (y - mean) / torch.pow(var + 1e-8, 0.5) + self.beta
        return gLN_y


class ResNet(nn.Module):

    """
    An 18-layer ResNet architecture.
    """

    def __init__(self):
        super(ResNet, self).__init__()
        self.layer1 = ResNetLayer(64, 64, stride=1)
        self.layer2 = ResNetLayer(64, 128, stride=2)
        self.layer3 = ResNetLayer(128, 256, stride=2)
        self.layer4 = ResNetLayer(256, 512, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=(4,4), stride=(1,1))
        
        return


    def forward(self, inputBatch):
        batch = self.layer1(inputBatch)
        batch = self.layer2(batch)
        batch = self.layer3(batch)
        batch = self.layer4(batch)
        outputBatch = self.avgpool(batch)
        return outputBatch

class visualFrontend(nn.Module):

    """
    A visual feature extraction module. Generates a 512-dim feature vector per video frame.
    Architecture: A 3D convolution block followed by an 18-layer ResNet.
    """

    def __init__(self):
        super(visualFrontend, self).__init__()
        #self.test=nn.Conv3d(1, 64, kernel_size=(5,7,7), stride=(1,2,2), padding=(2,3,3), bias=False)
        #self.test2=nn.BatchNorm3d(64, momentum=0.01, eps=0.001)
        #self.test3=nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        self.frontend3D = nn.Sequential(
                            nn.Conv3d(1, 64, kernel_size=(5,7,7), stride=(1,2,2), padding=(2,3,3), bias=False),
                            nn.BatchNorm3d(64, momentum=0.01, eps=0.001),
                            nn.ReLU(),
                            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
                        )
        self.resnet = ResNet()
        return


    def forward(self, inputBatch):
        #print('visualFrontend')
        #print(inputBatch.shape)
        inputBatch = inputBatch.transpose(0, 1).transpose(1, 2)
        #print(inputBatch.shape)
        batchsize = inputBatch.shape[0]
        #batch = self.test(inputBatch)
        #batch=self.test2(batch)
        #batch=self.test3(batch)
        batch = self.frontend3D(inputBatch)
        #print(batch.shape)
        batch = batch.transpose(1, 2)
        batch = batch.reshape(batch.shape[0]*batch.shape[1], batch.shape[2], batch.shape[3], batch.shape[4])
        #print(batch.shape)
        outputBatch = self.resnet(batch)
        #print(outputBatch.shape)
        outputBatch = outputBatch.reshape(batchsize, -1, 512)
        #print(outputBatch.shape)
        outputBatch = outputBatch.transpose(1 ,2)
        #print(outputBatch.shape)
        outputBatch = outputBatch.transpose(1, 2).transpose(0, 1)
        #print(outputBatch.shape)
        return outputBatch

class Encoder(nn.Module):
    def __init__(self, L, N):
        super(Encoder, self).__init__()
        self.L, self.N = L, N
        self.conv1d_U = nn.Conv1d(1, N, kernel_size=L, stride=L // 2, bias=False)

    def forward(self, mixture):

        mixture = torch.unsqueeze(mixture, 1)  # [M, 1, T]

        mixture_w = F.relu(self.conv1d_U(mixture))  # [M, N, K]

        return mixture_w

class Decoder(nn.Module):
    def __init__(self, N, L):
        super(Decoder, self).__init__()
        self.N, self.L = N, L
        self.basis_signals = nn.Linear(N, L, bias=False)
        print('N',N,'L',L)
    def forward(self, mixture_w, est_mask):

        est_source = mixture_w * est_mask  # [M,  N, K]
        est_source = torch.transpose(est_source, 2, 1) # [M,  K, N]

        est_source = self.basis_signals(est_source)  # [M,  K, L]
        est_source = overlap_and_add(est_source, self.L//2) # M x C x T
        return est_source


class TemporalConvNet(nn.Module):
    def __init__(self, N, B, H, P, X, R, C):
        super(TemporalConvNet, self).__init__()
        self.C = C
        self.layer_norm = ChannelWiseLayerNorm(N)
        self.bottleneck_conv1x1 = nn.Conv1d(N, B, 1, bias=False)
        #print(N,B,X,R)
        # Audio TCN
        blocks = []
        for x in range(X):#N=256,B=256,X=8,R=4 4block, each contains 8 temporalblocks
            dilation = 2**x
            padding = (P - 1) * dilation // 2
            blocks += [TemporalBlock(B, H, P, stride=1,
                                     padding=padding,
                                     dilation=dilation)]

        self.tcn = _clones(nn.Sequential(*blocks), R)

        self.mask_conv1x1 = nn.Conv1d(B, N, 1, bias=False)

        # visual
        stacks = []
        for x in range(5):
            stacks += [VisualConv1D()]
        self.visual_conv = nn.Sequential(*stacks)
        self.av_conv = _clones(nn.Conv1d(B+512, B, 1, bias=False),R)


    def forward(self, x, visual):
        #print(x.shape)
        #print(visual.shape)
        K = x.size()[-1]
        visual = visual.transpose(1,2)
        #print(visual.size()[-1])

        visual = self.visual_conv(visual)

        ###

        x = self.layer_norm(x)
        x = self.bottleneck_conv1x1(x)

        visual = F.interpolate(visual, (32*visual.size()[-1]), mode='linear')
        visual = F.pad(visual,(0,K-visual.size()[-1]))# complement for the matching size of audio

        x = self.tcn[0](x)
        x = torch.cat((x, visual),1)
        #print(x.shape)
        x = self.av_conv[0](x)
        #print(x.shape)
        x = self.tcn[1](x)
        x = torch.cat((x, visual),1)
        x = self.av_conv[1](x)

        x = self.tcn[2](x)
        x = torch.cat((x, visual),1)
        x = self.av_conv[2](x)

        x = self.tcn[3](x)
        x = torch.cat((x, visual),1)
        x = self.av_conv[3](x)
        x = self.mask_conv1x1(x)
        x = F.relu(x)
        #done 2023.08.01
        return x

class VisualConv1D(nn.Module):
    def __init__(self):
        super(VisualConv1D, self).__init__()
        relu = nn.ReLU()
        norm_1 = nn.BatchNorm1d(512)
        dsconv = nn.Conv1d(512, 512, 3, stride=1, padding=1,dilation=1, groups=512, bias=False)
        prelu = nn.PReLU()
        norm_2 = nn.BatchNorm1d(512)
        pw_conv = nn.Conv1d(512, 512, 1, bias=False)
        self.test1=nn.ReLU()
        self.test2=nn.BatchNorm1d(512)
        self.test3=nn.Conv1d(512, 512, 3, stride=1, padding=1,dilation=1, groups=512, bias=False)
        self.test4=nn.PReLU()
        self.test5=nn.BatchNorm1d(512)
        self.test6=nn.Conv1d(512, 512, 1, bias=False)
        self.net = nn.Sequential(relu, norm_1 ,dsconv, prelu, norm_2, pw_conv)

    def forward(self, x):

        out = self.net(x)
        return out + x

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, dilation):
        super(TemporalBlock, self).__init__()
        #print(in_channels,out_channels,kernel_size,stride, padding, dilation)
        conv1x1 = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        prelu = nn.PReLU()
        norm = GlobalLayerNorm(out_channels)
        dsconv = DepthwiseSeparableConv(out_channels, in_channels, kernel_size,
                                        stride, padding, dilation)
        # Put together
        self.net = nn.Sequential(conv1x1, prelu, norm, dsconv)
        self.test1 = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.test2 = nn.PReLU()
        self.test3 = GlobalLayerNorm(out_channels)
        self.test4 = DepthwiseSeparableConv(out_channels, in_channels, kernel_size,
                                        stride, padding, dilation)


    def forward(self, x):
        '''
        print('sdas')
        print(x.shape)
        x=self.test1(x)
        print(x.shape)
        x=self.test2(x)
        print(x.shape)
        x=self.test3(x)
        print(x.shape)
        x=self.test4(x)
        residual = x
        out=x
        '''
        residual = x
        out = self.net(x)
        return out + residual  # look like w/o F.relu is better than w/ F.relu


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, dilation):
        super(DepthwiseSeparableConv, self).__init__()
        depthwise_conv = nn.Conv1d(in_channels, in_channels, kernel_size,
                                   stride=stride, padding=padding,
                                   dilation=dilation, groups=in_channels,
                                   bias=False)
        #print(in_channels, out_channels, kernel_size,
        #         stride, padding, dilation)
        prelu = nn.PReLU()
        norm = GlobalLayerNorm(in_channels)
        pointwise_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.net = nn.Sequential(depthwise_conv, prelu, norm,
                                 pointwise_conv)
        self.test1=nn.Conv1d(in_channels, in_channels, kernel_size,
                                   stride=stride, padding=padding,
                                   dilation=dilation, groups=in_channels,
                                   bias=False)
        self.test2=nn.PReLU()
        self.test3=GlobalLayerNorm(in_channels)
        self.test4=nn.Conv1d(in_channels, out_channels, 1, bias=False)
    def forward(self, x):
        '''
        print('depthwise')
        y=x
        y=self.test1(y)
        print(y.shape)
        y=self.test2(y)
        print(y.shape)
        y=self.test3(y)
        print(y.shape)
        y=self.test4(y)
        print(y.shape)
        '''
        return self.net(x)#y

class ChannelWiseLayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super(ChannelWiseLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        # N x C x T => N x T x C
        x = torch.transpose(x, 1, 2)
        # LN
        x = super().forward(x)
        # N x C x T => N x T x C
        x = torch.transpose(x, 1, 2)
        return x