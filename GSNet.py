from numpy import outer
import torch
import torch.nn as nn
import torch.nn.functional as F
from pvtv2 import pvt_v2_b5

torch.cuda.set_device(0)

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class FEF(nn.Module):
    def __init__(self, channel):
        super(FEF, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample6 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample7 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample8 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample9 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample10 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample11 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample12 = BasicConv2d(channel, channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 *channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv_concat4 = BasicConv2d(4 * channel, 4 * channel, 3, padding=1)


        self.channel_dowm_2 = BasicConv2d( 2*channel, channel, 3, padding=1)
        self.channel_dowm_3 = BasicConv2d( 3*channel, channel, 3, padding=1)
        self.channel_dowm_4 = BasicConv2d( 4*channel, channel, 3, padding=1)


    def forward(self, x1, x2, x3, x4):
        x1_1 = self.conv_upsample1(self.upsample_2(x1))
        x2_1 = self.conv_upsample2(self.upsample_2(x1))
        x3_1 = self.conv_upsample3(self.upsample_4(x1)) 
        x4_1 = self.conv_upsample4(self.upsample_8(x1)) 

        x3_2 = self.conv_upsample5(self.upsample_2(x2))
        x4_2 = self.conv_upsample6(self.upsample_4(x2))

        x4_3 = self.conv_upsample7(self.upsample_2(x3))
        
        x21_1 = self.conv_upsample8(self.upsample_2(x1))*x2
        x321_1 = self.conv_upsample9(self.upsample_2(x21_1))*x3

        x21_2 = self.conv_upsample10(self.upsample_2(x1))*x2
        x321_2 = self.conv_upsample11(self.upsample_2(x21_2))*x3
        x4321_2 =  self.conv_upsample12(self.upsample_2(x321_2))*x4



        out2 = torch.cat((x1_1, x2_1*x2), 1)
        out2 = self.conv_concat2(out2)

        out3 = torch.cat((x321_1, x3_1, x3_2), 1)
        out3 = self.conv_concat3(out3)

        out4 = torch.cat((x4321_2, x4_1, x4_2, x4_3), 1)
        out4 = self.conv_concat4(out4)

        out_2 = self.channel_dowm_2(out2)
        out_3 = self.channel_dowm_3(out3) 
        out_4 = self.channel_dowm_4(out4) 

        return out_2,out_3,out_4


class DAG(nn.Module):
    def __init__(self, num_in=32, plane_mid=32):
        super(DAG, self).__init__()

        self.DConv = DC3x3(plane_mid, plane_mid, plane_mid)

        self.num_s = int(plane_mid)

        self.conv_state = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.conv_proj = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.conv_proj_2 = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.conv_extend = nn.Conv2d(self.num_s, num_in, kernel_size=1, bias=False)

    def forward(self, x, edge): #x是被引导的特征 
        # edge = F.upsample(edge, (x.size()[-2], x.size()[-1]))

        n, c, h, w = x.size()

        x_state_reshaped = self.conv_state(x).view(n, self.num_s, -1) #[4, 16, 2304]
        x_proj_reshaped = self.DConv(self.conv_proj(x),self.conv_proj_2(edge)).reshape(n, self.num_s, -1)
        x_proj_reshaped = F.softmax(x_proj_reshaped, dim=1)
        x_state_reshaped = x_proj_reshaped * x_state_reshaped
        x_state = x_state_reshaped.view(n, self.num_s, *x.size()[2:])
        out = x + (self.conv_extend(x_state))

        return out
        

class DC3x3(nn.Module):
    def __init__(self, in_xC=32, in_yC=32, out_C=32):
        """DepthDC3x3_1，利用nn.Unfold实现的动态卷积模块

        Args:
            in_xC (int): 第一个输入的通道数
            in_yC (int): 第二个输入的通道数
            out_C (int): 最终输出的通道数
            down_factor (int): 用来降低卷积核生成过程中的参数量的一个降低通道数的参数
        """
        super(DC3x3, self).__init__()
        self.kernel_size = 3
        self.gernerate_kernel = nn.Sequential(
            nn.Conv2d(in_yC, in_yC, 3, 1, 1),
            nn.Conv2d(in_yC, in_xC * self.kernel_size ** 2, 1),
        )
        self.unfold = nn.Unfold(kernel_size=3, dilation=1, padding=1, stride=1)
   

    def forward(self, x, y): #x是被引导的特征，y是成核的特征
        N, xC, xH, xW = x.size()
        kernel = self.gernerate_kernel(y).reshape([N, xC, self.kernel_size ** 2, xH, xW])        
        unfold_x = self.unfold(x).reshape([N, xC, -1, xH, xW])
        result = (unfold_x * kernel).sum(2)
        
        return result


class GSNet(nn.Module):
    def __init__(self, channel=32):
        super(GSNet, self).__init__()

        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.backbone = pvt_v2_b5()  # [64, 128, 320, 512]
        path = '/mnt/cpfs/dataset/tuxiangzu/Classfication_Detection_Group/whb/V100/GSNet/pvt_v2_b5.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.Translayer1_1 = BasicConv2d(64, channel, 1)
        self.Translayer2_1 = BasicConv2d(128, channel, 1)
        self.Translayer3_1 = BasicConv2d(320, channel, 1)
        self.Translayer4_1 = BasicConv2d(512, channel, 1)

        self.DAG1 = DAG(channel,channel)
        self.DAG2 = DAG(channel,channel)
        self.FEF = FEF(channel)
 
        self.out_G = nn.Conv2d(channel, 1, 1)
        self.out_B = nn.Conv2d(channel, 1, 1)
        self.out_P = nn.Conv2d(channel, 1, 1)


    def forward(self, x):

        # backbone
        pvt = self.backbone(x)
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]
        
       
        # CFM
        x1_t = self.Translayer1_1(x1)  
        x2_t = self.Translayer2_1(x2)  
        x3_t = self.Translayer3_1(x3)  
        x4_t = self.Translayer4_1(x4)  

        gauss,body,pre = self.FEF(x4_t, x3_t, x2_t, x1_t)

        body = self.DAG1(body,self.upsample_2(gauss))
        pre = self.DAG2(pre,self.upsample_2(body))

        out_gauss = self.out_G(gauss)
        out_body = self.out_B(body)
        out_pre = self.out_P(pre)

        out_gauss = F.interpolate(out_gauss, scale_factor=16, mode='bilinear') 
        out_body = F.interpolate(out_body, scale_factor=8, mode='bilinear')  
        out_pre = F.interpolate(out_pre, scale_factor=4, mode='bilinear') 


        return out_gauss, out_body, out_pre


if __name__ == '__main__':
    model = GSNet().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    prediction1, prediction2, prediction3 = model(input_tensor)
    print(prediction1.size(), prediction2.size(), prediction3.size())


