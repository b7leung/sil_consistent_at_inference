
import torch
import torch.nn as nn

class RendersSemanticDiscriminatorNetwork(nn.Module):

    # DCGAN inspired discirminator, based on code from:
    # https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    def __init__(self, cfg):
        super().__init__()
        nc = 3
        ndf = 64
        dropout_p = cfg["semantic_dis_training"]["dropout_p"]
        input_img_size = cfg["semantic_dis_training"]["dis_input_size"]
        if input_img_size == 64:
            self.main = nn.Sequential(
                # input is (nc) x 64 x 64
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(p=dropout_p),
                # state size. (ndf*2) x 16 x 16
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(p=dropout_p),
                # state size. (ndf*4) x 8 x 8
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(p=dropout_p),
                # state size. (ndf*8) x 4 x 4
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                #nn.Sigmoid()
            )
        elif input_img_size == 128:
            self.main = nn.Sequential(
                # input is (nc) x 64 x 64
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(p=dropout_p),
                # state size. (ndf*2) x 16 x 16
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(p=dropout_p),
                # state size. (ndf*4) x 8 x 8
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(p=dropout_p),
                # state size. (ndf*8) x 4 x 4
                nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 16),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(p=dropout_p),

                nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),
                #nn.Sigmoid()
            )
        elif input_img_size == 256:
            self.main = nn.Sequential(
                # input is (nc) x 64 x 64
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(p=dropout_p),
                # state size. (ndf*2) x 16 x 16
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(p=dropout_p),
                # state size. (ndf*4) x 8 x 8
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(p=dropout_p),
                # state size. (ndf*8) x 4 x 4
                nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 16),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(p=dropout_p),

                nn.Conv2d(ndf * 16, ndf * 32, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 32),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(p=dropout_p),

                nn.Conv2d(ndf * 32, 1, 4, 1, 0, bias=False),
                #nn.Sigmoid()
            )
        else:
            raise ValueError("dis_input_size must be 64, 128, or 256")


    def forward(self, input):
        # [batch_size,1,1,1], so need to squeeze to [batch_size,1]
        logits = self.main(input)
        logits = torch.squeeze(logits,1)
        logits = torch.squeeze(logits,1)
        return logits