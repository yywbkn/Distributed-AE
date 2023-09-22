"""Model class template

This module provides a template for users to implement custom models.
You can specify '--model template' to use this model.
The class name should be consistent with both the filename and its model option.
The filename should be <model>_dataset.py
The class name should be <Model>Dataset.py
It implements a simple image-to-image translation baseline based on regression loss.

You need to implement the following functions:
    <modify_commandline_options>:　Add model-specific options and rewrite default values for existing options.
    <__init__>: Initialize this model class.
    <set_input>: Unpack input data and perform data pre-processing.
    <forward>: Run forward pass. This will be called by both <optimize_parameters> and <test>.
    <optimize_parameters>: Update network weights; it will be called in every training iteration.
"""
import torch
from . import networks
from .base_model import BaseModel
from torch import nn
import torch.nn.functional as F
import numpy as np
from .perception_loss import  perceptual_loss,mobilenet_feat
from sklearn.metrics import roc_auc_score


def text_save_w(filename, data):  # filename为写入CSV文件的路径，data为要写入数据列表.
    file = open(filename, 'w')
    for i in range(len(data)):
        # s = str(data[i]).replace('[', '').replace(']', '')  # 去除[],这两行按数据不同，可以选择
        # s = s.replace("'", '').replace(',', '') + '\n'  # 去除单引号，逗号，每行末尾追加换行符
        s = str(data[i])
        s = s + '\n'  # 去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print("保存文件成功")

def text_save_a(filename, data):  # filename为写入CSV文件的路径，data为要写入数据列表.
    file = open(filename, 'a')
    for i in range(len(data)):
        # s = str(data[i]).replace('[', '').replace(']', '')  # 去除[],这两行按数据不同，可以选择
        # s = s.replace("'", '').replace(',', '') + '\n'  # 去除单引号，逗号，每行末尾追加换行符
        s = str(data[i])
        s = s + '\n'  # 去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print("保存文件成功")


class asynKLModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.
        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        """
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--delta_perceptual', type=float, default=10.0, help='weight for perceptual loss')
            parser.add_argument('--lambda_L2', type=float, default=10.0, help='weight for L2 loss')
            parser.add_argument('--lambda_G', type=float, default=0.1, help='weight for asyndgan G ')
            parser.add_argument('--lambda_D', type=float, default=0.1, help='weight for asyndgan D')
        return parser

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        self.inde = 2
        # specify the training losses you want to print out. The program will call base_model.get_current_losses to plot the losses to the console and save them to the disk.
        self.loss_names = ['generator_loss_all','C','G_L1_all','G_perceptual_all','GC_loss_all','C_loss_all']
        # specify the images you want to save and display. The program will call base_model.get_current_visuals to save and display these images.
        if self.isTrain:
            self.visual_names = ['data_0','fake_image_0']
        else:  # during test time, only load Gs
            self.visual_names = ['data_0']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks to save and load networks.
        # you can use opt.isTrain to specify different behaviors for training and test. For example, some networks will not be used during test, and you don't need to load them.
        self.generator_output_list = []
        self.classifier_output_list = []
        self.label_list = []
        if self.isTrain:
            self.model_names = ['classifier', 'generator']
        else:  # during test time, only load Gs
            # self.model_names = ['classifier']
            self.model_names = ['generator']
            self.predlist = []
            self.scorelist = []
            self.targetlist = []
            self.correct = 0

        # define networks; you can use opt.isTrain to specify different behaviors for training and test.
        self.netclassifier = networks.define_C(opt.netC, gpu_ids=self.gpu_ids)


        self.netgenerator = []
        for i in range(3):
            # self.netgenerator.append(networks.resnet18(gpu_ids=self.gpu_ids))
            self.netgenerator.append(networks.define_G(opt.netG,flag = True, gpu_ids=self.gpu_ids))


        if self.isTrain:  # only defined during training time
            # define your loss functions. You can use losses provided by torch.nn such as torch.nn.L1Loss.
            # We also provide a GANLoss class "networks.GANLoss". self.criterionGAN = networks.GANLoss().to(self.device)
            self.criterionLoss = networks.CE_loss_fn()
            self.criterionL1 = torch.nn.SmoothL1Loss()
            self.mobilenet_model = mobilenet_feat().cuda()
            self.criterion_perceptual = perceptual_loss()

            # define and initialize optimizers. You can define one optimizer for each network.
            # If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
            self.optimizer_C = torch.optim.RMSprop(self.netclassifier.parameters(), lr=opt.lr)
            self.optimizer_G = []

            for i in self.netgenerator:
                opt_D = torch.optim.RMSprop(i.parameters(), lr=opt.lr)
                self.optimizer_G.append(opt_D)
                self.optimizers.append(opt_D)

            self.optimizers.append(self.optimizer_C)


        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        if self.isTrain:
            self.data = []
            self.label = []
            for i in range(3):
                self.data.append(input['data_' + str(i)].to(self.device))
                self.label.append(input['label_' + str(i)].to(self.device))

            self.data_0 = self.data[0]
            self.data_1 = self.data[1]

        if not self.isTrain:
            self.data = input['data'].to(self.device)  # get image data
            self.label = input['label'].to(self.device)  # get image label

            self.data_0 = self.data

        # self.image_paths = input['A_paths']  # get image paths

    def set_val_input(self, input):

        self.val_data = input['data'].to(self.device)  # get image data
        self.val_label = input['label'].to(self.device)  # get image label
        # self.data_0 = self.data


    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        # self.classifier_output = []
        # for i in range(len(self.data)):
        #     self.classifier_output_tmp = self.netclassifier(self.data[i])
        #     self.classifier_output.append(self.classifier_output_tmp)   #classifier
        # print('classifier_output[0] =', self.classifier_output[0])
        pass

    def backward_G(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results

        self.fake_image = []
        self.loss_generator_loss =[]
        self.loss_G_L1 = []
        self.latent_i_tmp = []
        self.loss_G_perceptual = []
        self.prob_tmp = []

        for i in range(len(self.data)):
            gen_image, latent_i, prob = self.netgenerator[i](self.data[i],self.label[i])

            self.fake_image.append(gen_image)
            self.latent_i_tmp.append(latent_i)
            self.prob_tmp.append(prob)

            pred_feat = self.mobilenet_model(gen_image)
            target_feat = self.mobilenet_model(self.data[i])

            self.loss_G_perceptual.append(self.criterion_perceptual(pred_feat, target_feat) * self.opt.delta_perceptual)
            self.loss_generator_loss.append(self.criterionLoss(prob, self.label[i].long()))
            self.loss_G_L1.append(self.criterionL1(gen_image, self.data[i]) * self.opt.lambda_L1)

        self.loss_generator_loss_all = None
        self.loss_G_L1_all = None
        self.loss_G_perceptual_all = None

        self.fake_image_0 = self.fake_image[0]

        for i in range(len(self.loss_generator_loss)):
            if self.loss_generator_loss_all is None:
                self.loss_generator_loss_all = self.loss_generator_loss[i]
                self.loss_G_L1_all = self.loss_G_L1[i]
                self.loss_G_perceptual_all = self.loss_G_perceptual[i]
            else:
                self.loss_generator_loss_all += self.loss_generator_loss[i]
                self.loss_G_L1_all += self.loss_G_L1[i]
                self.loss_G_perceptual_all += self.loss_G_perceptual[i]

        self.loss_generator_loss_all = self.loss_generator_loss_all *100
        # combine loss and calculate gradients
        self.loss_G = (self.loss_generator_loss_all + self.loss_G_L1_all  + self.loss_G_perceptual_all)*self.opt.lambda_G
        # self.loss_S.backward()
        self.loss_G.backward(retain_graph=True)



    def backward_C(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results
        # netclassifier
        self.loss_C_loss = []
        self.loss_GC_loss = []
        self.loss_feature_tmp = []

        for i in range(len(self.fake_image)):
            prob_c = self.netclassifier(self.fake_image[i])

            loss_C_tmp = self.criterionLoss(prob_c, self.label[i].long())
            loss_GC_tmp = nn.MultiLabelSoftMarginLoss()(prob_c, self.prob_tmp[i])


            self.loss_C_loss.append(loss_C_tmp)
            self.loss_GC_loss.append(loss_GC_tmp)

        self.loss_C_loss_all = None
        self.loss_GC_loss_all = None

        for i in range(len(self.fake_image)):
            if self.loss_C_loss_all is None:
                self.loss_C_loss_all = self.loss_C_loss[i]
                self.loss_GC_loss_all = self.loss_GC_loss[i]
            else:
                self.loss_C_loss_all += self.loss_C_loss[i]
                self.loss_GC_loss_all += self.loss_GC_loss[i]

        self.loss_C = (self.loss_C_loss_all*10 +  self.loss_GC_loss_all)
        self.loss_C.backward()





    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.forward()            # first call forward to calculate intermediate results

        # update netgenerator
        self.set_requires_grad(self.netgenerator, True)  # netclassifier requires no gradients when optimizing netgenerator
        self.set_requires_grad(self.netclassifier, False)
        for opt in self.optimizer_G:
            opt.zero_grad()
        self.backward_G()
        for opt in self.optimizer_G:
            opt.step()

        # update netclassifier
        self.set_requires_grad(self.netclassifier, True)
        self.optimizer_C.zero_grad()
        self.backward_C()
        self.optimizer_C.step()

    def val(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        print('-'*100)
        predlist = []
        scorelist = []
        targetlist = []

        pre, recall, accuracy, AUC_list = [], [], [], []
        for i in range(3):
            print(f'---this is the {i} generator model  self.data[{i}] 数据------')
            for j in range(len(self.val_data)+ 1):
                # print(f'---this is the self.val_data {j}    ------')
                _, _, val_pred_generator = self.netgenerator[i](self.val_data,self.val_label)

                # print('pred_generator type =',type(pred_generator))
                # print('pred_generator  =', pred_generator)
                target = self.val_label
                score = torch.softmax(val_pred_generator, dim=1)
                pred = val_pred_generator.argmax(dim=1, keepdim=True)

                targetcpu = target.long().cpu().numpy()
                predlist = np.append(predlist, pred.cpu().numpy())
                scorelist = np.append(scorelist, score.detach().cpu().numpy()[:, 1])
                targetlist = np.append(targetlist, targetcpu)

                # print('predlist = ',predlist)
                # print('scorelist = ', scorelist)
                # print('targetlist = ', targetlist)

            TP = ((predlist == 1) & (targetlist == 1)).sum()
            TN = ((predlist == 0) & (targetlist == 0)).sum()
            FN = ((predlist == 0) & (targetlist == 1)).sum()
            FP = ((predlist == 1) & (targetlist == 0)).sum()

            # print('TP=', TP, 'TN=', TN, 'FN=', FN, 'FP=', FP)

            p = TP / (TP + FP)
            # print('precision', p)
            r = TP / (TP + FN)
            # print('recall', r)

            F1 = 2 * r * p / (r + p)
            acc = (TP + TN) / (TP + TN + FP + FN)
            # print('f1', F1)
            # print('acc', acc)
            AUC = 0
            try:
                AUC = roc_auc_score(targetlist, scorelist)
                # print('AUC =', AUC)
            except ValueError:
                pass
            pre.append(p)
            recall.append(r)
            accuracy.append(acc)
            AUC_list.append(AUC)


        a = []
        # print('pre length = ', len(pre))
        for i in range(len(pre)):
            tmp = pre[i] + recall[i] + accuracy[i] + AUC_list[i]
            a.append(tmp)
        inde = a.index(max(a))
        # print('性能最好的权重的 index = ', inde)
        self.inde = inde




    def test_forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        print('-'*100)
        for i in range(len(self.data)):
            print(f'--------------------this is the {2} generator model------------------')
            # generator_loss
            # pred_generator = self.netclassifier(self.data)
            _, _, pred_generator = self.netgenerator[2](self.data,self.label)

            # print('pred_generator type =',type(pred_generator))
            # print('pred_generator  =', pred_generator)
            target = self.label
            score = F.softmax(pred_generator, dim=1)
            pred = pred_generator.argmax(dim=1, keepdim=True)
            self.correct += pred.eq(target.long().view_as(pred)).sum().item()

            self.targetcpu = target.long().cpu().numpy()
            self.predlist = np.append(self.predlist, pred.cpu().numpy())
            self.scorelist = np.append(self.scorelist, score.cpu().numpy()[:, 1])
            self.targetlist = np.append(self.targetlist, self.targetcpu)

            print('predlist = ',self.predlist)
            print('scorelist = ', self.scorelist)
            print('targetlist = ', self.targetlist)

        TP = ((self.predlist == 1) & (self.targetlist == 1)).sum()
        TN = ((self.predlist == 0) & (self.targetlist == 0)).sum()
        FN = ((self.predlist == 0) & (self.targetlist == 1)).sum()
        FP = ((self.predlist == 1) & (self.targetlist == 0)).sum()

        print('TP=', TP, 'TN=', TN, 'FN=', FN, 'FP=', FP)

        p = TP / (TP + FP)
        print('precision', p)
        r = TP / (TP + FN)
        print('recall', r)

        F1 = 2 * r * p / (r + p)
        acc = (TP + TN) / (TP + TN + FP + FN)
        print('f1', F1)
        print('acc', acc)
        try:
            AUC = roc_auc_score(self.targetlist, self.scorelist)
            print('AUC =', AUC)
        except ValueError:
            pass


    def save_list_to_txt(self):
        text_save_w('predlist-scorelist-targetlist.txt', [self.predlist,self.scorelist,self.targetlist])
        # text_save_w('predlist.txt', self.predlist)
        # text_save_w('label.txt', self.label)

