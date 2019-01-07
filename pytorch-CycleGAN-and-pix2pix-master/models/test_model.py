from .base_model import BaseModel
from . import networks
from .cycle_gan_model import CycleGANModel
import torch


class TestModel(BaseModel):
    def name(self):
        return 'TestModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        assert not is_train, 'TestModel cannot be used in train mode'
        parser = CycleGANModel.modify_commandline_options(parser, is_train=False)
        parser.set_defaults(dataset_mode='single')

        parser.add_argument('--model_suffix', type=str, default='',
                            help='In checkpoints_dir, [which_epoch]_net_G[model_suffix].pth will'
                            ' be loaded as the generator of TestModel')

        return parser

    def initialize(self, opt):
        assert(not opt.isTrain)
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = []
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'real_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        self.model_names = ['D_A' + opt.model_suffix]

        use_sigmoid = opt.no_lsgan
        self.netD_A = networks.define_D(opt.input_nc, opt.which_model_netD, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)

        # assigns the model to self.netG_[suffix] so that it can be loaded
        # please see BaseModel.load_networks
        setattr(self, 'netD_A' + opt.model_suffix, self.netD_A)

    def set_input(self, input):
        # we need to use single_dataset mode
        self.real_A = input['A'].to(self.device)
        self.image_paths = input['A_paths'] 

    def backward_D_A(self):
        self.pred_real_A = self.netD_A(self.real_A)
        Softmax = torch.nn.Softmax()
        self.pred_real_A = Softmax(self.pred_real_A)
        #predicted = torch.mean(self.pred_real_A.data)
        predicted_cpu = self.pred_real_A.cpu()
        predicted_array = predicted_cpu.detach().numpy()
        return predicted_array
        #print(self.pred_real_A)
