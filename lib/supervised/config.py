from argparse import ArgumentParser
import torch
import warnings
BATCHNORM_MOMENTUM = 0.01

def configure_torch_settings():
    # 使用新的 API 替代已弃用的方法
    cuda_available = torch.backends.cuda.is_built()
    cudnn_available = torch.backends.cudnn.is_available()
    mps_available = torch.backends.mps.is_built()
    mkldnn_available = torch.backends.mkldnn.is_available()
    
    # 忽略嵌套张量原型警告
    warnings.filterwarnings(
        "ignore", 
        message="The PyTorch API of nested tensors is in prototype stage"
    )
    
    return {
        'cuda': cuda_available,
        'cudnn': cudnn_available,
        'mps': mps_available,
        'mkldnn': mkldnn_available
    }

torch_settings = configure_torch_settings()

class Config(object):
    """Wrapper class for model hyperparameters."""

    def __init__(self):
        """
        Defaults
        """
        self.mode = None
        self.save_path = None
        self.model_path = None
        self.data_path = None
        self.datasize = None
        self.ckpt = None
        self.optimizer = None
        self.bce_loss = True
        self.lr = 1e-5
        self.enc_layer = 1
        self.dec_layer = 3
        self.nepoch = 10
        self.results_path = None
        self.method_name = None
        self.task_name = "sga"
        
        self.max_window = 5
        self.brownian_size = 1
        self.ode_ratio = 1.0
        self.sde_ratio = 1.0
        self.bbox_ratio = 0.1
        self.features_path = None
        self.additional_data_path = None
        
        self.baseline_context = 3
        self.max_future = 5
        
        self.hp_recon_loss = 1.0
        
        self.use_raw_data = False
        
        self.parser = self.setup_parser()
        self.args = vars(self.parser.parse_args())
        self.__dict__.update(self.args)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def setup_parser(self):
        """
        Sets up an argument parser
        :return:
        """
        parser = ArgumentParser(description='training code')
        parser.add_argument('--vid_sgg_method', dest='method', help='Method dsg_detr/sttran/tempura', default='dsg_detr',
                            type=str)
        parser.add_argument('--mode', dest='mode', help='predcls/sgcls/sgdet', default='predcls', type=str)
        parser.add_argument('--save_path', default='SceneSayer/checkpoints', type=str)
        parser.add_argument('--model_path', default=None, type=str)
        parser.add_argument('--llama_path', default=None, type=str)
        parser.add_argument('--lora_path', default=None, type=str)
        parser.add_argument('--classifier_path', default=None, type=str)
        parser.add_argument('--method_name', default='NeuralODE', type=str)
        parser.add_argument('--results_path', default='results', type=str)
        parser.add_argument('--max_window', default=5, type=int)
        parser.add_argument('--brownian_size', default=1, type=int)
        parser.add_argument('--ode_ratio', default=1.0, type=float)
        parser.add_argument('--sde_ratio', default=1.0, type=float)
        parser.add_argument('--bbox_ratio', default=0.1, type=float)
        parser.add_argument('--features_path', default=None, type=str)
        parser.add_argument('--additional_data_path', default=None, type=str)
        parser.add_argument('--baseline_context', default=3, type=int)
        parser.add_argument('--max_future', default=5, type=int)
        #------------scripts----------------
        parser.add_argument('--use_raw_data', action='store_true')
        parser.add_argument('--data_path', default='data/charades/ag', type=str)
        parser.add_argument('--script_require', action='store_true')
        parser.add_argument('--object_required', action='store_true')
        parser.add_argument('--relation_required', action='store_true')
        parser.add_argument('--object_align_weight', default=1, type=float)
        #--------------end------------------
        parser.add_argument('--datasize', dest='datasize', help='mini dataset or whole', default='large', type=str)
        parser.add_argument('--ckpt', dest='ckpt', help='checkpoint', default=None, type=str)
        parser.add_argument('--optimizer', help='adamw/adam/sgd', default='adamw', type=str)
        parser.add_argument('--lr', dest='lr', help='learning rate', default=1e-5, type=float)
        parser.add_argument('--nepoch', help='epoch number', default=10, type=float)
        parser.add_argument('--enc_layer', dest='enc_layer', help='spatial encoder layer', default=1, type=int)
        parser.add_argument('--dec_layer', dest='dec_layer', help='temporal decoder layer', default=3, type=int)
        parser.add_argument('--bce_loss', action='store_true')
        parser.add_argument('--modified_gt', action='store_true')
        parser.add_argument("--task_name", default="sga", type=str)
        #--------------distribution———————————    
        parser.add_argument('--distributed', action='store_true', help="Use distributed training")
        parser.add_argument('--local_rank', type=int, default=0, help="Local rank for distributed training")
        parser.add_argument('--batch_size', type=int, default=1, help="Batch size")
        parser.add_argument('--num_epochs', type=int, default=10, help="Number of epochs")
        parser.add_argument('--threshold', type=float, default=0.9, help="Number of epochs")
        parser.add_argument('--fraction', type=float, default=None, help="Number of epochs")
        parser.add_argument('--stage1_tokens', type=int, default=256, help="Number of epochs")
        parser.add_argument('--use_llm', action='store_true', help="Use llm training")
        parser.add_argument('--use_classify_head', action='store_true', help="Use llm training")
        parser.add_argument('--use_gt_anno', action='store_true', help="Use llm training")
        parser.add_argument('--use_fusion', action='store_true', help="Use llm fusion training")
        parser.add_argument('--label_smoothing', action='store_true', help="Use llm fusion training")
        parser.add_argument('--two_stage', action='store_true', help="Use llm fusion training")
        parser.add_argument('--three_stage', action='store_true', help="Use llm fusion training")
        parser.add_argument('--video_id_required', action='store_true', help="Use llm fusion training") #lora_path_stage1
        parser.add_argument('--use_stage0', action='store_true', help="Use llm fusion training")
        parser.add_argument('--not_use_merge', action='store_true', help="Use llm fusion training")
        parser.add_argument('--length_of_segments', default=None, type=int)
        parser.add_argument('--model_path_stage0', default=None, type=str)
        parser.add_argument('--lora_path_stage0', default=None, type=str)
        parser.add_argument('--lora_path_stage1', default=None, type=str)
        parser.add_argument('--lora_path_stage2', default=None, type=str)
        return parser
