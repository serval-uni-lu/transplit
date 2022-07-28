import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

fix_seed = 2022
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='[Transplit] Efficient Energy Load Forecasting at Scale')

# basic config
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='Transplit',
                    help='model name, options: [Transplit, Autoformer, Informer, Reformer, Transformer]')

# data loader
parser.add_argument('--data', type=str, default='custom', help='dataset type')
parser.add_argument('--root_path', type=str, default='./datasets/electricity/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='electricity.csv', help='data file')
parser.add_argument('--features', type=str, default='SA',
                    help=('forecasting task, options:[M, MS, S, SA]; '
                          'M:multivariate predict multivariate, '
                          'MS:multivariate predict univariate, '
                          'S:univariate predict univariate, '
                          'SA:univariate predict univariate (use all features)')
                    )
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help=('freq for time features encoding, '
                          'options: [s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                          'you can also use more detailed freq like 15min or 3h')
                    )
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location to save model checkpoints')
parser.add_argument('--save_every', type=int, default=10000, help='save model every x iterations during one epoch')
parser.add_argument('--load_checkpoint', type=str, default='',
                    help='if is_training is false, the model checkpoint to load and test')
parser.add_argument('--scale', type=bool, default=True, help='whether to standardize the data for training')
parser.add_argument('--unscale', type=bool, default=True,
                    help='whether the output data recovers the original scale for testing')

# forecasting task
parser.add_argument('--seq_len', type=int, default=720, help='input sequence length')
parser.add_argument('--pred_len', type=int, default=720, help='prediction sequence length')

# model define
parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
parser.add_argument('--c_out', type=int, default=1, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--period', type=int, default=24, help='size of one period, for Transplit')
parser.add_argument('--n_filters', type=int, default=512, help='num of filters, for Transplit')
parser.add_argument('--label_len', type=int, default=72, help='start token length, for Autoformer')
parser.add_argument('--bucket_size', type=int, default=4, help='for Reformer')
parser.add_argument('--n_hashes', type=int, default=4, help='for Reformer')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=3, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--do_nothing', action='store_true',
                    help='useful in IPython, allows to setup the model, then exit to use it')

# optimization
parser.add_argument('--num_workers', type=int, default=1, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=3, help='train epochs')
parser.add_argument('--batch_size', type=int, default=8, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=2, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multiple gpus')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

Exp = Exp_Main

setting_ = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_'.format(
    args.model_id,
    args.model,
    args.data,
    args.features,
    args.seq_len,
    args.label_len,
    args.pred_len,
    args.d_model,
    args.n_heads,
    args.e_layers,
    args.d_layers,
    args.d_ff,
    args.factor,
    args.embed,
    args.distil,
    args.des
)

# ---- Training ---- #
if args.is_training:
    for ii in range(args.itr):
        # setting record of experiments
        setting = setting_ + str(ii)

        exp = Exp(args)  # set experiments
        print(f'-------- start training: {setting} --------')
        exp.train(setting)

        print(f'-------- testing: {setting} --------')
        exp.test(setting)

        torch.cuda.empty_cache()

# ---- Testing ---- #
else:
    setting = setting_ + '0'

    exp = Exp(args)  # set experiments
    print(f'-------- testing: {setting} --------')
    exp.test(setting, checkpoint=args.load_checkpoint)
    torch.cuda.empty_cache()
