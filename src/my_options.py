from my_templates import set_template
import argparse

parser = argparse.ArgumentParser(description='BERT4ETH')

################
# Top Level
################
parser.add_argument('--mode', type=str, default='train', choices=['train'])
parser.add_argument('--template', type=str, default=None)

################
# Test
################
parser.add_argument('--test_model_path', type=str, default=None)

################
# Dataset
################
parser.add_argument('--dataset_split_seed', type=int, default=98765)
parser.add_argument('--eval_set_size', type=int, default=500, 
                    help='Size of val and test set. 500 for ML-1m and 10000 for ML-20m recommended')

################
# Dataloader
################
parser.add_argument('--dataloader_code', type=str, default='bert')
parser.add_argument('--dataloader_random_seed', type=float, default=12345)
parser.add_argument('--train_batch_size', type=int, default=64)

################
# NegativeSampler
################
parser.add_argument('--train_negative_sampler_code', type=str, default='random', choices=['popular', 'random'],
                    help='Method to sample negative items for training. Not used in bert')
parser.add_argument('--train_negative_sample_size', type=int, default=100)
parser.add_argument('--train_negative_sampling_seed', type=int, default=None)

################
# Trainer
################
parser.add_argument('--trainer_code', type=str, default='bert')
# device #
parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
parser.add_argument('--num_gpu', type=int, default=1)
parser.add_argument('--device_idx', type=str, default='0')
# optimizer #
parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam'])
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='l2 regularization')
parser.add_argument('--momentum', type=float, default=None, help='SGD momentum')
# lr scheduler #
parser.add_argument('--decay_step', type=int, default=15, help='Decay step for StepLR')
parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for StepLR')
# epochs #
parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs for training')
# logger #
parser.add_argument('--log_period_as_iter', type=int, default=12800)
# evaluation #
parser.add_argument('--metric_ks', nargs='+', type=int, default=[10, 20, 50], help='ks for Metric@k')
parser.add_argument('--best_metric', type=str, default='NDCG@10', help='Metric for determining the best model')

################
# Model
################
parser.add_argument('--model_code', type=str, default='bert')
parser.add_argument('--model_init_seed', type=int, default=54321)
# BERT4ETH #
parser.add_argument('--bert_hidden_units', type=int, default=64, help='Size of hidden vectors (d_model)')
parser.add_argument('--bert_num_blocks', type=int, default=8, help='Number of transformer layers')
parser.add_argument('--bert_num_heads', type=int, default=2, help='Number of heads for multi-attention')
parser.add_argument('--bert_dropout', type=float, default=0.2, help='Dropout probability to use throughout the model')
parser.add_argument('--masked_lm_prob', type=float, default=0.8, help='Masked LM probability.')


################
# Experiment
################
parser.add_argument('--experiment_dir', type=str, default='experiments')
parser.add_argument('--experiment_description', type=str, default='test')

parser.add_argument('--max_seq_length', type=int, default=100, help='max sequence length.')
parser.add_argument('--do_eval', action='store_true', help='Whether to do evaluation.')
parser.add_argument('--do_embed', action='store_true', default=True, help='Whether to do embedding.')
parser.add_argument('--dupe_factor', type=int, default=10, help='Number of times to duplicate the input data (with different masks).')
parser.add_argument('--data_dir', type=str, default='./inter_data/', help='data dir.')
parser.add_argument('--vocab_filename', type=str, default='vocab', help='vocab filename')
parser.add_argument('--bizdate', type=str, default="local_test", help='the signature of running experiments')

################
args = parser.parse_args()
set_template(args)
