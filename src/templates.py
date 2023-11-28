import math
def set_template(args):
    # if args.template is None:
    #     return

    args.train_batch_size = 256
    args.eval_batch_size = 1024

    # args.device = 'cuda'
    args.device = 'cpu'
    args.num_gpu = 1
    args.device_idx = '0'
    args.optimizer = 'Adam'
    args.lr = 0.001
    args.enable_lr_schedule = True
    args.decay_step = 25
    args.gamma = 1.0
    args.num_epochs = 5
    args.model_init_seed = 0

    # model configuration
    args.hidden_act = "gelu"
    args.hidden_size = 64
    args.initializer_range = 0.02
    args.num_hidden_layers = 2
    args.num_attention_heads = 2
    args.vocab_size = 500000
    args.max_seq_length = 100
    args.hidden_dropout_prob = 0.2
    args.attention_probs_dropout_prob = 0.2

    args.max_predictions_per_seq = math.ceil(args.max_seq_length * args.masked_lm_prob)
    args.sliding_step = round(args.max_seq_length * 0.6)
