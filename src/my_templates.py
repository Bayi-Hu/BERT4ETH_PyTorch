def set_template(args):
    if args.template is None:
        return

    args.mode = 'train'
    # args.dataset_code = 'ml-' + input('Input 1 for ml-1m, 20 for ml-20m: ') + 'm'
    args.dataset_code = 'ml-1m'
    args.min_rating = 0 if args.dataset_code == 'ml-1m' else 4
    args.min_uc = 5
    args.min_sc = 0
    args.split = 'leave_one_out'

    args.dataloader_code = 'bert'
    batch = 128
    args.train_batch_size = batch
    args.val_batch_size = batch
    args.test_batch_size = batch

    args.train_negative_sampler_code = 'random'
    args.train_negative_sample_size = 0
    args.train_negative_sampling_seed = 0
    args.test_negative_sampler_code = 'random'
    args.test_negative_sample_size = 100
    args.test_negative_sampling_seed = 98765

    args.trainer_code = 'bert'
    # args.device = 'cuda'
    args.device = 'cpu'
    args.num_gpu = 1
    args.device_idx = '0'
    args.optimizer = 'Adam'
    args.lr = 0.001
    args.enable_lr_schedule = True
    args.decay_step = 25
    args.gamma = 1.0
    args.num_epochs = 100 if args.dataset_code == 'ml-1m' else 200
    args.metric_ks = [1, 5, 10, 20, 50, 100]
    args.best_metric = 'NDCG@10'

    args.model_code = 'bert'
    args.model_init_seed = 0

    args.bert_dropout = 0.1
    args.bert_hidden_units = 256
    args.bert_mask_prob = 0.15
    args.bert_max_len = 100
    args.bert_num_blocks = 2
    args.bert_num_heads = 4
