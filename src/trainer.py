from config import STATE_DICT_KEY, OPTIMIZER_STATE_DICT_KEY
from utils import AverageMeterSet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm
import torch.nn.functional as F

def negative_sample(neg_strategy, vocab, sample_num):

    word_num = len(vocab.vocab_words) - 3
    if neg_strategy == "uniform":
        # Uniform negative sampling
        weights = torch.ones(word_num)
    elif neg_strategy == "zip":
        # Log-uniform (Zipfian) negative sampling
        weights = 1 / torch.arange(1., word_num + 1)
    elif neg_strategy == "freq":
        # Frequency-based negative sampling
        weights = torch.tensor(list(map(lambda x: pow(x, 1 / 1), vocab.frequency[:-3])), dtype=torch.float)
    else:
        raise ValueError("Please select correct negative sampling strategy: uniform, zip, freq.")

    sampler = Categorical(weights)
    neg_ids = sampler.sample((sample_num,))
    return neg_ids + 1

def gather_indexes(sequence_tensor, positions):
    """
    Gathers the vectors at the specific positions over a minibatch.
    """
    batch_size, seq_length, width = sequence_tensor.size()
    flat_offsets = torch.arange(0, batch_size, dtype=torch.long) * seq_length
    flat_offsets = flat_offsets.unsqueeze(-1)  # reshape to [batch_size, 1]
    flat_positions = (positions + flat_offsets).view(-1)
    flat_sequence_tensor = sequence_tensor.view(batch_size * seq_length, width)
    output_tensor = flat_sequence_tensor.index_select(0, flat_positions)
    return output_tensor


class BERT4ETHTrainer:
    def __init__(self, args, vocab, model, train_loader):
        self.args = args
        self.device = args.device
        self.vocab = vocab
        self.model = model.to(self.device)
        self.is_parallel = args.num_gpu > 1
        if self.is_parallel:
            self.model = nn.DataParallel(self.model)

        self.train_loader = train_loader
        self.optimizer = self._create_optimizer()
        if args.enable_lr_schedule:
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.decay_step, gamma=args.gamma)

        self.num_epochs = args.num_epochs
        self.metric_ks = args.metric_ks
        self.best_metric = args.best_metric

        # for loss calculation
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.transform_act_fn = F.gelu
        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=1e-12)
        # self.output_bias = nn.Parameter(torch.zeros())

    @classmethod
    def code(cls):
        return 'bert'

    def calculate_loss(self, batch):
        input_ids = batch[0]
        counts = batch[1]
        values = batch[2]
        io_flags = batch[3]
        positions = batch[4]
        input_mask = batch[5]
        labels = batch[6]

        # seqs, labels = batch
        h = self.model(input_ids)  # B x T x V

        neg_ids = negative_sample(self.args.neg_strategy,
                                  self.vocab,
                                  self.args.sample_num)



        masked_h = gather_indexes(h, input_mask)


        # Transformation
        input_tensor = self.dense(input_tensor)
        input_tensor = self.transform_act_fn(input_tensor)
        input_tensor = self.LayerNorm(input_tensor)

        # Get embeddings
        label_ids = labels.view(-1)
        label_weights = label_weights.view(-1)
        pos_output_weights = self.output_weights(label_ids)  # Assuming output_weights is a nn.Embedding layer
        neg_output_weights = self.output_weights(neg_ids)

        # Compute logits
        pos_logits = torch.sum(input_tensor * pos_output_weights, dim=-1).unsqueeze(1)
        neg_logits = torch.matmul(input_tensor, neg_output_weights.t())

        logits = torch.cat([pos_logits, neg_logits], dim=1)
        logits += self.output_bias

        log_probs = F.log_softmax(logits, dim=-1)
        per_example_loss = -log_probs[:, 0]
        numerator = torch.sum(label_weights * per_example_loss)
        denominator = torch.sum(label_weights) + 1e-5
        loss = numerator / denominator

        return loss, per_example_loss, log_probs

        # do negative sampling here

        logits = logits.view(-1, logits.size(-1))  # (B*T) x V
        labels = labels.view(-1)  # B*T
        # do negative sampling here.

        loss = nn.CrossEntropyLoss(logits, labels, ignore_index=-1)
        return loss


    def train(self):
        accum_iter = 0
        for epoch in range(self.num_epochs):
            accum_iter = self.train_one_epoch(epoch, accum_iter)

    def train_one_epoch(self, epoch, accum_iter):
        self.model.train()
        if self.args.enable_lr_schedule:
            self.lr_scheduler.step()

        average_meter_set = AverageMeterSet()
        tqdm_dataloader = tqdm(self.train_loader)

        for batch_idx, batch in enumerate(tqdm_dataloader):

            batch_size = batch[0].shape[0]
            batch = [x.to(self.device) for x in batch]

            self.optimizer.zero_grad()
            loss = self.calculate_loss(batch)
            loss.backward()

            self.optimizer.step()

            average_meter_set.update('loss', loss.item())
            tqdm_dataloader.set_description(
                'Epoch {}, loss {:.3f} '.format(epoch+1, average_meter_set['loss'].avg))

            accum_iter += batch_size

        return accum_iter

    def _create_optimizer(self):
        args = self.args
        if args.optimizer.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer.lower() == 'sgd':
            return optim.SGD(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
        else:
            raise ValueError

    def _create_state_dict(self):
        return {
            STATE_DICT_KEY: self.model.module.state_dict() if self.is_parallel else self.model.state_dict(),
            OPTIMIZER_STATE_DICT_KEY: self.optimizer.state_dict(),
        }
