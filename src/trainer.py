import numpy as np

from config import STATE_DICT_KEY, OPTIMIZER_STATE_DICT_KEY
from utils import AverageMeterSet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm
import torch.nn.functional as F
import os

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
    return neg_ids + 1 + 3

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
    def __init__(self, args, vocab, model, data_loader):
        self.args = args
        self.device = args.device
        self.vocab = vocab
        self.model = model.to(self.device)
        self.is_parallel = args.num_gpu > 1
        if self.is_parallel:
            self.model = nn.DataParallel(self.model)

        self.data_loader = data_loader
        self.optimizer = self._create_optimizer()
        if args.enable_lr_schedule:
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.decay_step, gamma=args.gamma)

        self.num_epochs = args.num_epochs

        # Parameters for pre-training task, not related to the model
        self.dense = nn.Linear(args.hidden_size, args.hidden_size).to(self.device)
        self.transform_act_fn = F.gelu
        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=1e-12).to(self.device)
        # self.bias = torch.nn.Parameter(torch.zeros(logits.shape[-1])).to(self.device)
        # self.output_bias = nn.Parameter(torch.zeros())

    def calculate_loss(self, batch):
        address_id = batch[0]
        input_ids = batch[1]
        counts = batch[2]
        values = batch[3]
        io_flags = batch[4]
        positions = batch[5]
        input_mask = batch[6]
        labels = batch[7]

        # seqs, labels = batch
        # h = self.model(input_ids)  # B x T x V
        h = self.model(input_ids, counts, values, io_flags, positions).to(self.device)
        # here forward we should also include other features.

        # Transformation
        input_tensor = self.dense(h)
        input_tensor = self.transform_act_fn(input_tensor)
        input_tensor = self.LayerNorm(input_tensor)


        neg_ids = negative_sample(self.args.neg_strategy,
                                  self.vocab,
                                  self.args.neg_sample_num).to(self.device)

        # labels = labels.view(-1)
        label_mask = torch.where(labels > 0, 1, 0)
        labels = torch.where(labels > 0, labels, 0)

        pos_output_weights = self.model.embedding.token_embed(labels) # positive embedding
        neg_output_weights = self.model.embedding.token_embed(neg_ids) # negative embedding

        pos_logits = torch.sum(input_tensor * pos_output_weights, dim=-1).unsqueeze(-1)
        neg_logits = torch.matmul(input_tensor, neg_output_weights.t())

        logits = torch.cat([pos_logits, neg_logits], dim=2)
        # print("================")
        # print(logits.shape)

        log_probs = torch.log_softmax(logits, -1)
        per_example_loss = -log_probs[:,:,0]

        numerator = torch.sum(label_mask * per_example_loss)
        denominator = torch.sum(label_mask) + 1e-5
        loss = numerator / denominator

        return loss

    def train(self):
        accum_iter = 0
        for epoch in range(self.num_epochs):
            accum_iter = self.train_one_epoch(epoch, accum_iter)
            self.save_model(epoch, self.args.ckpt_dir)

    def load(self, ckpt_dir):
        self.model.load_state_dict(torch.load(ckpt_dir))

    def infer_embedding(self, ):
        self.model.eval()
        tqdm_dataloader = tqdm(self.data_loader)
        embedding_list = []
        address_list = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.device) for x in batch]

                address = batch[0]
                input_ids = batch[1]
                counts = batch[2]
                values = batch[3]
                io_flags = batch[4]
                positions = batch[5]
                h = self.model(input_ids, counts, values, io_flags, positions).to(self.device)

                cls_embedding = h[:,0,:]
                # mean embedding
                # mean_embedding = torch.mean(h, dim=1)
                embedding_list.append(cls_embedding)
                address_ids = address.squeeze().tolist()

                addresses = self.vocab.convert_ids_to_tokens(address_ids)
                address_list += addresses

        embedding_list = torch.cat(embedding_list, dim=0)
        # mean pooling
        address_to_embedding = {}
        for i in range(len(address_list)):
            address = address_list[i]
            embedding = embedding_list[i]
            try:
                address_to_embedding[address].append(embedding.expand(size=[1,-1]))
            except:
                address_to_embedding[address] = [embedding.expand(size=[1,-1])]

        embedding_list = []
        for address, embeds in address_to_embedding.items():
            address_list.append(address)
            if len(embeds) > 1:
                embeds = torch.cat(embeds, dim=0)
                embedding_list.append(torch.mean(embeds, dim=0).expand(size=[1,-1]))
            else:
                embedding_list.append(embeds[0])
            # final embedding table
        address_array = np.array(address_list)
        embedding_array = np.array(torch.cat(embedding_list, dim=0))
        return address_array, embedding_array

    def train_one_epoch(self, epoch, accum_iter):
        self.model.train()
        if self.args.enable_lr_schedule:
            self.lr_scheduler.step()

        average_meter_set = AverageMeterSet()
        tqdm_dataloader = tqdm(self.data_loader)

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

    def save_model(self, epoch, ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_dir = os.path.join(ckpt_dir, "model_" + str(epoch)) + ".pth"
        print("Saving model to:", ckpt_dir)
        torch.save(self.model.state_dict(), ckpt_dir)

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
