import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import fix_random_seed_as
from tqdm import tqdm
import os

class PositionalEmbedding(nn.Module):

    def __init__(self, max_len, d_model):
        super().__init__()

        # Compute the positional encodings once in log space.
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)

class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size=512):
        super().__init__(3, embed_size, padding_idx=0)

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)
        self.weight.data.uniform_(-0.02, 0.02) # set initialization range

class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, args):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()

        self.token_embed = TokenEmbedding(vocab_size=args.vocab_size, embed_size=args.hidden_size)
        self.value_embed = TokenEmbedding(vocab_size=15 , embed_size=args.hidden_size)
        self.count_embed = TokenEmbedding(vocab_size=15, embed_size=args.hidden_size)
        self.position_embed = TokenEmbedding(vocab_size=args.max_seq_length , embed_size=args.hidden_size)
        self.io_embed = TokenEmbedding(vocab_size=3, embed_size=args.hidden_size)
        self.gas_embed = TokenEmbedding(vocab_size=15, embed_size=args.hidden_size)

        self.dropout = nn.Dropout(p=args.hidden_dropout_prob)

    def forward(self, args):
        input_ids, counts, values, io_flags, positions, gas_fee = args
        x = self.token_embed(input_ids) + self.count_embed(counts) + self.position_embed(positions) \
            + self.io_embed(io_flags) + self.value_embed(values) + self.gas_embed(gas_fee)
        return self.dropout(x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


class BERT4ETH(nn.Module):
    def __init__(self, args):
        super().__init__()

        fix_random_seed_as(args.model_init_seed)
        # self.init_weights()
        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(args)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(args.hidden_size,
                              args.num_attention_heads,
                              args.hidden_size * 4,
                              args.hidden_dropout_prob)
             for _ in range(args.num_hidden_layers)])

        # self.out = nn.Linear(config["hidden_size"], config["vocab_size"])

    def forward(self, args):
        input_ids = args[0]
        mask = (input_ids > 0).unsqueeze(1).repeat(1, input_ids.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(args)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x

    def init_weights(self):
        pass


"""
MLP for downstream task
"""

class MLP(nn.Module):
    def __init__(self, dataloader=None):
        super(MLP, self).__init__()
        self.dataloader = dataloader
        self.input_dim = 64
        self.hidden_dim = 256
        self.num_epochs = 2
        self.lr = 5e-4
        self.device = "cuda"
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim).to(self.device)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim).to(self.device)
        self.out_layer = nn.Linear(self.hidden_dim, 1).to(self.device)
        self.optimizer = torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=self.lr)

    def forward(self, x):
        dnn1 = F.relu(self.fc1(x))
        dnn2 = F.relu(self.fc2(dnn1))
        # logits = torch.squeeze(self.out_layer(dnn1+dnn2), -1)
        logits =self.out_layer(dnn1+dnn2)

        return logits

    def fit(self):
        self.train()
        accum_iter = 0
        for epoch in range(self.num_epochs):
            # for each epoch
            tqdm_dataloader = tqdm(self.dataloader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = [x for x in batch]

                X_batch = batch[0]
                y_batch = batch[1]
                X_batch = torch.tensor(X_batch).to(self.device)
                y_batch = torch.tensor(y_batch).to(self.device)

                self.optimizer.zero_grad()
                logits = self.forward(X_batch)
                loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), y_batch.view(-1))
                loss.backward()
                self.optimizer.step()
                tqdm_dataloader.set_description(
                    'Epoch {}, loss {:.3f} '.format(epoch, loss.item())
                )
                batch_size = X_batch.shape[0]
                accum_iter += batch_size

        return

    def predict_proba(self, X_test):
        X_test = torch.tensor(X_test).to(self.device)
        logits = self.forward(X_test)
        y_test = torch.softmax(logits, dim=1)[:,1].detach().cpu().numpy()

        return y_test


"""
Finetuning Model
"""

class FineTuneModel(nn.Module):
    def __init__(self, args, downstream_net=MLP()):
        super().__init__()

        self.pretrain_model = BERT4ETH(args)

        self.downstream_net=downstream_net
        # self.out = nn.Linear(config["hidden_size"], config["vocab_size"])

        self.init_pretrain(ckpt_dir=args.pre_train_ckpt_dir)

    def init_pretrain(self, ckpt_dir):
        if not os.path.isdir(ckpt_dir):
            raise FileNotFoundError("Must have pretrain model")

        content = os.listdir(ckpt_dir)
        full_path = [os.path.join(ckpt_dir, x)  for x in content]
        dir_content = sorted(full_path, key=lambda t: os.stat(t).st_mtime)

        if not len(dir_content):
            raise FileNotFoundError("Must have pretrain model")

        ckpt_dir = dir_content[-1]

        pre_train_ckpt = torch.load(ckpt_dir)
        self.pretrain_model.load_state_dict(pre_train_ckpt)

    def forward(self, args):

        # embedding the indexed sequence to sequence of vectors
        x = self.pretrain_model.forward(args)
        embed = x[:, 0, :]
        x = self.downstream_net.forward(embed)

        return x

