# BERT4ETH (PyTorch Version)

This is the PyTorch implementation for the paper [BERT4ETH: A Pre-trained Transformer for Ethereum Fraud Detection](https://dl.acm.org/doi/abs/10.1145/3543507.3583345), accepted by the ACM Web conference (WWW) 2023.

I have recovered the experiment results and am doing final check. (2023/11/29)

If you find this repository useful, please give us a star and cite our paper : ) Thank you!

## Getting Start

### Requirements

PyTorch > 1.12.0

### Preprocess dataset 

#### Step 1: Download dataset from Google Drive. 
* Transaction Dataset:
* * [Phishing Account](https://drive.google.com/file/d/11UAhLOcffzLyPhdsIqRuFsJNSqNvrNJf/view?usp=sharing)

* * [De-anonymization(ENS)](https://drive.google.com/file/d/1Yveis90jCx-nIA6pUL_4SUezMsVJr8dp/view?usp=sharing)

* * [De-anonymization(Tornado)](https://drive.google.com/file/d/1DMbPSZMSvTYMKUZg3oYKFrjPo2_jeeG4/view?usp=sharing)

* * [Normal Account](https://drive.google.com/file/d/1-htLUymg1UxDrXcI8tslU9wbn0E1vl9_/view?usp=sharing)

* [ERC-20 Log Dataset (all in one)](https://drive.google.com/file/d/1mB2Tf7tMq5ApKKOVdctaTh2UZzzrAVxq/view?usp=sharing)


#### Step 2: Unzip dataset under the directory of "BERT4ETH/Data/" 

```sh
cd BERT4ETH_PyTorch/data; # Labels are already included
unzip ...;
``` 

### Pre-training


#### Step 1: Transaction Sequence Generation

```sh
cd src;
python gen_seq.py --bizdate=bert4eth_exp
```


#### Step 2: Pre-train BERT4ETH 

```sh
python run_pretrain.py --bizdate="bert4eth_exp" \
                       --ckpt_dir="bert4eth_exp"
```

#### Step 3: Output Representation

```sh
python run_embed.py --bizdate="bert4eth_exp" \
                       --init_checkpoint="bert4eth_exp/xxx.pth"
```

### Evaluation 

#### Phishing Account Detection
```sh
cd eval
python phish_detection_mlp.py --input_dir="../outputs/xxx"
```

#### De-anonymization (ENS dataset)

```sh
python run_dean_ENS.py --metric=euclidean \
                       --init_checkpoint=bert4eth_exp/model_104000
```


### Fine-tuning for phishing account detection
```sh
  Will update later..
```



-----
## Citation

```
@inproceedings{hu2023bert4eth,
  title={BERT4ETH: A Pre-trained Transformer for Ethereum Fraud Detection},
  author={Hu, Sihao and Zhang, Zhen and Luo, Bingqiao and Lu, Shengliang and He, Bingsheng and Liu, Ling},
  booktitle={Proceedings of the ACM Web Conference 2023},
  pages={2189--2197},
  year={2023}
}
```

-----
## Q&A

If you have any questions, you can either open an issue or contact me (sihaohu@gatech.edu), and I will reply as soon as I see the issue or email.

