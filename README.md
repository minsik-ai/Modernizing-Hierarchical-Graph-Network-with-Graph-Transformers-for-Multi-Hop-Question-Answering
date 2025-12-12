# Modernizing Hierarchical Graph Network with Graph Transformers for Multi-Hop Question Answering (ModernHGN)

Hierarchical Graph Network (HGN) is a multi-hop reasoning pathway manifested in a hierarchical graph structure. Built and proposed by Microsoft Dynamics 365 AI Research, HGN aggregates clues from sources of differing granularity levels (e.g., paragraphs, sentences, entities). It effectively incorporates concepts from Graph Attention Network (GAT), Gated Attention and BiDAF (Seo et al., 2017) to construct a multi-hop reasoning graph network model on HotpotQA.

I have implemented it with the [Deep Graph Library (DGL)](https://www.dgl.ai/), which provides well-implemented versions of varous graph algorithms (e.g., GCN, GraphSAGE, GAT, Jumping Knowledge Network, etc.). By using `dgl.heterograph` and `dgl.nn.pytorch.HeteroGraphConv`, I was easily able to construct nodes of differing granularity levels and their update algorithm.

- Dataset: HotpotQA 

## Dependencies

```bash
pip install -r requirements.txt
```

or, if you prefer conda,
```bash
conda env create -f environment.yml
```

## Path Configurations

Change `--model_dir` and `--data_dir` to preferred `MODEL_OUTPUT_PATH/distilbert-base-uncased` and `DATA_DOWNLOAD_PATH` respectively.

Change `mkdir` and `cd` path in `download_hotpot.sh` to `DATA_DOWNLOAD_PATH`.

Install distilbert before proceeding.

```bash
cd MODEL_OUTPUT_PATH
git lfs clone https://huggingface.co/distilbert-base-uncased
```

## Usage

Download hotpot dataset
```bash
$ ./scripts/download_hotpot.sh
```

Training the HGN model:
```bash
$ ./scripts/train.sh --do_train
```

Evaluating the HGN model:
```bash
$ ./scripts/train.sh --do_eval
```

## References
- [Hierarchical Graph Network (paper)](https://arxiv.org/pdf/1911.03631.pdf)
