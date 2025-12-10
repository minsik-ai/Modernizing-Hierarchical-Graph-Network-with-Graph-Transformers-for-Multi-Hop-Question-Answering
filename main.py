import argparse

from trainer import Trainer
from utils import init_logger, load_tokenizer, set_seed, MODEL_CLASSES, MODEL_PATH_MAP
from data_loader import load_and_cache_examples


def main(args):
    init_logger()
    set_seed(args)

    tokenizer = load_tokenizer(args)
    train_dataset = dev_dataset = None
    if args.do_train:
        train_dataset, all_graph_out = load_and_cache_examples(args, tokenizer, mode="train")
    if args.do_eval:
        dev_dataset, all_graph_out = load_and_cache_examples(args, tokenizer, mode="dev")
    # test_dataset = load_and_cache_examples(args, tokenizer, mode="test")

    trainer = Trainer(args, all_graph_out, train_dataset, dev_dataset)

    if args.do_train:
        trainer.train()
        trainer.save_model()
        # Evaluate on train data immediately after training (no load)
        print("\n=== Evaluating on TRAIN data (without reload) ===")
        trainer.evaluate("train")

    if args.do_eval:
        trainer.load_model()
        # Try evaluating on train data to check if model learned
        # trainer.evaluate("train")  # Uncomment to test on train
        trainer.evaluate("dev")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", default="para_select", type=str, help="Task: `Paragraph Selection (para_select)` or `Train Model (train_model)`")  # Can be `opspam`, `yelp` or `amazon`
    parser.add_argument("--model_dir", default="/nlp/scr/minsik/hgn_dgl/models/distilbert-base-uncased/", type=str, help="Path to save, load model")
    parser.add_argument("--data_dir", default="/nlp/scr/minsik/hgn_dgl/data", type=str, help="The input data dir")
    parser.add_argument("--train_file", default="hotpot_train_v1.1.json", type=str, help="Train file")
    parser.add_argument("--dev_file", default="hotpot_dev_distractor_v1.json", type=str, help="Dev file (distractor & full_wiki)")
    parser.add_argument("--test_file", default="test.csv", type=str, help="Test file")

    parser.add_argument("--model_type", default="bert", type=str, help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))

    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument("--train_batch_size", default=1, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=1, type=int, help="Batch size for evaluation.")
    parser.add_argument("--max_seq_len", default=512, type=int, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=0.1, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=200, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=100, help="Save checkpoint every X updates steps.")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument('--logger', action="store_true", help="Activate neptune logger")

    parser.add_argument("--num_entities", default=60, type=int, help="Number of entities in HGN graph")
    parser.add_argument("--num_sentences", default=40, type=int, help="Number of sentences in HGN graph")
    parser.add_argument("--num_paragraphs", default=4, type=int, help="Number of paragraphs in HGN graph")

    # Ablation study arguments - Attention mechanisms
    parser.add_argument("--use_bi_attention", action="store_true", default=True, help="Use BiAttention layer")
    parser.add_argument("--no_bi_attention", action="store_true", help="Disable BiAttention layer")
    parser.add_argument("--use_gated_attention", action="store_true", default=True, help="Use Gated Attention layer")
    parser.add_argument("--no_gated_attention", action="store_true", help="Disable Gated Attention layer")
    parser.add_argument("--num_attention_heads", default=4, type=int, help="Number of attention heads in Graph Transformer")

    # Ablation study arguments - Graph architecture
    parser.add_argument("--use_gat", action="store_true", default=True, help="Use GAT for local message passing")
    parser.add_argument("--no_gat", action="store_true", help="Disable GAT")
    parser.add_argument("--use_graph_transformer", action="store_true", default=True, help="Use Graph Transformer for global reasoning")
    parser.add_argument("--no_graph_transformer", action="store_true", help="Disable Graph Transformer")
    parser.add_argument("--no_graph", action="store_true", help="Disable all graph reasoning (use BiLSTM output directly)")
    parser.add_argument("--num_gat_layers", default=1, type=int, help="Number of GAT layers")
    parser.add_argument("--num_transformer_layers", default=1, type=int, help="Number of Graph Transformer layers")

    # Ablation study arguments - Node types
    parser.add_argument("--use_entity_nodes", action="store_true", default=True, help="Use entity nodes in graph")
    parser.add_argument("--no_entity_nodes", action="store_true", help="Disable entity nodes")
    parser.add_argument("--use_sentence_nodes", action="store_true", default=True, help="Use sentence nodes in graph")
    parser.add_argument("--no_sentence_nodes", action="store_true", help="Disable sentence nodes")
    parser.add_argument("--use_node_type_embed", action="store_true", default=True, help="Use node type embeddings in Graph Transformer")
    parser.add_argument("--no_node_type_embed", action="store_true", help="Disable node type embeddings")

    args = parser.parse_args()

    # Handle negation flags - Attention
    if args.no_bi_attention:
        args.use_bi_attention = False
    if args.no_gated_attention:
        args.use_gated_attention = False

    # Handle negation flags - Graph architecture
    if args.no_gat:
        args.use_gat = False
    if args.no_graph_transformer:
        args.use_graph_transformer = False
    if args.no_graph:
        args.use_gat = False
        args.use_graph_transformer = False

    # Handle negation flags - Node types
    if args.no_entity_nodes:
        args.use_entity_nodes = False
    if args.no_sentence_nodes:
        args.use_sentence_nodes = False
    if args.no_node_type_embed:
        args.use_node_type_embed = False

    args.model_name_or_path = MODEL_PATH_MAP[args.model_type]
    main(args)
