import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

from transformers import RobertaPreTrainedModel
import dgl.nn.pytorch as dglnn

from utils import MODEL_CLASSES


class GraphTransformerLayer(nn.Module):
    """
    Graph Transformer Layer for heterogeneous graphs.
    Implements multi-head self-attention over graph structure.
    """
    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super(GraphTransformerLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        # Feed-forward network (smaller, simpler)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values to prevent gradient explosion."""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight, gain=0.1)
            nn.init.zeros_(module.bias)
        for module in self.ffn:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                nn.init.zeros_(module.bias)

    def forward(self, g, h):
        """
        Args:
            g: DGL graph (homogeneous, converted from heterogeneous)
            h: Node features [num_nodes, hidden_size]
        Returns:
            Updated node features [num_nodes, hidden_size]
        """
        # Self-attention with residual
        h_norm = self.norm1(h)

        # Compute Q, K, V
        Q = self.q_proj(h_norm)
        K = self.k_proj(h_norm)
        V = self.v_proj(h_norm)

        # Reshape for multi-head attention
        num_nodes = h.size(0)
        Q = Q.view(num_nodes, self.num_heads, self.head_dim)
        K = K.view(num_nodes, self.num_heads, self.head_dim)
        V = V.view(num_nodes, self.num_heads, self.head_dim)

        # Message passing attention
        with g.local_scope():
            g.ndata['q'] = Q
            g.ndata['k'] = K
            g.ndata['v'] = V

            # Compute attention scores on edges
            g.apply_edges(self._compute_attention_scores)

            # Apply softmax over incoming edges
            edge_weights = dgl.ops.edge_softmax(g, g.edata['attn'])
            g.edata['attn'] = edge_weights

            # Message passing: aggregate values weighted by attention
            g.update_all(
                dgl.function.u_mul_e('v', 'attn', 'm'),
                dgl.function.sum('m', 'h_new')
            )

            h_attn = g.ndata['h_new'].view(num_nodes, self.hidden_size)

        h_attn = self.out_proj(h_attn)
        h = h + self.dropout(h_attn)

        # FFN with residual
        h = h + self.ffn(self.norm2(h))

        return h

    def _compute_attention_scores(self, edges):
        """Compute attention scores for edges."""
        # [num_edges, num_heads, head_dim]
        q = edges.dst['q']
        k = edges.src['k']

        # Scaled dot-product attention
        attn = (q * k).sum(dim=-1) * self.scale  # [num_edges, num_heads]

        return {'attn': attn.unsqueeze(-1)}  # [num_edges, num_heads, 1]


class HeteroGraphTransformer(nn.Module):
    """
    Heterogeneous Graph Transformer that handles different node and edge types.
    """
    def __init__(self, hidden_size, num_heads=4, num_layers=1, dropout=0.1):
        super(HeteroGraphTransformer, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Single Graph Transformer layer (simpler)
        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Node type embeddings to distinguish different node types
        self.node_type_embed = nn.Embedding(4, hidden_size)  # 4 types: question, paragraph, sentence, entity

        # Skip connection weight - heavily favor original features to preserve sample variance
        self.skip_weight = nn.Parameter(torch.tensor(0.05))  # Only 5% transformer, 95% original

    def forward(self, g, node_feats):
        """
        Args:
            g: DGL heterogeneous graph
            node_feats: dict of node features {ntype: tensor}
        Returns:
            dict of updated node features {ntype: tensor}
        """
        # Convert heterogeneous graph to homogeneous for transformer
        node_types = ['question', 'paragraph', 'sentence', 'entity']
        type_to_idx = {t: i for i, t in enumerate(node_types)}

        # Collect all node features and create type indicators
        all_feats = []
        all_type_ids = []
        node_counts = {}

        for ntype in node_types:
            if ntype in node_feats and g.num_nodes(ntype) > 0:
                feat = node_feats[ntype]
                num_nodes = feat.size(0)
                node_counts[ntype] = num_nodes
                all_feats.append(feat)
                all_type_ids.append(torch.full((num_nodes,), type_to_idx[ntype],
                                               dtype=torch.long, device=feat.device))
            else:
                node_counts[ntype] = 0

        if len(all_feats) == 0:
            return node_feats

        # Concatenate all features
        h = torch.cat(all_feats, dim=0)  # [total_nodes, hidden_size]
        h_input = h.clone()  # Save for skip connection
        type_ids = torch.cat(all_type_ids, dim=0)  # [total_nodes]

        # Add node type embeddings (very small scale to not dominate)
        h = h + 0.01 * self.node_type_embed(type_ids)

        # Convert to homogeneous graph
        homo_g = dgl.to_homogeneous(g)

        # Apply transformer layers
        for layer in self.layers:
            h = layer(homo_g, h)

        # Skip connection: combine original and transformed features
        h = self.skip_weight * h + (1 - self.skip_weight) * h_input

        # Split back to heterogeneous format
        result = {}
        offset = 0
        for ntype in node_types:
            count = node_counts[ntype]
            if count > 0:
                result[ntype] = h[offset:offset + count]
                offset += count

        return result


class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        self.g = g
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """ Reinitialize learnable parameters. """
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # edge UDF
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, h):
        z = self.fc(h)
        self.g.ndata['z'] = z
        self.g.apply_edges(self.edge_attntion)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')


class BiAttention(nn.Module):

    """
    bi-attention layer (Seo et al., 2017)

    Placed on top of pre-trained encoder (e.g., RoBERTa, BERT) to fuse information from both the query and the context
    """

    def __init__(self, args, hidden_size, dropout=0.1):
        super(BiAttention, self).__init__()
        self.args = args
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.simW = nn.Linear(self.hidden_size * 3, 1, bias=False)  # nn.Linear(hidden_size * 6, 1, bias=False)

    def forward(self, encoder_out, question_ends):
        query = encoder_out[:, : question_ends + 1, :]
        context = encoder_out[:, question_ends + 1 :, :]
        
        # print("Query hidden : ", query.shape)
        # print("Context hidden : ", context.shape)

        T = context.size(1)
        J = query.size(1)

        sim_shape = (self.args.train_batch_size, T, J, self.hidden_size)  # (self.args.train_batch_size, T, J, 2 * self.hidden_size)
        context_embed = context.unsqueeze(2)  # (N, T, 1, 2d)
        context_embed = context_embed.expand(sim_shape)  # (N, T, J, 2d)
        query_embed = query.unsqueeze(1)  # (N, 1, J, 2d)
        query_embed = query_embed.expand(sim_shape)  # (N, T, J, 2d)
        elemwise_mul = torch.mul(context_embed, query_embed)  # (N, T, J, 2d)
        concat_sim_input = torch.cat((context_embed, query_embed, elemwise_mul), 3)  # (N, T, J, 6d) - [h ; u ; h o u]

        S = self.simW(concat_sim_input).view(self.args.train_batch_size, T, J)  # (N, T, J)
        # print("S : ", S.shape)

        # Context2Query
        c2q = torch.bmm(F.softmax(S, dim=-1), query)  # bmm( (N, T, J), (N, J, 2d) ) = (N, T, 2d)
        # Query2Context
        b = F.softmax(torch.max(S, 2)[0], dim=-1)  # Apply the "maximum function (max_col) across the column"
        q2c = torch.bmm(b.unsqueeze(1), context)
        q2c = q2c.repeat(1, T, 1)  # (N, T, 2d) - tiled `T` times

        # G: query-aware representation of each context word
        G = torch.cat((context, c2q, context.mul(c2q), context.mul(q2c)), -1)  # (N, T, 8d)
        # print("G: ", G.shape)

        return query, G


class ContextEncoder(RobertaPreTrainedModel):
    def __init__(self, args, config):
        super().__init__(config)
        self.args = args
        self.config = config
        _, self.model_class, _ = MODEL_CLASSES[args.model_type]
        self.encoder = self.model_class.from_pretrained(self.args.model_name_or_path, config=self.config)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # RoBERTa doesn't use token_type_ids, only pass if model supports it
        if 'roberta' in self.args.model_type.lower():
            encoded_out = self.encoder(input_ids, attention_mask=attention_mask)
        else:
            encoded_out = self.encoder(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return encoded_out


class GatedAttention(nn.Module):
    def __init__(self, args, config):
        super(GatedAttention, self).__init__()
        self.args = args
        self.config = config
        self.context_mlp = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size * 2, bias=True)
        self.graph_mlp = nn.Linear(self.config.hidden_size, self.config.hidden_size * 2, bias=True)
        self.gate_linear = nn.Linear(self.config.hidden_size * 3, self.config.hidden_size * 4, bias=True)
    
    def forward(self, context_rep, graph_rep):
        """
        context_rep : M (in the HGN paper)
        graph_rep : H' (in the HGN paper)
        context_initial : C (in the HGN paper)

        """
        ctx_dot = F.relu(self.context_mlp(context_rep))
        graph_dot = F.relu(self.graph_mlp(graph_rep))

        # print("ctx_dot (shape): ", ctx_dot.shape)
        # print("graph_dot (shape): ", graph_dot.shape)

        C_merged = torch.bmm(ctx_dot, graph_dot.permute(0, 2, 1).contiguous())
        ctx2nodes = F.softmax(C_merged, dim=-1)  # Context-to-node (attention)
        # print("C_merged: ", C_merged.shape)
        # print("ctx2nodes: ", ctx2nodes.shape)
        # print("graph_rep: ", graph_rep.shape)

        H_attn = torch.bmm(ctx2nodes, graph_rep)  # (B, N, M) x (B, M, H) = (B, N, H)
        ctx_graph = torch.cat((context_rep, H_attn), dim=-1)
        gated1 = F.sigmoid(self.gate_linear(ctx_graph))  # (B, N, 2H) ; (B, N, H) = (B, N, 3H)
        gated2 = torch.tanh(self.gate_linear(ctx_graph))  # (B, N, 2H) ; (B, N, H) = (B, N, 3H)
        G = gated1 * gated2
        # print("gated1: ", gated1.shape)
        # print("gated2: ", gated2.shape)
        # print("G: ", G.shape)
        return G


class NumericHGN(nn.Module):
    def __init__(self, args, config):
        super(NumericHGN, self).__init__()
        self.args = args
        self.config = config
        self.encoder = ContextEncoder(self.args, config)

        # Don't freeze encoder - train end-to-end
        # for param in self.encoder.parameters():
        #     param.requires_grad = False

        self.bi_attn = BiAttention(args, self.config.hidden_size)
        self.bi_attn_linear = nn.Linear(self.config.hidden_size * 4, self.config.hidden_size)
        self.bi_lstm = nn.LSTM(self.config.hidden_size, self.config.hidden_size, bidirectional=True)
        self.para_node_mlp = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size)
        self.sent_node_mlp = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size)
        self.ent_node_mlp = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size)


        # GAT for local message passing
        self.gat = dglnn.HeteroGraphConv({
            'ps': dglnn.GATv2Conv(self.config.hidden_size, self.config.hidden_size, num_heads=1),
            'sp': dglnn.GATv2Conv(self.config.hidden_size, self.config.hidden_size, num_heads=1),
            'se': dglnn.GATv2Conv(self.config.hidden_size, self.config.hidden_size, num_heads=1),
            'es': dglnn.GATv2Conv(self.config.hidden_size, self.config.hidden_size, num_heads=1),
            'pp': dglnn.GATv2Conv(self.config.hidden_size, self.config.hidden_size, num_heads=1),
            'ss': dglnn.GATv2Conv(self.config.hidden_size, self.config.hidden_size, num_heads=1),
            'qp': dglnn.GATv2Conv(self.config.hidden_size, self.config.hidden_size, num_heads=1),
            'pq': dglnn.GATv2Conv(self.config.hidden_size, self.config.hidden_size, num_heads=1),
            'qe': dglnn.GATv2Conv(self.config.hidden_size, self.config.hidden_size, num_heads=1),
            'eq': dglnn.GATv2Conv(self.config.hidden_size, self.config.hidden_size, num_heads=1),
        }, aggregate='sum')

        # Graph Transformer for global reasoning
        self.graph_transformer = HeteroGraphTransformer(
            hidden_size=self.config.hidden_size,
            num_heads=4,
            num_layers=1,
            dropout=0.1
        )

        # Learnable weight for GAT vs Transformer combination (initialized to favor GAT)
        self.gat_weight = nn.Parameter(torch.tensor(0.9))

        self.gated_attn = GatedAttention(self.args, self.config)

        self.para_mlp = nn.Sequential(nn.Linear(self.config.hidden_size, self.config.hidden_size), nn.Linear(self.config.hidden_size, args.num_paragraphs))
        self.sent_mlp = nn.Sequential(nn.Linear(self.config.hidden_size, self.config.hidden_size), nn.Linear(self.config.hidden_size, args.num_sentences))
        self.ent_mlp = nn.Sequential(nn.Linear(self.config.hidden_size, self.config.hidden_size), nn.Linear(self.config.hidden_size, args.num_entities))
        self.span_mlp = nn.Sequential(nn.Linear(self.config.hidden_size * 4, self.config.hidden_size), nn.Linear(self.config.hidden_size, self.config.num_labels))
        self.answer_type_mlp = nn.Sequential(
            nn.Linear(self.config.hidden_size * 4, self.config.hidden_size),
            nn.Tanh(),
            nn.Linear(self.config.hidden_size, 3)
        )

        # Initialize the final classification layer with small weights
        # to prevent initial bias toward any class
        self._init_answer_type_weights()

    def _init_answer_type_weights(self):
        """Initialize answer type MLP weights to prevent bias toward any class."""
        # Get the final linear layer (index 2 after Linear, Tanh)
        final_layer = self.answer_type_mlp[2]
        nn.init.xavier_uniform_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

    def forward(self, input_ids, attention_mask, token_type_ids, labels, graph_out, question_ends):
        """
        Args

        input_ids : [Question ; Context] tokenized by tokenizer (e.g., BertTokenizer)
        attention_mask : attention_mask
        token_type_ids : token_type_ids
        labels : labels in tuple: (para_lbl, sent_lbl, span_idx, answer_type_lbl)
        graph_out = (g, node_idx, span_dict)
        g : dgl.graph - the hierarchical graph neural network
        question_ends : question end index (e.g., first occurrence of [SEP] token in [Q;C] input)

        """
        para_lbl, sent_lbl, answer_type_lbl, span_idx = labels

        encoder_out = self.encoder(input_ids, attention_mask, token_type_ids)
        seq_out = encoder_out[0]
        Q, C = self.bi_attn(seq_out, question_ends)
        C = self.bi_attn_linear(C)
        C = C.permute(1, 0, 2)  # Change dimension from `batch-first` to `sequence-first`
        h0 = torch.randn(2, self.args.train_batch_size, self.config.hidden_size).to(C.device)  # TODO: Think about new ways to initialize self.bi_lstm's h0 and c0
        c0 = torch.randn(2, self.args.train_batch_size, self.config.hidden_size).to(C.device)
        print("Q: ", Q.shape)
        print("C: ", C.shape)
        M, (hn, cn) = self.bi_lstm(C, (h0, c0))
    
        # Extract from `M` with the given spans of paragraphs, sentences and entities
        # (i) The hidden state of the backward LSTM at the start position
        # (ii) And the hidden state of the forward LSTM at the end position
        print("M (shape) : ", M.shape)
        print("h_n (shape) : ", hn.shape)
        print("c_n (shape) : ", cn.shape)

        # TODO: Need to extract paragraph, sentence and entity representations from `M`
        g, node_idx, span_dict = graph_out
        question_idx = span_dict['question']
        para_idx = span_dict['paragraph']
        sent_idx = span_dict['sentence']
        ent_idx = span_dict['entity']

        print("Question_idx: ", question_idx)
        print("Paragraph_idx: ", para_idx)
        print("Sentence_idx: ", sent_idx)
        print("Entity_idx: ", ent_idx)

        print("g : ", g)

        for q_idx in question_idx:
            start_q, _ = q_idx
            # assert end_q == question_ends  # TODO: Rebuild cached_data later on to fix this issue
            end_q = question_ends.item()

        # Extract spans from `M`
        M_temp = M.squeeze(-2)  # TODO: This only works for `batch_size = 1` case
        max_len = M_temp.size(0)

        # Use graph node counts as source of truth to ensure feature/node count match
        num_para_nodes = g.num_nodes('paragraph')
        num_sent_nodes = g.num_nodes('sentence')
        num_ent_nodes = g.num_nodes('entity')

        para_node_input = torch.zeros(num_para_nodes, self.config.hidden_size * 2).to(M.device)
        sent_node_input = torch.zeros(num_sent_nodes, self.config.hidden_size * 2).to(M.device)
        ent_node_input = torch.zeros(num_ent_nodes, self.config.hidden_size * 2).to(M.device)

        valid_para_count = 0
        for i, p_span in enumerate(para_idx[:num_para_nodes]):
            if p_span[0] < max_len and p_span[1] < max_len:
                para_node_input[i] = torch.cat((M_temp[p_span[0]][self.config.hidden_size:], M_temp[p_span[1]][:self.config.hidden_size]))
                valid_para_count += 1
        valid_sent_count = 0
        for i, s_span in enumerate(sent_idx[:num_sent_nodes]):
            if s_span[0] < max_len and s_span[1] < max_len:
                sent_node_input[i] = torch.cat((M_temp[s_span[0]][self.config.hidden_size:], M_temp[s_span[1]][:self.config.hidden_size]))
                valid_sent_count += 1
        valid_ent_count = 0
        for i, e_span in enumerate(ent_idx[:num_ent_nodes]):
            if e_span[0] < max_len and e_span[1] < max_len:
                ent_node_input[i] = torch.cat((M_temp[e_span[0]][self.config.hidden_size:], M_temp[e_span[1]][:self.config.hidden_size]))
                valid_ent_count += 1

        print(f"Valid para spans: {valid_para_count}/{num_para_nodes} (max_len={max_len})")
        print(f"Valid sent spans: {valid_sent_count}/{num_sent_nodes}")
        print(f"Valid ent spans: {valid_ent_count}/{num_ent_nodes}")
        
        # Max-pooling `Q` for question representation
        Q_temp = Q.squeeze(0)
        q, _ = torch.max(Q_temp, dim=0)
        print("q (shape): ", q.shape)

        # Construct node representations
        para_rep = self.para_node_mlp(para_node_input)
        sent_rep = self.sent_node_mlp(sent_node_input)
        ent_rep = self.ent_node_mlp(ent_node_input)

        print("para_initial_embed: ", para_rep.shape)
        print("sent_initial_embed: ", sent_rep.shape)
        print("ent_initial_embed: ", ent_rep.shape)

        # Initialize the paragraph, sentence and entity nodes with the node representations
        # Ensure q has shape [1, hidden_size] for single question node
        q = q.unsqueeze(0)

        print("Graph node counts:", {ntype: g.num_nodes(ntype) for ntype in g.ntypes})
        print("Feature shapes - q:", q.shape, "para:", para_rep.shape, "sent:", sent_rep.shape, "ent:", ent_rep.shape)

        in_feats = {"question": q, "paragraph": para_rep, "sentence": sent_rep, "entity": ent_rep}

        # GAT for local message passing
        gat_out = self.gat(g, (in_feats, in_feats))
        gat_rep_list = []
        for ntype in ['question', 'paragraph', 'sentence', 'entity']:
            if ntype in gat_out:
                gat_rep_list.append(gat_out[ntype].squeeze(-2))  # Remove head dimension
        gat_rep = torch.cat(gat_rep_list, dim=0)

        # Graph Transformer for global reasoning
        transformer_out = self.graph_transformer(g, in_feats)
        transformer_rep_list = []
        for ntype in ['question', 'paragraph', 'sentence', 'entity']:
            if ntype in transformer_out:
                transformer_rep_list.append(transformer_out[ntype])
        transformer_rep = torch.cat(transformer_rep_list, dim=0)

        # Debug: check GAT vs Transformer variance
        print("gat_rep MEAN/STD: ", gat_rep.mean().item(), gat_rep.std().item())
        print("transformer_rep MEAN/STD: ", transformer_rep.mean().item(), transformer_rep.std().item())

        # Learnable weighted combination: GAT preserves variance better
        # Use sigmoid to keep weight between 0 and 1
        gat_w = torch.sigmoid(self.gat_weight)
        graph_rep = gat_w * gat_rep + (1 - gat_w) * transformer_rep  # [num_nodes, hidden]
        print(f"GAT weight: {gat_w.item():.4f}")

        # Add batch dimension: [num_nodes, hidden] -> [num_nodes, 1, hidden]
        graph_rep = graph_rep.unsqueeze(1)

        print("graph_rep (shape): ", graph_rep.shape)
        print("g (ntypes): ", g.ntypes)
        print("g (etypes): ", g.etypes)

        M_perm = M.permute(1, 0, 2)
        graph_rep_perm = graph_rep.permute(1, 0, 2)
        gated_rep = self.gated_attn(M_perm, graph_rep_perm)

        print("G (shape): ", gated_rep.shape)
        start_end_logits = self.span_mlp(gated_rep)
        print("start_end (shape): ", start_end_logits.shape)
        print("gated_rep[0] (CLS rep): ", gated_rep.squeeze(0)[:1].shape)

        start_logits, end_logits = start_end_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        print("start_logits: ", start_logits.shape)
        print("end_logits: ", end_logits.shape)
        # Use combination of CLS representation and mean pooling for answer type classification
        # This gives better signal for distinguishing answer types
        cls_rep = gated_rep[:, 0, :]  # CLS token representation [batch, hidden*4]
        mean_rep = gated_rep.mean(dim=1)  # Mean pooling [batch, hidden*4]
        # Use CLS representation primarily (it captures global semantics better)
        pooled_rep = cls_rep + 0.5 * mean_rep  # Weighted combination
        answer_type_logits = self.answer_type_mlp(pooled_rep)
        print("answer_type_logit (shape): ", answer_type_logits.shape)
        print("answer_type_logits VALUES: ", answer_type_logits.detach().cpu().numpy())
        print("pooled_rep MEAN/STD: ", pooled_rep.mean().item(), pooled_rep.std().item())
        print("gated_rep MEAN/STD: ", gated_rep.mean().item(), gated_rep.std().item())
        print("graph_rep MEAN/STD: ", graph_rep.mean().item(), graph_rep.std().item())
        print("answer_type_lbl: ", answer_type_lbl)

        ignored_index = start_logits.size(-1)
        print("span_idx: ", span_idx)
        start_pos, end_pos = span_idx[0].unsqueeze(-1)
        # TODO: In https://huggingface.co/transformers/v2.10.0/_modules/transformers/modeling_bert.html#BertForQuestionAnswering
        # TODO: Is this necessary?
        # start_pos.clamp_(0, ignored_index)
        # end_pos.clamp_(0, ignored_index)

        losses = {}

        # sometimes the start/end positions are outside our model inputs, we ignore these terms
        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

        # Simple cross-entropy without fancy techniques
        # Class distribution: {0: 18, 1: 8, 2: 6} out of 32
        # Inverse frequency weights: 32/18=1.78, 32/8=4.0, 32/6=5.33
        type_weights = torch.tensor([1.78, 4.0, 5.33], device=answer_type_logits.device)
        loss_fct_type = nn.CrossEntropyLoss(weight=type_weights)
        loss_type = loss_fct_type(answer_type_logits, answer_type_lbl)

        # Clamp positions to valid range or set to -1 (ignored)
        num_classes = start_logits.size(-1)
        valid_start = start_pos.item() >= 0 and start_pos.item() < num_classes
        valid_end = end_pos.item() >= 0 and end_pos.item() < num_classes

        # Only compute span losses if positions are valid, otherwise set to 0
        if valid_start:
            loss_start = loss_fct(start_logits, start_pos)
        else:
            loss_start = torch.tensor(0.0, device=start_logits.device)

        if valid_end:
            loss_end = loss_fct(end_logits, end_pos)
        else:
            loss_end = torch.tensor(0.0, device=end_logits.device)

        losses["start"] = loss_start
        losses["end"] = loss_end
        losses["type"] = loss_type

        return loss_start, loss_end, loss_type, start_logits, end_logits, answer_type_logits
