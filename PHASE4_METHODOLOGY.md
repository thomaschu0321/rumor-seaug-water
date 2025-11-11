# Phase 4: GNN Classification with Global Pooling

In the fourth and final phase, we perform graph-level classification by integrating the baseline BERT features and **LLM-enhanced embeddings** through a Graph Neural Network (GNN) architecture. This phase consists of four key steps: (1) feature fusion, (2) GNN processing, (3) graph-level pooling, and (4) classification.

---

## 4.1 Feature Matrix Construction

Given a graph G = (V, E) with nodes V and edges E, let X_initial ∈ R^(|V| × 768) be the initial BERT feature matrix from Phase 1, and let S ⊂ V be the set of nodes selected by DBSCAN in Phase 2. For each selected node v_i ∈ S, we obtain an augmented embedding through **a two-stage LLM-LM pipeline** in Phase 3, which is a core component of our data mining with LLM framework:

### Stage 1: LLM-based Text Augmentation

We leverage Large Language Models (LLMs) to generate semantically enriched text variations for selected nodes:

```
text_aug(v_i) = LLM(text_original(v_i); P)
```

where P represents carefully designed prompts that instruct the LLM (e.g., GPT-4, Claude) to perform semantic expansion, paraphrasing, and contextual rewriting. This step is **critical** for capturing nuanced semantic variations that may indicate misinformation, sarcasm, or misleading content—precisely the types of anomalous nodes identified by DBSCAN in Phase 2.

### Stage 2: Language Model Encoding

The LLM-augmented text is then encoded into dense vector representations using Sentence-BERT:

```
f_LLM→LM(v_i) = SentenceBERT(text_aug(v_i)) ∈ R^384
```

where we employ the `all-MiniLM-L6-v2` model to generate 384-dimensional embeddings that capture the enriched semantic information from the LLM augmentation.

The complete augmentation pipeline can be formulated as:

```
f_LLM→LM(v_i) = SentenceBERT ∘ LLM(text(v_i)) ∈ R^384
```

where ∘ denotes function composition, emphasizing that LLM augmentation is applied **before** the final encoding step.

### Feature Matrix Construction

Rather than simply replacing features, we adopt a **feature fusion** strategy that combines both baseline BERT representations and LLM-enhanced embeddings. For each node v_i in the graph, we construct the input feature vector x_i' as:

```
x_i' = { Fusion(X_initial[i], f_LLM→LM(v_i))    if v_i ∈ S
       { X_initial[i]                            if v_i ∉ S
```

This approach preserves the semantic information from BERT while incorporating **LLM-enhanced representations** for critical nodes identified by our adaptive selection mechanism. The integration of LLMs addresses a key limitation of traditional feature extraction methods: **pre-trained encoders like BERT may fail to capture subtle semantic anomalies** (e.g., sarcasm, implicit misinformation, nuanced linguistic patterns) that are crucial for rumor detection. By selectively applying LLM augmentation to semantically anomalous nodes, we leverage the deep contextual understanding and generation capabilities of large language models while maintaining computational efficiency through selective augmentation (only ~30% of nodes).

---

## 4.2 Feature Fusion Strategies

We explore four fusion strategies to combine baseline and augmented features, each with different representational capacity:

### (a) Concatenation Fusion

The simplest approach directly concatenates the two feature vectors:

```
x_fused[i] = [X_base[i]; X_aug[i]] ∈ R^1152
```

where [·; ·] denotes concatenation. This results in a 768 + 384 = 1152 dimensional feature vector.

### (b) Weighted Fusion

A learnable weighted combination projects both features to a common hidden dimension:

```
x_fused[i] = α · W_base · X_base[i] + (1-α) · W_aug · X_aug[i]
```

where:
- W_base ∈ R^(d × 768) and W_aug ∈ R^(d × 384) are learnable projection matrices
- α ∈ (0,1) is a learnable weight parameter constrained by a sigmoid function
- d is the hidden dimension (default: 64)

### (c) Gated Fusion

A gating mechanism dynamically controls the information flow from each feature source:

```
g_i = σ(W_g · [X_base[i]; X_aug[i]] + b_g)
x_fused[i] = g_i ⊙ (W_aug · X_aug[i]) + (1-g_i) ⊙ (W_base · X_base[i])
```

where:
- σ is the sigmoid activation function
- ⊙ denotes element-wise multiplication
- g_i ∈ R^d represents the gate values

### (d) Attention-based Fusion

Attention weights adaptively determine the contribution of each feature:

```
α_base, α_aug = softmax(W_a · [W_base·X_base[i]; W_aug·X_aug[i]])
x_fused[i] = α_base · W_base·X_base[i] + α_aug · W_aug·X_aug[i]
```

where W_a is a learnable attention parameter matrix that outputs normalized attention scores.

**Note:** For nodes v_i ∉ S (not selected for augmentation), we apply only the baseline projection: x_fused[i] = W_base·X_base[i].

---

## 4.3 Graph Neural Network Processing

After feature fusion, we apply a multi-layer GNN to propagate and aggregate information across the graph structure. We support two GNN backbones to demonstrate the generalizability of our framework:

### (a) Graph Convolutional Network (GCN)

Following Kipf & Welling (2017), the layer-wise propagation rule for GCN is:

```
H^(l+1) = σ(D̃^(-1/2) · Ã · D̃^(-1/2) · H^(l) · W^(l))
```

where:
- H^(l) ∈ R^(|V| × d^(l)) is the matrix of node features at layer l
- W^(l) ∈ R^(d^(l) × d^(l+1)) is the learnable weight matrix for layer l
- Ã = A + I_N is the adjacency matrix A with added self-loops (I_N)
- D̃ is the diagonal degree matrix of Ã
- σ is an activation function (ReLU in our implementation)

### (b) Graph Attention Network (GAT)

For GAT, we employ multi-head attention mechanisms to learn adaptive neighbor importance:

```
h_i^(l+1) = σ( Σ_{j∈N(i)} α_ij^(l) · W^(l) · h_j^(l) )
```

where the attention coefficients α_ij^(l) are computed as:

```
α_ij^(l) = exp(LeakyReLU(a^T · [W^(l)·h_i^(l) || W^(l)·h_j^(l)]))
           ────────────────────────────────────────────────────────
           Σ_{k∈N(i)} exp(LeakyReLU(a^T · [W^(l)·h_i^(l) || W^(l)·h_k^(l)]))
```

where:
- a is a learnable attention vector
- || denotes concatenation
- N(i) represents the neighborhood of node i

For multi-head attention with K heads, we concatenate the outputs:

```
h_i^(l+1) = ||_{k=1}^K σ( Σ_{j∈N(i)} α_ij^k · W^k · h_j^(l) )
```

### Network Architecture

We employ a 2-layer GNN architecture with batch normalization and dropout for regularization:

```
H^(1) = GNN(X_fused, Ã) ∈ R^(|V| × d)
H^(1) ← BatchNorm(ReLU(Dropout(H^(1), p=0.3)))

H^(2) = GNN(H^(1), Ã) ∈ R^(|V| × d)
H^(2) ← BatchNorm(ReLU(Dropout(H^(2), p=0.3)))
```

where:
- d = 64 is the hidden dimension
- GNN denotes either GCN or GAT layers
- p = 0.3 is the dropout probability

---

## 4.4 Global Graph Pooling

Since rumor detection requires graph-level classification (predicting the label of the entire propagation cascade), we aggregate node-level representations into a single graph-level representation. We employ **global mean pooling** to obtain the graph embedding:

```
h_G = (1/|V|) · Σ_{i=1}^{|V|} h_i^(L) ∈ R^d
```

where:
- h_i^(L) is the final node embedding after L=2 GNN layers
- h_G is the graph-level representation
- |V| is the number of nodes in the graph

**Rationale:** Mean pooling is chosen over max or sum pooling because:
1. It is **invariant to graph size**, allowing fair comparison across propagation cascades of different scales
2. It captures the **overall propagation pattern** without being dominated by individual outlier nodes
3. It provides a **balanced representation** of the entire information diffusion structure

---

## 4.5 Classification Layer

The graph embedding h_G is passed through a fully connected layer with softmax activation to produce the final prediction:

```
ŷ = softmax(W_cls · h_G + b_cls) ∈ R^C
```

where:
- W_cls ∈ R^(C × d) and b_cls ∈ R^C are learnable parameters
- C is the number of classes:
  - C = 2 for binary classification (Rumor/Non-rumor) on Weibo dataset
  - C = 4 for multi-class classification (True/False/Unverified/Non-rumor) on Twitter15/16 datasets
- ŷ represents the predicted probability distribution over classes

---

## 4.6 Training Objective

We optimize the model using cross-entropy loss with L2 regularization:

```
L = -(1/N) · Σ_{i=1}^N Σ_{c=1}^C y_i^c · log(ŷ_i^c) + λ · ||Θ||_2^2
```

where:
- N is the number of graphs in the training set
- y_i^c is the ground truth label (one-hot encoded)
- ŷ_i^c is the predicted probability for class c
- λ is the L2 regularization coefficient (default: 5e-4)
- Θ represents all learnable parameters in the model

### Optimization

We employ the **Adam optimizer** (Kingma & Ba, 2015) with:
- Initial learning rate: 0.001
- Weight decay: 5e-4
- Early stopping: based on validation accuracy with patience of 10 epochs

---

## 4.7 Dimension Flow Summary

The complete dimension transformation through Phase 4 is:

```
Input:
  X_base ∈ R^(|V| × 768)              [BERT features for all nodes]
  X_aug ∈ R^(|S| × 384)               [LLM-LM features for selected nodes]

Feature Fusion:
  X_fused ∈ R^(|V| × 1152)            [Concatenation strategy]
  or
  X_fused ∈ R^(|V| × 64)              [Weighted/Gated/Attention strategies]

GNN Processing:
  H^(1) ∈ R^(|V| × 64)                [After GNN Layer 1]
  H^(2) ∈ R^(|V| × 64)                [After GNN Layer 2]

Global Pooling:
  h_G ∈ R^(1 × 64)                    [Graph-level representation]

Classification:
  ŷ ∈ R^(1 × C)                       [Final prediction probabilities]
```

where |V| is the number of nodes, |S| is the number of selected nodes (|S| ≈ 0.3|V|), and C is the number of classes.

---

## 4.8 Computational Complexity

The computational complexity of Phase 4 is:

### Feature Fusion
- Concatenation: O(|V|)
- Weighted/Gated/Attention: O(|V| · d²)

### GNN Layers
- GCN: O(L · |E| · d²) where L=2 is the number of layers
- GAT: O(L · |E| · d² + L · |V| · K · d²) where K=4 is the number of attention heads

### Global Pooling
- O(|V| · d)

### Classification
- O(d · C)

**Overall complexity**: O(|E| · d²) dominated by GNN layers, which is efficient for sparse social network graphs where |E| ≪ |V|².

---

## 4.9 Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `hidden_dim` | 64 | Hidden dimension for GNN layers |
| `num_gnn_layers` | 2 | Number of GNN layers |
| `dropout` | 0.3 | Dropout probability |
| `gat_heads` | 4 | Number of attention heads (GAT only) |
| `learning_rate` | 0.001 | Initial learning rate for Adam |
| `weight_decay` | 5e-4 | L2 regularization coefficient |
| `batch_size` | 32 | Training batch size |
| `num_epochs` | 50 | Maximum training epochs |
| `patience` | 10 | Early stopping patience |

These hyperparameters were selected through preliminary experiments to balance model expressiveness, generalization, and computational efficiency.

---

## 4.10 Design Rationale

### Why 2-layer GNN?
- **1 layer**: Insufficient to capture multi-hop propagation patterns
- **2 layers**: Captures 2-hop neighborhood information, sufficient for most rumor propagation trees
- **3+ layers**: Risk of over-smoothing and over-fitting on our dataset sizes

### Why Multiple Fusion Strategies?
We implement four fusion strategies to demonstrate:
1. **Concatenation**: Simple baseline, preserves all information
2. **Weighted**: Learnable balance between baseline and augmented features
3. **Gated**: Dynamic, node-specific fusion decisions
4. **Attention**: Most flexible, learns optimal combination per node

This variety validates that performance gains come from **selective LLM augmentation**, not from a specific fusion choice.

### Why Both GCN and GAT?
Supporting multiple GNN backbones demonstrates the **generalizability** of the TAPE framework:
- GCN: Uniform neighbor aggregation (baseline)
- GAT: Attention-weighted aggregation (adaptive)

Consistent improvements across both architectures validate that gains stem from our **LLM-enhanced selective augmentation strategy** rather than architectural choices.

---

## References

1. Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. ICLR.
2. Veličković, P., et al. (2018). Graph attention networks. ICLR.
3. Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. ICLR.
4. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. EMNLP.

---

**Last Updated**: 2025-11-11  
**Version**: 1.0  
**Author**: TAPE Framework Team

