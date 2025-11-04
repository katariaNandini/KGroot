# KGroot Implementation - Complete Explanation

## Overview

KGroot is a graph-based machine learning system for **fault detection and root cause analysis** in microservices architectures. It learns to identify when system states are similar to known fault scenarios by comparing graph representations of system telemetry data.

## 🎯 What KGroot Does

**Problem**: In complex microservices systems, when a fault occurs, it's difficult to:
1. Quickly identify that a fault has occurred
2. Find the root cause of the fault
3. Determine if current issues match previously seen problems

**Solution**: KGroot uses Graph Neural Networks (GNNs) to:
1. Convert system telemetry data into graphs
2. Learn similarity between different system states
3. Identify when current state matches known fault patterns
4. Provide insights for root cause analysis

## 📊 Data Flow - Step by Step

### Step 1: Raw Data Collection
```
Bank/telemetry/2021_03_04/
├── log/log_service.csv          # Application logs with error messages
├── metric/metric_app.csv        # Performance metrics (response time, success rate)
├── metric/metric_container.csv  # System metrics (CPU, memory usage)
└── trace/trace_span.csv         # Distributed tracing data (service calls)
```

### Step 2: Component Extraction
**File**: `bank_data_processor.py` → `_load_telemetry_data()`

The system extracts unique components (services, containers, applications) from:
- **Logs**: `cmdb_id` column identifies services
- **Metrics**: `tc` (service name) and `cmdb_id` columns
- **Traces**: `cmdb_id` identifies services in call chains

**Example Components Found**:
```
- payment-service-001
- user-service-002  
- database-container-003
- api-gateway-004
```

### Step 3: Feature Engineering
**File**: `bank_data_processor.py` → `_create_component_graph()`

Each component gets a **100-dimensional feature vector**:

| Feature Index | Description | Calculation |
|---------------|-------------|-------------|
| 0 | Component Type | Hash-based encoding of service name |
| 1 | Activity Level | Log count normalized to [0,1] |
| 2 | Error Rate | Error logs / Total logs |
| 3 | Response Rate | Average response rate from metrics |
| 4 | Success Rate | Average success rate from metrics |
| 5 | Response Time | Mean response time normalized |
| 6 | Request Count | Total requests normalized |
| 7-9 | Container Metrics | CPU, memory usage statistics |
| 10-99 | Random Features | Filled with random values for now |

### Step 4: Graph Construction
**File**: `bank_data_processor.py` → `_create_component_graph()`

Creates **3 different adjacency matrices** representing different relationship types:

#### Adjacency Matrix 1: Service Call Relationships
- **Source**: Distributed traces (`trace_span.csv`)
- **Logic**: If service A calls service B, create edge A→B
- **Purpose**: Captures the actual service dependency graph

#### Adjacency Matrix 2: Temporal Correlations  
- **Source**: Log data
- **Logic**: Services active around the same time are correlated
- **Purpose**: Captures services that tend to have issues together

#### Adjacency Matrix 3: Metric Correlations
- **Source**: Performance metrics
- **Logic**: Services with similar performance patterns are related
- **Purpose**: Captures services with similar behavior patterns

### Step 5: Graph Data Structure
**File**: `bank_data_processor.py` → `process_all_dates()`

Each graph is saved as a pickle file with:
```python
{
    'fetures': np.array(shape=(n_components, 100)),  # Node features
    'adj': [adj1, adj2, adj3],                      # 3 adjacency matrices
    'node_index_value': [0, 1, 2, ...]              # Component indices
}
```

### Step 6: Training Data Creation
**File**: `bank_data_processor.py` → `create_labeled_data()`

Creates positive and negative pairs for training:
- **Positive pairs**: Same date compared to itself (label=1)
- **Negative pairs**: Different dates compared to each other (label=0)

**Example**:
```json
[
    ["2021_03_04.pkl", "2021_03_04.pkl", 1],  // Positive: same state
    ["2021_03_04.pkl", "2021_03_05.pkl", 0],  // Negative: different states
    ["2021_03_05.pkl", "2021_03_05.pkl", 1],  // Positive: same state
    ["2021_03_05.pkl", "2021_03_04.pkl", 0]   // Negative: different states
]
```

## 🧠 Neural Network Architecture

### Graph Convolutional Network (GCN)
**File**: `model_batch.py` → `GraphConvolution`

**Key Innovation**: Multi-relational GCN with basis decomposition

#### Traditional GCN:
```
New_node_features = Activation(Adjacency × Node_features × Weight_matrix)
```

#### KGroot's Multi-relational GCN:
```
For each relationship type r:
    Support_r = Adjacency_r × Node_features
    
Combined_support = Concat([Support_1, Support_2, Support_3])

# Basis decomposition for efficiency
Weight_basis = Σ(basis_i × coefficient_i)
New_features = Combined_support × Weight_basis
```

**Benefits**:
- Handles multiple relationship types
- Reduces parameters through basis sharing
- More expressive than single-relation GCNs

### Graph Similarity Model
**File**: `model_batch.py` → `GraphSimilarity`

#### Architecture:
```
Input: Two graphs (G1, G2)
├── GCN_online(G1) → Node embeddings for graph 1
├── GCN_kb(G2) → Node embeddings for graph 2
├── MaxPooling → Graph-level embeddings
├── Attention mechanism → Weighted combination
├── MLP → Final similarity score
└── Output: [similarity_prob, dissimilarity_prob]
```

#### Key Components:

1. **Separate GCNs**: Different networks for online vs knowledge base graphs
2. **Max Pooling**: Aggregates node features to graph-level representation
3. **Attention Mechanism**: Learns which components are most important
4. **Gate Control**: Combines information from both graphs intelligently

## 🚀 Training Process

### Configuration
**File**: `config_graph_sim.ini`
```ini
[model]
input_dim = 100           # Feature vector size
gcn_hidden_dim = 64       # GCN hidden layer size  
support = 3              # Number of adjacency matrices
max_node_num = 128       # Maximum nodes per graph
num_bases = 30           # Basis decomposition size

[train]
LR = 0.001               # Learning rate
NB_EPOCH = 10           # Training epochs
batch_size = 4          # Batch size
```

### Training Loop
**File**: `graph_sim_dej_X.py` → `train_model()`

1. **Data Loading**: Load graph pairs with labels
2. **Forward Pass**: 
   - Process both graphs through GCNs
   - Compute similarity score
3. **Loss Calculation**: Cross-entropy loss with class weighting
4. **Backpropagation**: Update model parameters
5. **Evaluation**: Test on validation and test sets

### Loss Function
```python
# Weighted cross-entropy to handle class imbalance
loss = CrossEntropyLoss(
    predictions, 
    labels, 
    weights=[neg_count, pos_count]  # More negative than positive pairs
)
```

## 📈 Evaluation Metrics

The system evaluates performance using:

1. **Accuracy**: Overall correct predictions
2. **Precision**: True positives / (True positives + False positives)
3. **Recall**: True positives / (True positives + False negatives)  
4. **F1-Score**: Harmonic mean of precision and recall
5. **MAR (Mean Average Rank)**: Ranking quality for retrieval tasks

## 🔧 Key Files Explained

### Core Processing Files:

1. **`bank_data_processor.py`**
   - Converts raw telemetry to graph format
   - Extracts components and creates features
   - Builds adjacency matrices
   - Creates training data

2. **`model_batch.py`**
   - Neural network architectures
   - Graph convolution implementation
   - Multi-relational graph handling

3. **`graph_sim_dej_X.py`**
   - Main training and evaluation script
   - Model training loop
   - Performance evaluation

4. **`DataSetGraphSimGenerator.py`**
   - Dataset loading and batching
   - Handles graph padding and preprocessing

### Configuration Files:

1. **`config_graph_sim.ini`**
   - Model hyperparameters
   - Training settings
   - Data paths

2. **`run_bank_kgroot.py`**
   - Main execution script
   - Orchestrates data processing and training

## 🎯 How to Use This System

### Step 1: Prepare Data
```bash
# Ensure Bank telemetry data is in Bank/ directory
Bank/
├── telemetry/2021_03_04/...
├── telemetry/2021_03_05/...
├── query.csv
└── record.csv
```

### Step 2: Process Data
```bash
python bank_data_processor.py
# Creates bank_from_bank/ with pickle files and labeled_data.json
```

### Step 3: Train Model
```bash
python run_bank_kgroot.py
# Processes data and trains KGroot model
```

### Step 4: Evaluate Results
Check the `runs/` directory for TensorBoard logs and model performance.

## 🔍 Understanding the Output

### Graph Representation
Each processed graph represents a system state at a specific time:
- **Nodes**: Services, containers, applications
- **Node Features**: 100-dimensional vectors with telemetry statistics
- **Edges**: 3 types of relationships between components

### Similarity Learning
The model learns to answer: "Is the current system state similar to this known fault scenario?"

### Fault Detection
When a new system state is processed:
1. Convert to graph format
2. Compare with all known fault patterns
3. High similarity → Potential fault detected
4. Low similarity → System appears normal

## 🚀 Advanced Features

### Multi-relational Graphs
Unlike traditional graphs with single edge types, KGroot uses multiple adjacency matrices to capture:
- Service dependencies (traces)
- Temporal correlations (logs)  
- Performance correlations (metrics)

### Basis Decomposition
Efficiently handles multiple relationship types by sharing parameters across relation types, reducing model complexity.

### Attention Mechanisms
Learns which components are most important for similarity decisions, providing interpretability.

## 📚 Key Concepts Summary

1. **Graph Neural Networks**: Learn from graph-structured data
2. **Multi-relational Graphs**: Multiple edge types in same graph
3. **Graph Similarity**: Comparing two graphs to determine similarity
4. **Feature Engineering**: Converting raw telemetry to meaningful features
5. **Fault Detection**: Identifying when system states match known problems
6. **Root Cause Analysis**: Understanding which components are involved in faults

This system represents a sophisticated approach to automated fault detection in complex distributed systems using modern graph machine learning techniques.
