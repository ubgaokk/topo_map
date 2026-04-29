"""
Default configuration for TopoMap models.
"""

class Config:
    # === Model Architecture ===
    dim = 256                    # Feature dimension
    num_lane_queries = 200       # Number of lane detection queries
    num_te_queries = 100         # Number of traffic element queries
    num_points = 32              # Number of points sampled per lane
    bev_h = 100                  # BEV feature map height
    bev_w = 200                  # BEV feature map width
    bev_z = 4                    # Number of height planes
    num_decoder_layers = 6       # Number of transformer decoder layers
    num_heads = 8                # Number of attention heads
    
    # === BEV Encoder ===
    backbone = 'resnet50'        # Backbone network
    pretrained = True            # Use pretrained weights
    
    # === Endpoint Detection ===
    endpoint_hidden_dim = 128    # Hidden dimension for endpoint MLP
    endpoint_num_classes = 2     # x, y offset + confidence
    
    # === Graph Neural Network ===
    gnn_layers = 3               # Number of GNN layers
    edge_types = 3               # successor, predecessor, self-loop
    adjacency_threshold = 5.0    # Distance threshold for lane adjacency (meters)
    dropout = 0.1                # Dropout rate
    
    # === Training ===
    batch_size = 4
    num_workers = 4
    lr = 2.0e-4                  # Learning rate
    weight_decay = 0.01
    epochs = 30
    warmup_epochs = 2
    clip_max_norm = 1.0          # Gradient clipping
    
    # === Loss Weights ===
    lambda_detection = 1.0       # Detection loss weight
    lambda_topology = 1.0        # Topology loss weight
    lambda_endpoint = 0.5        # Endpoint loss weight
    
    # === Data ===
    data_root = None             # Path to OpenLane-V2 dataset
    annotation_range_longitudinal = 50.0   # meters
    annotation_range_lateral = 25.0        # meters
    
    # === Evaluation ===
    eval_interval = 5            # Evaluate every N epochs
    save_interval = 5            # Save checkpoint every N epochs
    
    # === Logging ===
    log_interval = 50            # Print logs every N batches
    project_name = 'topo_map'
    experiment_name = 'default'


def get_config():
    return Config()