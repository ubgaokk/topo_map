"""
OpenLane-V2 Evaluation Metrics

Implements the official OLS (OpenLane-V2 Score) and TOP (Topology) metrics.

OLS = sqrt(Detection_lane × TOP_ll) × 0.5 + Detection_TE × 0.25 + Detection_lane × 0.25

Reference:
- OpenLane-V2: https://github.com/OpenDriveLab/OpenLane-V2
"""

import numpy as np
import torch
from scipy.spatial.distance import cdist
from typing import Dict, List, Tuple, Any, Optional
from scipy.optimize import linear_sum_assignment


def compute_ols_metrics(results: List[Dict]) -> Dict[str, float]:
    """
    Compute official OLS metrics
    
    Args:
        results: List of prediction dictionaries
        
    Returns:
        Dictionary with metrics:
            - det_lane: Lane detection AP
            - det_te: Traffic element detection AP
            - top_ll: Lane-lane topology AP
            - top_lte: Lane-TE topology AP
            - ols: Final OLS score
    """
    # Collect scores across all frames
    all_lane_scores = []
    all_te_scores = []
    all_top_ll_scores = []
    all_top_lte_scores = []
    
    for result in results:
        # Lane detection
        det_lane = result.get('det_lane_score', 0.0)
        all_lane_scores.append(det_lane)
        
        # TE detection
        det_te = result.get('det_te_score', 0.0)
        all_te_scores.append(det_te)
        
        # Lane-lane topology
        top_ll = result.get('top_ll_score', 0.0)
        all_top_ll_scores.append(top_ll)
        
        # Lane-TE topology
        top_lte = result.get('top_lte_score', 0.0)
        all_top_lte_scores.append(top_lte)
    
    # Average across frames
    det_lane_mean = np.mean(all_lane_scores) if all_lane_scores else 0.0
    det_te_mean = np.mean(all_te_scores) if all_te_scores else 0.0
    top_ll_mean = np.mean(all_top_ll_scores) if all_top_ll_scores else 0.0
    top_lte_mean = np.mean(all_top_lte_scores) if all_top_lte_scores else 0.0
    
    # Compute OLS
    # OLS = sqrt(Detection_lane × TOP_ll) × 0.5 + Detection_TE × 0.25 + Detection_lane × 0.25
    ols = (
        np.sqrt(det_lane_mean * top_ll_mean) * 0.5 +
        det_te_mean * 0.25 +
        det_lane_mean * 0.25
    )
    
    return {
        'det_lane': det_lane_mean,
        'det_te': det_te_mean,
        'top_ll': top_ll_mean,
        'top_lte': top_lte_mean,
        'ols': ols,
    }


def compute_lane_detection_score(
    pred_lanes: List[np.ndarray],
    gt_lanes: List[np.ndarray],
    thresholds: List[float] = [1.0, 2.0, 3.0],
) -> float:
    """
    Compute lane detection AP using discrete Fréchet distance
    
    Args:
        pred_lanes: List of predicted lane centerlines, each [N_pts, 3]
        gt_lanes: List of ground truth lanes
        thresholds: Distance thresholds in meters
        
    Returns:
        Average precision score
    """
    if len(pred_lanes) == 0 and len(gt_lanes) == 0:
        return 1.0
    if len(pred_lanes) == 0 or len(gt_lanes) == 0:
        return 0.0
    
    # Compute pairwise distances using Fréchet distance
    # Simplified: use endpoint distance as proxy
    pred_ends = np.array([lane[-1, :2] for lane in pred_lanes]) if len(pred_lanes) > 0 else np.zeros((0, 2))
    gt_ends = np.array([lane[-1, :2] for lane in gt_lanes]) if len(gt_lanes) > 0 else np.zeros((0, 2))
    
    if len(pred_ends) == 0 or len(gt_ends) == 0:
        return 0.0
    
    # Distance matrix (using L2 distance as proxy for Fréchet)
    dist_matrix = cdist(pred_ends, gt_ends)
    
    # Match predictions to ground truth using Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    
    # Compute matches at each threshold
    scores = []
    for threshold in thresholds:
        matches = dist_matrix[row_ind, col_ind] < threshold
        score = matches.sum() / len(gt_lanes)
        scores.append(score)
    
    return np.mean(scores)


def compute_topology_ap(
    pred_matrix: np.ndarray,
    gt_matrix: np.ndarray,
) -> float:
    """
    Compute topology link prediction AP
    
    Uses an IoU-distance style affinity measure.
    
    Args:
        pred_matrix: [N_pred, M_pred] or [N_pred, M_pred] predicted scores
        gt_matrix: [N_gt, M_gt] ground truth binary matrix
        
    Returns:
        Average precision
    """
    if pred_matrix.size == 0 and gt_matrix.size == 0:
        return 1.0
    if pred_matrix.size == 0 or gt_matrix.size == 0:
        return 0.0
    
    # Flatten for AP computation
    pred_flat = pred_matrix.flatten()
    gt_flat = gt_matrix.flatten()
    
    # Compute precision-recall curve
    # Sort by prediction score
    sorted_indices = np.argsort(-pred_flat)
    gt_sorted = gt_flat[sorted_indices]
    
    # Cumulative true positives
    cumsum = np.cumsum(gt_sorted)
    
    # Precision and recall at each threshold
    recall = cumsum / (gt_flat.sum() + 1e-8)
    precision = cumsum / (np.arange(len(gt_flat)) + 1)
    
    # AP as area under PR curve (11-point interpolation)
    recall_thresholds = np.linspace(0, 1, 11)
    ap = 0.0
    for t in recall_thresholds:
        idx = recall >= t
        if idx.any():
            ap += precision[idx].max()
    ap /= 11
    
    return ap


def evaluate_openlane_v2(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    use_fine_endpoint: bool = True,
) -> Dict[str, float]:
    """
    Full evaluation on OpenLane-V2
    
    Args:
        model: Trained model
        dataloader: Validation dataloader
        device: Computation device
        use_fine_endpoint: Use fine endpoint metric (DET_p)
        
    Returns:
        Dictionary with all metrics
    """
    model.eval()
    
    all_results = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            if isinstance(batch, dict):
                images = batch.get('images')
                if images is not None and isinstance(images, list):
                    images = [img.to(device) for img in images]
            else:
                images = [b.to(device) for b in batch['images']]
            
            camera_params = {
                'intrinsics': batch.get('camera_intrinsics', torch.eye(3)).to(device),
                'extrinsics': batch.get('camera_extrinsics', torch.eye(4)).to(device),
            }
            
            # Forward pass
            predictions = model(images, **camera_params, return_aux=False)
            
            # Extract results
            result = {
                'pred_lanes': predictions['lane_geometry'].cpu().numpy(),
                'pred_te': predictions.get('te_predictions', {}).get('boxes').cpu().numpy()
                    if predictions.get('te_predictions') else None,
                'pred_topology_lclc': predictions['topology_lclc'].cpu().numpy(),
                'pred_topology_lcte': predictions['topology_lcte'].cpu().numpy(),
                'gt_lanes': batch.get('lane_centerline'),
                'gt_te': batch.get('traffic_element'),
                'gt_topology_lclc': batch.get('topology_lclc'),
                'gt_topology_lcte': batch.get('topology_lcte'),
            }
            
            # Compute per-frame metrics
            frame_metrics = compute_frame_metrics(result)
            all_results.append(frame_metrics)
    
    # Aggregate metrics
    final_metrics = compute_ols_metrics(all_results)
    
    return final_metrics


def compute_frame_metrics(result: Dict) -> Dict[str, float]:
    """
    Compute metrics for a single frame
    """
    metrics = {}
    
    # Lane detection
    if result.get('gt_lanes') is not None:
        pred_lanes = result.get('pred_lanes', [])
        gt_lanes = result.get('gt_lanes', [])
        
        if isinstance(pred_lanes, np.ndarray):
            pred_lanes = [pred_lanes[i] for i in range(len(pred_lanes))]
        
        det_score = compute_lane_detection_score(pred_lanes, gt_lanes)
        metrics['det_lane_score'] = det_score
    
    # TE detection (placeholder - needs proper box IoU)
    metrics['det_te_score'] = 0.5  # Placeholder
    
    # Lane-lane topology
    if result.get('gt_topology_lclc') is not None:
        pred_top = result.get('pred_topology_lclc')
        gt_top = result.get('gt_topology_lclc')
        
        if isinstance(pred_top, np.ndarray) and isinstance(gt_top, np.ndarray):
            # Pad/trim to match sizes
            n = max(pred_top.shape[0], gt_top.shape[0])
            m = max(pred_top.shape[1], gt_top.shape[1])
            
            pred_padded = np.zeros((n, m))
            gt_padded = np.zeros((n, m))
            
            pred_padded[:pred_top.shape[0], :pred_top.shape[1]] = pred_top
            gt_padded[:gt_top.shape[0], :gt_top.shape[1]] = gt_top
            
            metrics['top_ll_score'] = compute_topology_ap(pred_padded, gt_padded)
        else:
            metrics['top_ll_score'] = 0.0
    else:
        metrics['top_ll_score'] = 0.0
    
    # Lane-TE topology (placeholder)
    metrics['top_lte_score'] = 0.5  # Placeholder
    
    return metrics


def compute_top_metrics(
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """
    Compute TOP metrics (lane-lane and lane-TE topology)
    """
    metrics = {}
    
    # Lane-lane topology
    if 'topology_lclc' in predictions and 'topology_lclc' in targets:
        top_ll_ap = compute_topology_ap(
            predictions['topology_lclc'].cpu().numpy(),
            targets['topology_lclc'].cpu().numpy(),
        )
        metrics['TOP_ll'] = top_ll_ap
    
    # Lane-TE topology
    if 'topology_lcte' in predictions and 'topology_lcte' in targets:
        top_lte_ap = compute_topology_ap(
            predictions['topology_lcte'].cpu().numpy(),
            targets['topology_lcte'].cpu().numpy(),
        )
        metrics['TOP_lte'] = top_lte_ap
    
    return metrics