"""
Combined Loss Function for Topology Reasoning

Includes:
1. Lane Detection Loss (focal + L1 geometry)
2. Endpoint Detection Loss (keypoint regression)
3. Traffic Element Detection Loss
4. Topology Relation Loss (focal for link prediction)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple


class FocalLoss(nn.Module):
    """
    Focal Loss for class imbalance
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [*,] Predicted probability
            target: [*,] Ground truth (0 or 1)
        """
        pred = pred.clamp(1e-6, 1 - 1e-6)
        
        pos_mask = target >= 0.5
        neg_mask = ~pos_mask
        
        # Focal weights
        pos_weight = (1 - pred) ** self.gamma
        neg_weight = pred ** self.gamma
        
        # Alpha weighting
        pos_alpha = self.alpha
        neg_alpha = 1 - self.alpha
        
        # Losses
        pos_loss = -pos_alpha * pos_weight * torch.log(pred) * pos_mask.float()
        neg_loss = -(1 - neg_alpha) * neg_weight * torch.log(1 - pred) * neg_mask.float()
        
        return (pos_loss + neg_loss).mean()


class TopologyLoss(nn.Module):
    """
    Combined loss for topology reasoning model
    
    Weights:
        - lambda_detection: Lane and TE detection
        - lambda_topology: Lane-lane and lane-TE topology
        - lambda_endpoint: Endpoint detection
    """
    
    def __init__(
        self,
        lambda_detection: float = 1.0,
        lambda_topology: float = 1.0,
        lambda_endpoint: float = 0.5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.lambda_detection = lambda_detection
        self.lambda_topology = lambda_topology
        self.lambda_endpoint = lambda_endpoint
        
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        
        # L1 loss for regression
        self.l1_loss = nn.L1Loss(reduction='mean')
        
        # GIoU loss for boxes
        self.giou_loss = GeneralizedIoULoss()
    
    def forward(
        self,
        predictions: Dict[str, Any],
        targets: Dict[str, Any],
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Compute total loss
        
        Args:
            predictions: Model output dictionary
            targets: Ground truth dictionary
            
        Returns:
            losses: Dictionary of individual losses
            total_loss: Combined weighted loss
        """
        losses = {}
        
        # === Detection Losses ===
        detection_loss = self._compute_detection_loss(predictions, targets)
        losses['detection'] = detection_loss * self.lambda_detection
        
        # === Endpoint Losses ===
        if predictions.get('start_points') is not None:
            endpoint_loss = self._compute_endpoint_loss(predictions, targets)
            losses['endpoint'] = endpoint_loss * self.lambda_endpoint
        
        # === Topology Losses ===
        topology_loss = self._compute_topology_loss(predictions, targets)
        losses['topology'] = topology_loss * self.lambda_topology
        
        # Total loss
        total_loss = sum(losses.values())
        losses['total'] = total_loss
        
        return losses, total_loss
    
    def _compute_detection_loss(
        self,
        predictions: Dict[str, Any],
        targets: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Lane and TE detection losses
        """
        loss = 0.0
        count = 0
        
        # Lane detection
        if 'lane_boxes' in predictions and 'lane_gt' in targets:
            lane_loss = self._lane_detection_loss(
                predictions['lane_boxes'],
                predictions['lane_geometry'],
                targets['lane_gt']
            )
            loss = loss + lane_loss
            count += 1
        
        # TE detection
        if 'te_predictions' in predictions and 'te_gt' in targets:
            te_loss = self._te_detection_loss(
                predictions['te_predictions'],
                targets['te_gt']
            )
            loss = loss + te_loss
            count += 1
        
        return loss / max(count, 1)
    
    def _lane_detection_loss(
        self,
        pred_boxes: torch.Tensor,
        pred_geometry: torch.Tensor,
        gt_lanes: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Lane detection: focal for presence + L1 for geometry
        """
        # Presence confidence (focal loss)
        # pred_boxes: [B, N, 34] = conf + geometry
        # Simplified: just use geometry loss for now
        # Real implementation needs Hungarian matching
        
        if 'geometry' in gt_lanes:
            # L1 loss on centerline points
            loss_geom = self.l1_loss(pred_geometry, gt_lanes['geometry'])
        else:
            loss_geom = 0.0
        
        return loss_geom
    
    def _te_detection_loss(
        self,
        pred_te: Dict[str, torch.Tensor],
        gt_te: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        TE detection: box GIoU + attribute classification
        """
        loss = 0.0
        count = 0
        
        if 'boxes' in pred_te and 'boxes' in gt_te:
            # GIoU loss
            loss_box = self.giou_loss(pred_te['boxes'], gt_te['boxes'])
            loss = loss + loss_box
            count += 1
        
        if 'attributes' in pred_te and 'attributes' in gt_te:
            # Cross entropy for attributes
            loss_attr = F.cross_entropy(
                pred_te['attributes'],
                gt_te['attributes'],
                reduction='mean'
            )
            loss = loss + loss_attr
            count += 1
        
        return loss / max(count, 1)
    
    def _compute_endpoint_loss(
        self,
        predictions: Dict[str, Any],
        targets: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Endpoint detection loss: L1 regression on start/end points
        """
        start_pred = predictions['start_points']   # [B, N, 3]
        end_pred   = predictions['end_points']     # [B, N, 3]
        
        gt_lanes = targets.get('lane_gt', {})
        
        # Ground truth endpoints
        gt_start = gt_lanes.get('start_points')    # [B, N, 2]
        gt_end   = gt_lanes.get('end_points')      # [B, N, 2]
        
        if gt_start is None or gt_end is None:
            return torch.tensor(0.0, device=start_pred.device)
        
        # Position losses (x, y only, not confidence)
        loss_start = self.l1_loss(start_pred[..., :2], gt_start)
        loss_end   = self.l1_loss(end_pred[..., :2], gt_end)
        
        # Confidence losses (if available)
        if 'start_valid' in gt_lanes and 'end_valid' in gt_lanes:
            start_valid = gt_lanes['start_valid']
            end_valid   = gt_lanes['end_valid']
            
            # BCE for confidence prediction
            loss_start_conf = F.binary_cross_entropy_with_logits(
                start_pred[..., 2], start_valid.float()
            )
            loss_end_conf = F.binary_cross_entropy_with_logits(
                end_pred[..., 2], end_valid.float()
            )
            conf_loss = (loss_start_conf + loss_end_conf) * 0.1  # Scale down
        else:
            conf_loss = 0.0
        
        return (loss_start + loss_end + conf_loss) / 2
    
    def _compute_topology_loss(
        self,
        predictions: Dict[str, Any],
        targets: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Topology link prediction loss (focal loss for imbalanced links)
        """
        pred_ll  = predictions['topology_lclc']  # [B, N_lane, N_lane]
        pred_lte = predictions['topology_lcte']  # [B, N_lane, N_te]
        
        gt_ll  = targets.get('topology_lclc')
        gt_lte = targets.get('topology_lcte')
        
        loss = 0.0
        count = 0
        
        # Lane-lane topology
        if gt_ll is not None:
            loss_ll = self.focal_loss(pred_ll, gt_ll)
            loss = loss + loss_ll
            count += 1
        
        # Lane-TE topology
        if gt_lte is not None:
            loss_lte = self.focal_loss(pred_lte, gt_lte)
            loss = loss + loss_lte
            count += 1
        
        return loss / max(count, 1)


class GeneralizedIoULoss(nn.Module):
    """
    Generalized IoU Loss for bounding box regression
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        pred_boxes: torch.Tensor,
        gt_boxes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_boxes: [*, 4] predicted boxes [x1, y1, x2, y2]
            gt_boxes: [*, 4] ground truth boxes
        """
        # Expand dims if needed
        if pred_boxes.dim() == 2:
            pred_boxes = pred_boxes.unsqueeze(0)
        if gt_boxes.dim() == 2:
            gt_boxes = gt_boxes.unsqueeze(0)
        
        # Calculate intersection
        x1 = torch.max(pred_boxes[..., 0], gt_boxes[..., 0])
        y1 = torch.max(pred_boxes[..., 1], gt_boxes[..., 1])
        x2 = torch.min(pred_boxes[..., 2], gt_boxes[..., 2])
        y2 = torch.min(pred_boxes[..., 3], gt_boxes[..., 3])
        
        intersection = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        
        # Union
        area_pred = (pred_boxes[..., 2] - pred_boxes[..., 0]) * (pred_boxes[..., 3] - pred_boxes[..., 1])
        area_gt   = (gt_boxes[..., 2] - gt_boxes[..., 0]) * (gt_boxes[..., 3] - gt_boxes[..., 1])
        union = area_pred + area_gt - intersection
        
        # IoU
        iou = intersection / (union + 1e-8)
        
        # Enclosing box for GIoU
        x1_enclosing = torch.min(pred_boxes[..., 0], gt_boxes[..., 0])
        y1_enclosing = torch.min(pred_boxes[..., 1], gt_boxes[..., 1])
        x2_enclosing = torch.max(pred_boxes[..., 2], gt_boxes[..., 2])
        y2_enclosing = torch.max(pred_boxes[..., 3], gt_boxes[..., 3])
        
        area_enclosing = (x2_enclosing - x1_enclosing).clamp(min=0) * (y2_enclosing - y1_enclosing).clamp(min=0)
        
        # GIoU
        giou = iou - (area_enclosing - union) / (area_enclosing + 1e-8)
        
        return (1 - giou).mean()