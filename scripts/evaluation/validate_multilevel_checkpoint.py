#!/usr/bin/env python3
"""
Quick validation script to verify multi-level checkpoint loads correctly
and sensor distances are computed properly.
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "anomaly-detection"))

from gdn_processor import GDNPredictor

def validate_multilevel_checkpoint(checkpoint_path: str):
    """Validate that multi-level checkpoint loads and works correctly."""
    print("=" * 80)
    print("Multi-Level Checkpoint Validation")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}\n")
    
    # Load checkpoint
    print("1. Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Check for multi-level centers
    has_window_centers = "window_centers" in checkpoint
    has_sensor_centers = "sensor_centers" in checkpoint
    has_center_loss_state = "center_loss_state_dict" in checkpoint
    
    print(f"   ✓ Checkpoint loaded")
    print(f"   - Window centers: {has_window_centers}")
    print(f"   - Sensor centers: {has_sensor_centers}")
    print(f"   - Center loss state: {has_center_loss_state}")
    
    if not (has_window_centers and has_sensor_centers):
        print("\n   ⚠ WARNING: Checkpoint missing multi-level centers!")
        return False
    
    # Check shapes
    window_centers = checkpoint["window_centers"]
    sensor_centers = checkpoint["sensor_centers"]
    
    print(f"\n2. Checking center shapes...")
    print(f"   Window centers shape: {window_centers.shape} (expected: (2, hidden_dim))")
    print(f"   Sensor centers shape: {sensor_centers.shape} (expected: (8, 2, hidden_dim))")
    
    if window_centers.shape[0] != 2:
        print("   ⚠ Window centers should have 2 classes")
        return False
    
    if sensor_centers.shape[0] != 8 or sensor_centers.shape[1] != 2:
        print("   ⚠ Sensor centers should be (8, 2, hidden_dim)")
        return False
    
    hidden_dim = window_centers.shape[1]
    print(f"   Hidden dim: {hidden_dim}")
    
    # Check separations
    if "separations" in checkpoint:
        separations = checkpoint["separations"]
        print(f"\n3. Checking separations...")
        print(f"   Window separation: {separations.get('window_separation', 'N/A'):.4f}")
        print(f"   Sensor mean separation: {separations.get('sensor_mean_separation', 'N/A'):.4f}")
        print(f"   Sensor min separation: {separations.get('sensor_min_separation', 'N/A'):.4f}")
        print(f"   Sensor max separation: {separations.get('sensor_max_separation', 'N/A'):.4f}")
    
    # Test loading with GDNPredictor
    print(f"\n4. Testing GDNPredictor loading...")
    try:
        sensor_names = checkpoint.get("sensor_names", [
            "ENGINE_RPM ()",
            "VEHICLE_SPEED ()",
            "THROTTLE ()",
            "ENGINE_LOAD ()",
            "COOLANT_TEMPERATURE ()",
            "INTAKE_MANIFOLD_PRESSURE ()",
            "SHORT_TERM_FUEL_TRIM_BANK_1 ()",
            "LONG_TERM_FUEL_TRIM_BANK_1 ()",
        ])
        
        predictor = GDNPredictor(
            model_path=checkpoint_path,
            sensor_names=sensor_names,
            window_size=checkpoint.get("window_size", 300),
            embed_dim=checkpoint.get("embed_dim", 64),
            top_k=checkpoint.get("top_k", 3),
            hidden_dim=hidden_dim,
            device="cpu",
        )
        print(f"   ✓ GDNPredictor loaded successfully")
    except Exception as e:
        print(f"   ✗ Failed to load GDNPredictor: {e}")
        return False
    
    # Test with dummy data
    print(f"\n5. Testing distance computation with dummy data...")
    try:
        # Create dummy window (batch_size=1, window_size=300, num_sensors=8)
        dummy_window = np.random.randn(1, 300, 8).astype(np.float32)
        
        # Get embeddings
        from scripts.evaluation.evaluate_gdn import get_embeddings_and_distances
        
        embeddings, distances, normal_center, sensor_embeddings, sensor_distances = (
            get_embeddings_and_distances(
                predictor,
                dummy_window,
                batch_size=1,
                use_tta=False,
            )
        )
        
        print(f"   ✓ Embeddings extracted")
        print(f"   - Window embeddings shape: {embeddings.shape}")
        print(f"   - Window distances shape: {distances.shape}")
        print(f"   - Sensor embeddings shape: {sensor_embeddings.shape}")
        print(f"   - Sensor distances shape: {sensor_distances.shape}")
        
        if sensor_distances is not None:
            # Check distance ratio
            window_dist_mean = np.mean(distances) if distances is not None else 0
            sensor_dist_mean = np.mean(sensor_distances) if sensor_distances is not None else 0
            
            if window_dist_mean > 0 and sensor_dist_mean > 0:
                ratio = window_dist_mean / sensor_dist_mean
                print(f"\n6. Distance ratio (window/sensor): {ratio:.4f}")
                print(f"   Target: ~1.0 (compatible embedding spaces)")
                
                if 0.5 < ratio < 2.0:
                    print(f"   ✓ Ratio is reasonable (compatible spaces)")
                else:
                    print(f"   ⚠ Ratio is outside expected range")
            else:
                print(f"\n6. Could not compute distance ratio (zero distances)")
        else:
            print(f"   ⚠ Sensor distances not computed")
            
    except Exception as e:
        print(f"   ✗ Failed to compute distances: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n{'=' * 80}")
    print("✓ Validation complete - Multi-level checkpoint is working!")
    print(f"{'=' * 80}\n")
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate multi-level checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/stage2_multilevel.pt",
        help="Path to multi-level checkpoint",
    )
    
    args = parser.parse_args()
    
    success = validate_multilevel_checkpoint(args.checkpoint)
    sys.exit(0 if success else 1)
