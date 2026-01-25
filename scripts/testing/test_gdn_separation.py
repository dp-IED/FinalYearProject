#!/usr/bin/env python3
"""
Test the trained GDN model with separation loss and visualize separation.

Plots the distribution of distances to normal center for normal vs anomalous windows.
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Add evaluation directory for metrics
eval_dir = str(project_root / "llm" / "evaluation")
if eval_dir not in sys.path:
    sys.path.insert(0, eval_dir)

# Import GDN processor
sys.path.insert(0, str(project_root / "anomaly-detection"))
from gdn_processor import GDNPredictor


def test_separation(
    model_path: str = "anomaly-detection/best_multilabel_gdn_separation.pt",
    dataset_path: str = "llm/evaluation/shared_dataset/test.npz",
    output_plot: str = "results/gdn_separation_test.png"
):
    """Test model separation and create visualization."""
    
    print("=" * 80)
    print("GDN SEPARATION TEST")
    print("=" * 80)
    print()
    
    # Load dataset
    print("1. Loading test dataset...")
    data = np.load(dataset_path, allow_pickle=True)
    normalized_windows = data['normalized_windows']
    sensor_labels_true = data['sensor_labels']
    window_labels_true = data['window_labels']
    
    sensor_names = ['ENGINE_RPM', 'VEHICLE_SPEED', 'THROTTLE', 'ENGINE_LOAD',
                    'COOLANT_TEMPERATURE', 'INTAKE_MANIFOLD_PRESSURE',
                    'SHORT_TERM_FUEL_TRIM_BANK_1', 'LONG_TERM_FUEL_TRIM_BANK_1']
    
    num_windows = len(normalized_windows)
    print(f"   Loaded {num_windows} windows")
    print()
    
    # Load GDN model
    print("2. Loading GDN model...")
    try:
        predictor = GDNPredictor(
            model_path=model_path,
            sensor_names=sensor_names,
            window_size=300,
            embed_dim=32,
            top_k=3,
            hidden_dim=32,
            device='cpu'
        )
        print(f"   ✓ Model loaded from: {model_path}")
        
        # Check if distance-based scoring is enabled
        if predictor.use_distance_scoring:
            print(f"   ✓ Distance-based scoring enabled")
            if hasattr(predictor, 'normal_center') and predictor.normal_center is not None:
                print(f"   ✓ Normal center available for distance computation")
        else:
            print(f"   ⚠️  Distance-based scoring not enabled (using probability-based)")
    except Exception as e:
        print(f"   ✗ Error loading model: {e}")
        return
    
    print()
    
    # Process through GDN
    print("3. Processing windows through GDN...")
    kg_data = predictor.process_for_kg(
        X_windows=normalized_windows,
        sensor_labels=sensor_labels_true,
        window_labels=window_labels_true,
        batch_size=32
    )
    
    gdn_predictions = kg_data['gdn_predictions']  # (num_windows, num_sensors)
    print(f"   GDN predictions shape: {gdn_predictions.shape}")
    print()
    
    # Compute distance to normal center if available
    print("4. Computing distances to normal center...")
    
    if predictor.use_distance_scoring and hasattr(predictor, 'normal_center') and predictor.normal_center is not None:
        # Compute raw distances from embeddings (not normalized)
        print(f"   ✓ Computing raw distances from embeddings...")
        
        # Get embeddings for all windows
        X_tensor = torch.from_numpy(normalized_windows).float()
        num_windows = len(X_tensor)
        
        predictor.model.eval()
        all_embeddings = []
        
        with torch.no_grad():
            batch_size = 32
            for i in range(0, num_windows, batch_size):
                batch = X_tensor[i:i+batch_size]
                embeddings = predictor.model.get_embeddings(batch)
                all_embeddings.append(embeddings.cpu().numpy())
        
        all_embeddings = np.concatenate(all_embeddings, axis=0)  # (num_windows, embed_dim)
        
        # Get normal center
        normal_center = predictor.normal_center.cpu().numpy() if hasattr(predictor.normal_center, 'cpu') else predictor.normal_center
        
        # Compute raw L2 distances (not normalized)
        distances = np.linalg.norm(all_embeddings - normal_center, axis=1)  # (num_windows,)
        window_distances = distances
        print(f"   ✓ Computed raw L2 distances (mean: {np.mean(distances):.4f}, std: {np.std(distances):.4f})")
    else:
        # Fallback: compute distances manually from embeddings
        print(f"   ⚠️  Computing distances manually from embeddings...")
        
        # Get embeddings for all windows
        X_tensor = torch.from_numpy(normalized_windows).float()
        num_windows = len(X_tensor)
        
        predictor.model.eval()
        all_embeddings = []
        
        with torch.no_grad():
            batch_size = 32
            for i in range(0, num_windows, batch_size):
                batch = X_tensor[i:i+batch_size]
                embeddings = predictor.model.get_embeddings(batch)
                all_embeddings.append(embeddings.cpu().numpy())
        
        all_embeddings = np.concatenate(all_embeddings, axis=0)  # (num_windows, embed_dim)
        
        # Try to get normal center from checkpoint
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'center_loss_state_dict' in checkpoint:
                # Import CenterLoss
                import importlib.util
                train_script_path = project_root / "anomaly-detection" / "train_gdn_separation.py"
                if train_script_path.exists():
                    spec = importlib.util.spec_from_file_location("train_gdn_separation", train_script_path)
                    train_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(train_module)
                    CenterLoss = train_module.CenterLoss
                    
                    center_loss = CenterLoss(embed_dim=32, num_classes=2)
                    center_loss.load_state_dict(checkpoint['center_loss_state_dict'])
                    normal_center = center_loss.centers[0].detach().cpu().numpy()  # (embed_dim,)
                    
                    # Compute distances
                    distances = np.linalg.norm(all_embeddings - normal_center, axis=1)  # (num_windows,)
                    window_distances = distances
                    print(f"   ✓ Computed distances from embeddings to normal center")
                else:
                    # Fallback: use max probability as proxy
                    window_distances = np.max(gdn_predictions, axis=1)
                    print(f"   ⚠️  Using max probability as proxy for distance")
            else:
                # Fallback: use max probability as proxy
                window_distances = np.max(gdn_predictions, axis=1)
                print(f"   ⚠️  No center loss in checkpoint, using max probability as proxy")
        except Exception as e:
            print(f"   ⚠️  Error computing distances: {e}")
            # Fallback: use max probability as proxy
            window_distances = np.max(gdn_predictions, axis=1)
            print(f"   Using max probability as proxy for distance")
    
    print()
    
    # Identify normal vs faulty windows
    print("5. Analyzing separation...")
    is_faulty_window = (sensor_labels_true.sum(axis=1) > 0)
    normal_mask = ~is_faulty_window
    faulty_mask = is_faulty_window
    
    num_normal = np.sum(normal_mask)
    num_faulty = np.sum(faulty_mask)
    
    print(f"   Normal windows: {num_normal}")
    print(f"   Faulty windows: {num_faulty}")
    print()
    
    # Extract scores
    normal_scores = window_distances[normal_mask]
    faulty_scores = window_distances[faulty_mask]
    
    # Compute statistics
    normal_mean = np.mean(normal_scores)
    faulty_mean = np.mean(faulty_scores)
    separation = faulty_mean - normal_mean
    
    normal_std = np.std(normal_scores)
    faulty_std = np.std(faulty_scores)
    
    normal_p95 = np.percentile(normal_scores, 95)
    faulty_p5 = np.percentile(faulty_scores, 5)
    
    # Calculate overlap
    normal_above_faulty_mean = np.sum(normal_scores > faulty_mean) / len(normal_scores) * 100
    faulty_below_normal_mean = np.sum(faulty_scores < normal_mean) / len(faulty_scores) * 100
    overlap_pct = (normal_above_faulty_mean + faulty_below_normal_mean) / 2
    
    # Separation ratio
    separation_ratio = faulty_mean / (normal_mean + 1e-8)
    
    print("=" * 80)
    print("SEPARATION RESULTS")
    print("=" * 80)
    print()
    print(f"Normal windows:")
    print(f"   Mean: {normal_mean:.6f} ± {normal_std:.6f}")
    print(f"   Min: {np.min(normal_scores):.6f}, Max: {np.max(normal_scores):.6f}")
    print(f"   P95: {normal_p95:.6f}")
    print()
    print(f"Faulty windows:")
    print(f"   Mean: {faulty_mean:.6f} ± {faulty_std:.6f}")
    print(f"   Min: {np.min(faulty_scores):.6f}, Max: {np.max(faulty_scores):.6f}")
    print(f"   P5: {faulty_p5:.6f}")
    print()
    print(f"Separation:")
    print(f"   Mean difference: {separation:.6f}")
    print(f"   Separation ratio: {separation_ratio:.2f}x")
    print(f"   Overlap: {overlap_pct:.1f}%")
    print()
    
    if separation >= 0.3:
        print(f"✓ SUFFICIENT SEPARATION (>= 0.3)")
    else:
        print(f"✗ INSUFFICIENT SEPARATION (< 0.3)")
    print()
    
    # Create visualization
    print("6. Creating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Histogram of scores
    ax1 = axes[0, 0]
    ax1.hist(normal_scores, bins=50, alpha=0.6, label=f'Normal (n={num_normal})', 
             density=True, color='green', edgecolor='black', linewidth=0.5)
    ax1.hist(faulty_scores, bins=50, alpha=0.6, label=f'Faulty (n={num_faulty})', 
             density=True, color='red', edgecolor='black', linewidth=0.5)
    ax1.axvline(x=normal_mean, color='green', linestyle='--', linewidth=2, label=f'Normal mean: {normal_mean:.4f}')
    ax1.axvline(x=faulty_mean, color='red', linestyle='--', linewidth=2, label=f'Faulty mean: {faulty_mean:.4f}')
    ax1.set_xlabel('Distance to Normal Center', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax1.set_title('Distribution of Distances: Normal vs Faulty Windows', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: CDF
    ax2 = axes[0, 1]
    sorted_normal = np.sort(normal_scores)
    sorted_faulty = np.sort(faulty_scores)
    cdf_normal = np.arange(1, len(sorted_normal) + 1) / len(sorted_normal)
    cdf_faulty = np.arange(1, len(sorted_faulty) + 1) / len(sorted_faulty)
    
    ax2.plot(sorted_normal, cdf_normal, linewidth=2.5, label='Normal', color='green')
    ax2.plot(sorted_faulty, cdf_faulty, linewidth=2.5, label='Faulty', color='red')
    ax2.axvline(x=normal_mean, color='green', linestyle='--', alpha=0.5)
    ax2.axvline(x=faulty_mean, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Distance to Normal Center', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
    ax2.set_title('CDF: Score Distribution', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Box plot
    ax3 = axes[1, 0]
    box_data = [normal_scores, faulty_scores]
    bp = ax3.boxplot(box_data, labels=['Normal', 'Faulty'], patch_artist=True)
    bp['boxes'][0].set_facecolor('green')
    bp['boxes'][0].set_alpha(0.6)
    bp['boxes'][1].set_facecolor('red')
    bp['boxes'][1].set_alpha(0.6)
    ax3.set_ylabel('Distance to Normal Center', fontsize=12, fontweight='bold')
    ax3.set_title('Box Plot: Score Distribution', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Statistics summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    stats_text = f'''
    SEPARATION STATISTICS
    
    Normal Windows (n={num_normal}):
    ├─ Mean: {normal_mean:.6f}
    ├─ Std:  {normal_std:.6f}
    ├─ Min:   {np.min(normal_scores):.6f}
    ├─ Max:   {np.max(normal_scores):.6f}
    ├─ P95:   {normal_p95:.6f}
    
    Faulty Windows (n={num_faulty}):
    ├─ Mean: {faulty_mean:.6f}
    ├─ Std:  {faulty_std:.6f}
    ├─ Min:   {np.min(faulty_scores):.6f}
    ├─ Max:   {np.max(faulty_scores):.6f}
    ├─ P5:    {faulty_p5:.6f}
    
    Separation Metrics:
    ├─ Mean Difference: {separation:.6f}
    ├─ Separation Ratio: {separation_ratio:.2f}x
    ├─ Overlap: {overlap_pct:.1f}%
    
    Status: {'✓ SUFFICIENT' if separation >= 0.3 else '✗ INSUFFICIENT'}
    '''
    
    ax4.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
             verticalalignment='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_plot)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Plot saved to: {output_path}")
    print()
    print("=" * 80)
    print("✓ Test completed successfully!")
    print("=" * 80)
    
    return {
        'normal_mean': normal_mean,
        'faulty_mean': faulty_mean,
        'separation': separation,
        'separation_ratio': separation_ratio,
        'overlap_pct': overlap_pct
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test GDN model separation')
    parser.add_argument('--model', type=str,
                       default='anomaly-detection/best_multilabel_gdn_separation.pt',
                       help='Path to trained model')
    parser.add_argument('--dataset', type=str,
                       default='llm/evaluation/shared_dataset/test.npz',
                       help='Path to test dataset')
    parser.add_argument('--output', type=str,
                       default='results/gdn_separation_test.png',
                       help='Output plot path')
    
    args = parser.parse_args()
    
    test_separation(args.model, args.dataset, args.output)
