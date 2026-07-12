import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set matplotlib backend to Agg to prevent GUI popups
import matplotlib.pyplot as plt
from config import SAVE_DEBUG_IMAGES

def normalize_cluster_map(cluster_map):
    """
    Refinement 4: Normalize cluster maps before colormap rendering.
    """
    cmap_float = cluster_map.astype(np.float32)
    max_val = cmap_float.max()
    if max_val > 0:
        cmap_float /= max_val
    return cmap_float

def generate_comparison_dashboard(
    reference_image, reference_mask, ref_cluster_map, reference_color, ref_rgb_list, ref_hex, ref_exact, ref_meta,
    target_image, target_mask, tgt_cluster_map, target_color, tgt_rgb_list, tgt_hex, tgt_exact, tgt_meta,
    delta_e, rgb_dist, same_meta, match_status, output_path
):
    """
    Generate a highly refined, premium 3-row by 4-column visual Comparison Dashboard.
    """
    # 1. Normalize cluster maps
    ref_norm_cmap = normalize_cluster_map(ref_cluster_map)
    tgt_norm_cmap = normalize_cluster_map(tgt_cluster_map)

    # 2. Draw subplots with Matplotlib
    fig, axes = plt.subplots(3, 4, figsize=(16, 12.5), facecolor='#FDFDFD')
    cmap_clusters = plt.get_cmap("tab10")

    # Consistent font dicts for elegant presentation
    title_font = {'fontsize': 12, 'fontweight': 'bold', 'color': '#2C3E50'}

    # Helper function to style solid dominant color patch with crisp boundaries
    def render_color_patch(ax, color, title):
        # Create solid True RGB patch
        patch = np.zeros((100, 100, 3), dtype=np.uint8)
        patch[:] = color
        ax.imshow(patch, interpolation='nearest')
        
        # Add a crisp clean solid border around the patch to prevent bleeding/distortion
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#2C3E50')
            spine.set_linewidth(2)
        ax.set_title(title, **title_font, pad=10)
        ax.set_xticks([])
        ax.set_yticks([])

    # --- Row 1: Reference Panels ---
    axes[0, 0].imshow(reference_image)
    axes[0, 0].set_title("Reference: Original", **title_font, pad=10)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(reference_mask, cmap="gray", interpolation='nearest')
    axes[0, 1].set_title("Reference: Foreground Mask", **title_font, pad=10)
    axes[0, 1].axis("off")

    axes[0, 2].imshow(ref_norm_cmap, cmap=cmap_clusters, interpolation="nearest")
    axes[0, 2].set_title("Reference: Cluster Map (Normalized)", **title_font, pad=10)
    axes[0, 2].axis("off")

    ref_dominant_title = (
        f"Reference: Dominant Color\n"
        f"RGB: {ref_rgb_list}\n"
        f"HEX: {ref_hex}\n"
        f"EXACT: {ref_exact}\n"
        f"META: {ref_meta}"
    )
    render_color_patch(axes[0, 3], reference_color, ref_dominant_title)

    # --- Row 2: Target Panels ---
    axes[1, 0].imshow(target_image)
    axes[1, 0].set_title("Target: Original", **title_font, pad=10)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(target_mask, cmap="gray", interpolation='nearest')
    axes[1, 1].set_title("Target: Foreground Mask", **title_font, pad=10)
    axes[1, 1].axis("off")

    axes[1, 2].imshow(tgt_norm_cmap, cmap=cmap_clusters, interpolation="nearest")
    axes[1, 2].set_title("Target: Cluster Map (Normalized)", **title_font, pad=10)
    axes[1, 2].axis("off")

    tgt_dominant_title = (
        f"Target: Dominant Color\n"
        f"RGB: {tgt_rgb_list}\n"
        f"HEX: {tgt_hex}\n"
        f"EXACT: {tgt_exact}\n"
        f"META: {tgt_meta}"
    )
    render_color_patch(axes[1, 3], target_color, tgt_dominant_title)

    # --- Row 3: Centered Comparison Text Panel (Hierarchical Match Condition) ---
    for col in range(4):
        axes[2, col].axis("off")

    comparison_text = (
        "================ FINAL COMPARISON =================\n\n"
        f"{'DeltaE Perceptual Diff (CIE 2000)':<34}: {delta_e:.2f}\n"
        f"{'RGB Euclidean Distance':<34}: {rgb_dist:.2f}\n"
        f"{'Same MetaFamily?':<34}: {same_meta}\n"
        f"{'Match Status':<34}: {match_status}\n\n"
        "==================================================="
    )

    # High-fidelity monospace text block styled with round padded box
    axes[2, 1].text(
        0.5, 0.5, comparison_text, fontsize=13, fontweight='bold',
        family='monospace', color='#1A252F', ha='center', va='center',
        bbox=dict(
            boxstyle="round,pad=1.4",
            facecolor='#F4F6F7',
            edgecolor='#BDC3C7',
            lw=2
        )
    )

    plt.suptitle(
        "Perceptual Text-Color Semantic Extraction & Comparison Dashboard",
        fontsize=16, fontweight='bold', color='#2C3E50', y=0.97
    )
    
    # Elegant spacing
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the main dashboard output directory safety
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()

    # Refinement 8: Optional Debug Images
    if SAVE_DEBUG_IMAGES:
        try:
            # Save intermediate visual sub-plots
            plt.imsave(os.path.join(output_dir, "ref_foreground_mask.png"), reference_mask, cmap='gray')
            plt.imsave(os.path.join(output_dir, "tgt_foreground_mask.png"), target_mask, cmap='gray')
            
            # Save normalized cluster map
            plt.imsave(os.path.join(output_dir, "ref_cluster_map.png"), ref_norm_cmap, cmap=cmap_clusters)
            plt.imsave(os.path.join(output_dir, "tgt_cluster_map.png"), tgt_norm_cmap, cmap=cmap_clusters)
            
            # Save solid patches
            ref_patch = np.zeros((100, 100, 3), dtype=np.uint8)
            ref_patch[:] = reference_color
            plt.imsave(os.path.join(output_dir, "ref_dominant_patch.png"), ref_patch)
            
            tgt_patch = np.zeros((100, 100, 3), dtype=np.uint8)
            tgt_patch[:] = target_color
            plt.imsave(os.path.join(output_dir, "tgt_dominant_patch.png"), tgt_patch)
        except Exception:
            pass
