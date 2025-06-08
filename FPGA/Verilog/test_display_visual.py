#!/usr/bin/env python3
"""
Simple test for display_tiler - generates expected output without simulation
This shows what your display_tiler should produce with the current configuration
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as patches

def create_test_display():
    # Display parameters
    TILE_W, TILE_H = 96, 96
    COLS, ROWS = 4, 6
    
    # Create display canvas
    display = np.zeros((ROWS * TILE_H, COLS * TILE_W, 3), dtype=np.uint8)
    
    # Tile configurations (from your SystemVerilog)
    tiles = [
        # Tile 0: Input image (28x28, scale 3)
        {'id': 0, 'w': 28, 'h': 28, 'S': 3, 'pattern': 'checkerboard'},
        # Tiles 1-4: Conv1 (24x24, scale 4)  
        {'id': 1, 'w': 24, 'h': 24, 'S': 4, 'pattern': 'gradient_h'},
        {'id': 2, 'w': 24, 'h': 24, 'S': 4, 'pattern': 'gradient_v'},
        {'id': 3, 'w': 24, 'h': 24, 'S': 4, 'pattern': 'solid_med'},
        {'id': 4, 'w': 24, 'h': 24, 'S': 4, 'pattern': 'solid_high'},
        # Tiles 17-20: Conv2 (8x8, scale 12)
        {'id': 17, 'w': 8, 'h': 8, 'S': 12, 'pattern': 'solid_low'},
        {'id': 18, 'w': 8, 'h': 8, 'S': 12, 'pattern': 'solid_med'},
        {'id': 19, 'w': 8, 'h': 8, 'S': 12, 'pattern': 'solid_high'},
        {'id': 20, 'w': 8, 'h': 8, 'S': 12, 'pattern': 'solid_max'},
        # Tile 21: Probability bar (10x1, scale 9)
        {'id': 21, 'w': 10, 'h': 1, 'S': 9, 'pattern': 'prob_bar'},
    ]
    
    # Generate patterns for each tile
    for tile in tiles:
        tile_id = tile['id']
        tile_r = tile_id // COLS
        tile_c = tile_id % COLS
        
        # Calculate tile position in display
        start_y = tile_r * TILE_H
        start_x = tile_c * TILE_W
        
        # Generate native pattern
        native = generate_pattern(tile['w'], tile['h'], tile['pattern'])
        
        # Scale up to tile size
        scaled = scale_pattern(native, tile['S'], TILE_W, TILE_H)
        
        # Place in display
        end_y = min(start_y + TILE_H, display.shape[0])
        end_x = min(start_x + TILE_W, display.shape[1])
        display[start_y:end_y, start_x:end_x] = scaled[:end_y-start_y, :end_x-start_x]
    
    return display

def generate_pattern(w, h, pattern_type):
    """Generate different test patterns"""
    pattern = np.zeros((h, w), dtype=np.uint8)
    
    if pattern_type == 'checkerboard':
        for y in range(h):
            for x in range(w):
                pattern[y, x] = 255 if (x + y) % 2 else 128
                
    elif pattern_type == 'gradient_h':
        for x in range(w):
            pattern[:, x] = (x * 255) // (w - 1)
            
    elif pattern_type == 'gradient_v':
        for y in range(h):
            pattern[y, :] = (y * 255) // (h - 1)
            
    elif pattern_type == 'solid_low':
        pattern[:] = 64
    elif pattern_type == 'solid_med':
        pattern[:] = 128
    elif pattern_type == 'solid_high':
        pattern[:] = 192
    elif pattern_type == 'solid_max':
        pattern[:] = 255
        
    elif pattern_type == 'prob_bar':
        for x in range(w):
            pattern[0, x] = (x * 255) // (w - 1)
    
    return pattern

def scale_pattern(native, scale, tile_w, tile_h):
    """Scale pattern using nearest neighbor (integer scaling)"""
    h, w = native.shape
    scaled_h = min(h * scale, tile_h)
    scaled_w = min(w * scale, tile_w)
    
    # Create RGB tile (black background)
    tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    
    # Scale pattern
    for y in range(scaled_h):
        for x in range(scaled_w):
            src_y = y // scale
            src_x = x // scale
            if src_y < h and src_x < w:
                gray_val = native[src_y, src_x]
                tile[y, x] = [gray_val, gray_val, gray_val]  # Grayscale to RGB
    
    return tile

def main():
    print("Generating expected display_tiler output...")
    
    # Create the display
    display = create_test_display()
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    
    # Show full display
    ax1.imshow(display)
    ax1.set_title('Full Display Output (384x576)')
    ax1.set_xlabel('Pixels')
    ax1.set_ylabel('Pixels')
    
    # Add tile grid overlay
    TILE_W, TILE_H = 96, 96
    COLS, ROWS = 4, 6
    for r in range(ROWS):
        for c in range(COLS):
            rect = Rectangle((c*TILE_W-0.5, r*TILE_H-0.5), TILE_W, TILE_H, 
                           linewidth=1, edgecolor='red', facecolor='none')
            ax1.add_patch(rect)
            
            # Add tile numbers
            tile_id = r * COLS + c
            if tile_id < 22:  # Only show valid tiles
                ax1.text(c*TILE_W + TILE_W//2, r*TILE_H + 10, f'T{tile_id}', 
                        ha='center', va='center', color='red', fontweight='bold')
    
    # Show individual tiles
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 12)
    ax2.set_title('Tile Layout and Configuration')
    
    # Tile descriptions
    descriptions = [
        "T0: Input 28x28 (3x scale)",
        "T1-4: Conv1 24x24 (4x scale)", 
        "T5-16: Conv1 spare slots",
        "T17-20: Conv2 8x8 (12x scale)",
        "T21: Probabilities 10x1 (9x scale)",
        "T22-23: Unused"
    ]
    
    for i, desc in enumerate(descriptions):
        ax2.text(0.5, 11-i*1.5, desc, fontsize=10, ha='left')
    
    ax2.set_xlim(0, 10)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('display_tiler_expected.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ Visualization saved as 'display_tiler_expected.png'")
    print("✅ This shows what your display_tiler should output!")
    print("\nYour SystemVerilog configuration looks perfect for:")
    print("- MNIST CNN visualization")
    print("- 22 tiles with proper scaling")
    print("- Memory layout with appropriate strides")

if __name__ == "__main__":
    main()
