#!/usr/bin/env python3
"""
Enhanced GNSS Position Plotting Tool for rnx2rtkp output
Features:
- Fixed ±50cm range centered on GT coordinates
- 1-sigma confidence bands with transparency
- Color coding by positioning quality
- Accuracy statistics and histograms
- No 2D ground track plot
"""

import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

# Ground Truth coordinates from readme.md
GT_COORDINATES = {
    'latitude': 36.068742145,
    'longitude': 140.128346910,
    'height': 112.5059
}

# Quality factor color mapping
QUALITY_COLORS = {
    1: '#2E7D32',  # FIX - Dark green
    2: '#1976D2',  # FLOAT - Blue
    3: '#FF8F00',  # SBAS - Orange
    4: '#7B1FA2',  # DGPS - Purple
    5: '#C62828',  # SINGLE - Red
    6: '#00796B',  # PPP - Teal
}

QUALITY_NAMES = {
    1: 'FIX',
    2: 'FLOAT',
    3: 'SBAS',
    4: 'DGPS',
    5: 'SINGLE',
    6: 'PPP'
}

def read_rnx2rtkp_file(filepath):
    """Read rnx2rtkp output file and parse data"""
    data_lines = []
    
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('%') or line.strip() == '':
                continue
            data_lines.append(line.strip())
    
    columns = ['datetime', 'latitude', 'longitude', 'height', 'Q', 'ns', 
               'sdn', 'sde', 'sdu', 'sdne', 'sdeu', 'sdun', 'age', 'ratio']
    
    data = []
    for line in data_lines:
        parts = line.split()
        if len(parts) >= 14:
            datetime_str = parts[0] + ' ' + parts[1]
            row = [datetime_str] + [float(x) for x in parts[2:15]]
            data.append(row)
    
    df = pd.DataFrame(data, columns=columns)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Calculate errors relative to GT
    df['lat_error'] = (df['latitude'] - GT_COORDINATES['latitude']) * 111320  # meters
    df['lon_error'] = (df['longitude'] - GT_COORDINATES['longitude']) * 111320 * np.cos(np.radians(GT_COORDINATES['latitude']))  # meters
    df['hgt_error'] = df['height'] - GT_COORDINATES['height']  # meters
    
    return df

def plot_enhanced_time_series(df, output_file=None):
    """Plot enhanced position time series with GT-centered view and confidence bands"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Create time array for plotting
    time_array = df['datetime']
    
    # Plot 1: Latitude errors
    ax = axes[0]
    
    # Plot confidence bands for each quality
    for q in sorted(df['Q'].unique()):
        mask = df['Q'] == q
        if mask.sum() > 0:
            subset = df[mask]
            ax.fill_between(subset['datetime'], 
                           subset['lat_error'] - subset['sdn'],
                           subset['lat_error'] + subset['sdn'],
                           alpha=0.2, color=QUALITY_COLORS.get(q, 'gray'))
    
    # Plot points colored by quality
    for q in sorted(df['Q'].unique()):
        mask = df['Q'] == q
        if mask.sum() > 0:
            subset = df[mask]
            ax.scatter(subset['datetime'], subset['lat_error'], 
                      c=QUALITY_COLORS.get(q, 'gray'), 
                      s=8, alpha=0.7, label=f'{QUALITY_NAMES.get(q, f"Q{q}")} ({mask.sum()})')
    
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='GT')
    ax.set_ylabel('Latitude Error (m)')
    ax.set_ylim(-0.5, 0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_title('GNSS Position Errors (GT-centered, ±50cm range)')
    
    # Plot 2: Longitude errors
    ax = axes[1]
    
    # Plot confidence bands for each quality
    for q in sorted(df['Q'].unique()):
        mask = df['Q'] == q
        if mask.sum() > 0:
            subset = df[mask]
            ax.fill_between(subset['datetime'], 
                           subset['lon_error'] - subset['sde'],
                           subset['lon_error'] + subset['sde'],
                           alpha=0.2, color=QUALITY_COLORS.get(q, 'gray'))
    
    # Plot points colored by quality
    for q in sorted(df['Q'].unique()):
        mask = df['Q'] == q
        if mask.sum() > 0:
            subset = df[mask]
            ax.scatter(subset['datetime'], subset['lon_error'], 
                      c=QUALITY_COLORS.get(q, 'gray'), 
                      s=8, alpha=0.7, label=f'{QUALITY_NAMES.get(q, f"Q{q}")} ({mask.sum()})')
    
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='GT')
    ax.set_ylabel('Longitude Error (m)')
    ax.set_ylim(-0.5, 0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 3: Height errors
    ax = axes[2]
    
    # Plot confidence bands for each quality
    for q in sorted(df['Q'].unique()):
        mask = df['Q'] == q
        if mask.sum() > 0:
            subset = df[mask]
            ax.fill_between(subset['datetime'], 
                           subset['hgt_error'] - subset['sdu'],
                           subset['hgt_error'] + subset['sdu'],
                           alpha=0.2, color=QUALITY_COLORS.get(q, 'gray'))
    
    # Plot points colored by quality
    for q in sorted(df['Q'].unique()):
        mask = df['Q'] == q
        if mask.sum() > 0:
            subset = df[mask]
            ax.scatter(subset['datetime'], subset['hgt_error'], 
                      c=QUALITY_COLORS.get(q, 'gray'), 
                      s=8, alpha=0.7, label=f'{QUALITY_NAMES.get(q, f"Q{q}")} ({mask.sum()})')
    
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='GT')
    ax.set_ylabel('Height Error (m)')
    ax.set_xlabel('Time (GPST)')
    ax.set_ylim(-0.5, 0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Enhanced position time series saved to {output_file}")
    else:
        plt.show()

def plot_accuracy_statistics(df, output_file=None):
    """Plot accuracy statistics and histograms"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Calculate 2D and 3D position errors
    df['2d_error'] = np.sqrt(df['lat_error']**2 + df['lon_error']**2)
    df['3d_error'] = np.sqrt(df['lat_error']**2 + df['lon_error']**2 + df['hgt_error']**2)
    
    # Plot 1: RMS error over time
    ax = axes[0, 0]
    
    # Calculate rolling RMS (10-minute windows)
    window_size = 600  # 10 minutes at 1Hz
    df['rms_2d'] = df['2d_error'].rolling(window=window_size, min_periods=1).apply(lambda x: np.sqrt(np.mean(x**2)))
    df['rms_3d'] = df['3d_error'].rolling(window=window_size, min_periods=1).apply(lambda x: np.sqrt(np.mean(x**2)))
    
    ax.plot(df['datetime'], df['rms_2d'], 'b-', label='2D RMS', linewidth=1.5)
    ax.plot(df['datetime'], df['rms_3d'], 'r-', label='3D RMS', linewidth=1.5)
    ax.set_ylabel('RMS Error (m)')
    ax.set_title('Rolling RMS Error (10-min window)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(0, 1.0)
    
    # Plot 2: Error histograms
    ax = axes[0, 1]
    
    ax.hist(df['lat_error'], bins=50, alpha=0.7, label='Latitude', color='blue', density=True)
    ax.hist(df['lon_error'], bins=50, alpha=0.7, label='Longitude', color='red', density=True)
    ax.hist(df['hgt_error'], bins=50, alpha=0.7, label='Height', color='green', density=True)
    ax.set_xlabel('Error (m)')
    ax.set_ylabel('Density')
    ax.set_title('Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, 0.5)
    
    # Plot 3: 2D/3D error histogram
    ax = axes[0, 2]
    
    ax.hist(df['2d_error'], bins=50, alpha=0.7, label='2D Error', color='blue', density=True)
    ax.hist(df['3d_error'], bins=50, alpha=0.7, label='3D Error', color='red', density=True)
    ax.set_xlabel('Error (m)')
    ax.set_ylabel('Density')
    ax.set_title('Position Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.0)
    
    # Plot 4: Standard deviation over time
    ax = axes[1, 0]
    
    ax.plot(df['datetime'], df['sdn'], 'b-', label='North σ', linewidth=1)
    ax.plot(df['datetime'], df['sde'], 'r-', label='East σ', linewidth=1)
    ax.plot(df['datetime'], df['sdu'], 'g-', label='Up σ', linewidth=1)
    ax.set_ylabel('Standard Deviation (m)')
    ax.set_xlabel('Time (GPST)')
    ax.set_title('Positioning Standard Deviation')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(0, 1.0)
    
    # Plot 5: Cumulative accuracy
    ax = axes[1, 1]
    
    errors_2d = np.sort(df['2d_error'])
    errors_3d = np.sort(df['3d_error'])
    percentiles = np.linspace(0, 100, len(errors_2d))
    
    ax.plot(percentiles, errors_2d, 'b-', label='2D Error', linewidth=2)
    ax.plot(percentiles, errors_3d, 'r-', label='3D Error', linewidth=2)
    ax.axhline(y=0.1, color='gray', linestyle='--', alpha=0.7, label='10cm')
    ax.axhline(y=0.2, color='gray', linestyle='--', alpha=0.7, label='20cm')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='50cm')
    ax.set_xlabel('Percentile (%)')
    ax.set_ylabel('Error (m)')
    ax.set_title('Cumulative Accuracy')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(0, 1.0)
    
    # Plot 6: Quality factor distribution
    ax = axes[1, 2]
    
    quality_counts = df['Q'].value_counts().sort_index()
    colors = [QUALITY_COLORS.get(q, 'gray') for q in quality_counts.index]
    labels = [f'{QUALITY_NAMES.get(q, f"Q{q}")} ({count})' for q, count in quality_counts.items()]
    
    wedges, texts, autotexts = ax.pie(quality_counts.values, labels=labels, colors=colors, autopct='%1.1f%%')
    ax.set_title('Solution Quality Distribution')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Accuracy statistics saved to {output_file}")
    else:
        plt.show()

def print_enhanced_statistics(df):
    """Print comprehensive statistics"""
    print("\n" + "="*60)
    print("ENHANCED GNSS POSITION STATISTICS")
    print("="*60)
    
    print(f"Total epochs: {len(df)}")
    print(f"Time span: {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")
    print(f"Duration: {df['datetime'].iloc[-1] - df['datetime'].iloc[0]}")
    
    print(f"\nGround Truth Reference:")
    print(f"  Latitude:  {GT_COORDINATES['latitude']:.9f}°")
    print(f"  Longitude: {GT_COORDINATES['longitude']:.9f}°")
    print(f"  Height:    {GT_COORDINATES['height']:.4f} m")
    
    # Calculate 2D and 3D errors
    df['2d_error'] = np.sqrt(df['lat_error']**2 + df['lon_error']**2)
    df['3d_error'] = np.sqrt(df['lat_error']**2 + df['lon_error']**2 + df['hgt_error']**2)
    
    print(f"\nPosition Errors (relative to GT):")
    print(f"  Latitude:  {df['lat_error'].mean():.4f} ± {df['lat_error'].std():.4f} m")
    print(f"  Longitude: {df['lon_error'].mean():.4f} ± {df['lon_error'].std():.4f} m")
    print(f"  Height:    {df['hgt_error'].mean():.4f} ± {df['hgt_error'].std():.4f} m")
    
    print(f"\nRMS Errors:")
    print(f"  2D RMS: {np.sqrt(np.mean(df['2d_error']**2)):.4f} m")
    print(f"  3D RMS: {np.sqrt(np.mean(df['3d_error']**2)):.4f} m")
    
    print(f"\nAccuracy Percentiles:")
    print(f"  2D Error (50%): {np.percentile(df['2d_error'], 50):.4f} m")
    print(f"  2D Error (68%): {np.percentile(df['2d_error'], 68):.4f} m")
    print(f"  2D Error (95%): {np.percentile(df['2d_error'], 95):.4f} m")
    print(f"  3D Error (50%): {np.percentile(df['3d_error'], 50):.4f} m")
    print(f"  3D Error (68%): {np.percentile(df['3d_error'], 68):.4f} m")
    print(f"  3D Error (95%): {np.percentile(df['3d_error'], 95):.4f} m")
    
    print(f"\nPredicted Accuracy (mean σ):")
    print(f"  North σ: {df['sdn'].mean():.4f} m")
    print(f"  East σ:  {df['sde'].mean():.4f} m")
    print(f"  Up σ:    {df['sdu'].mean():.4f} m")
    
    print(f"\nSolution Quality Distribution:")
    quality_counts = df['Q'].value_counts().sort_index()
    total_epochs = len(df)
    for q, count in quality_counts.items():
        percentage = (count / total_epochs) * 100
        print(f"  {QUALITY_NAMES.get(q, f'Q{q}'):8}: {count:4d} epochs ({percentage:5.1f}%)")
    
    print(f"\nSatellite Information:")
    print(f"  Average satellites: {df['ns'].mean():.1f} ± {df['ns'].std():.1f}")
    print(f"  Min satellites: {df['ns'].min():.0f}")
    print(f"  Max satellites: {df['ns'].max():.0f}")
    
    # Convergence analysis
    if len(df) > 1800:  # If more than 30 minutes of data
        first_30min = df[:1800]
        last_30min = df[-1800:]
        
        print(f"\nConvergence Analysis:")
        print(f"  First 30 min 2D RMS: {np.sqrt(np.mean(first_30min['2d_error']**2)):.4f} m")
        print(f"  Last 30 min 2D RMS:  {np.sqrt(np.mean(last_30min['2d_error']**2)):.4f} m")
        print(f"  First 30 min 3D RMS: {np.sqrt(np.mean(first_30min['3d_error']**2)):.4f} m")
        print(f"  Last 30 min 3D RMS:  {np.sqrt(np.mean(last_30min['3d_error']**2)):.4f} m")

def main():
    parser = argparse.ArgumentParser(description='Enhanced rnx2rtkp positioning results plotter')
    parser.add_argument('input_file', help='Input rnx2rtkp position file')
    parser.add_argument('--output', '-o', help='Output directory for plots')
    parser.add_argument('--stats', '-s', action='store_true', help='Show detailed statistics')
    parser.add_argument('--format', '-f', default='png', choices=['png', 'pdf', 'svg'],
                       help='Output format (default: png)')
    
    args = parser.parse_args()
    
    try:
        print(f"Reading {args.input_file}...")
        df = read_rnx2rtkp_file(args.input_file)
        
        if args.stats:
            print_enhanced_statistics(df)
        
        if args.output:
            import os
            os.makedirs(args.output, exist_ok=True)
            
            plot_enhanced_time_series(df, 
                os.path.join(args.output, f'enhanced_position_time_series.{args.format}'))
            plot_accuracy_statistics(df, 
                os.path.join(args.output, f'accuracy_statistics.{args.format}'))
        else:
            plot_enhanced_time_series(df)
            plot_accuracy_statistics(df)
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()