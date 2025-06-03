#!/usr/bin/env python3
"""
Transient Editor - Version 20250602_011500_0_0_1_1
Enhanced with fine-tuning controls for precise transient positioning
- Coarse adjustment: 1/4 of smallest segment (2048 samples)
- Fine adjustment: 1/32 of smallest segment (256 samples)
- Keyboard controls for left/right adjustments
- 250ms real-time monitoring
"""

import os
import sqlite3
import scipy.io
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify, send_from_directory
import requests
import json
import time
import json
from pathlib import Path
from io import BytesIO
import math
# STYLES GALLERY INTEGRATION
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from styles import styles_gallery
from image_utils import universal_saver
# API COMPATIBILITY
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from api_formats import standardize_transient_output, standardize_error_response, parse_worker3_database_input

app = Flask(__name__)

# Configuration
RAW_DATA_DIR = "/Volumes/ArcData/V3_raw_data"
DB_DIR = "/Users/kjensen/Documents/GitHub/data_processor_project/arc_detection_project"
TEMP_PLOTS_DIR = "/Users/kjensen/Documents/GitHub/data_processor_project/arc_detection_project/temp_transient_plots"
MAIN_TOOL_URL = "http://localhost:5030"

# Database paths
V3_DATABASE_PATH = "/Volumes/ArcData/V3_database/arc_detection.db"
TRANSIENT_DB_PATH = os.path.join(DB_DIR, "transient_centers.db")

# Ensure temp plots directory exists
os.makedirs(TEMP_PLOTS_DIR, exist_ok=True)

# Fine-tuning constants (based on segment configuration [524288, 65536, 8192])
SMALLEST_SEGMENT = 8192  # Smallest segment size
COARSE_ADJUSTMENT = SMALLEST_SEGMENT // 4  # 2048 samples (1/4 of smallest)
FINE_ADJUSTMENT = SMALLEST_SEGMENT // 32   # 256 samples (1/32 of smallest)
SAMPLING_RATE = 5.0e6  # 5 MSPS

def init_transient_database():
    """Initialize database for storing fine-tuned transient centers"""
    conn = sqlite3.connect(TRANSIENT_DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transient_centers (
            experiment_path TEXT PRIMARY KEY,
            center_time REAL,
            center_sample INTEGER,
            initial_chunk_size INTEGER,
            final_chunk_size INTEGER,
            detection_confidence REAL,
            timestamp REAL
        )
    ''')
    conn.commit()
    conn.close()

# Initialize database on startup
init_transient_database()

# Global variable for current file info (like the working viewer)
current_file_info = {'file_id': None, 'filename': None}

def remove_outliers_from_chunk(signal, width=5, threshold=2.5):
    """Remove outliers from signal chunk"""
    if len(signal) < width:
        return signal
    
    cleaned_signal = signal.copy()
    signal_std = np.std(signal)
    signal_mean = np.mean(signal)
    
    if signal_std == 0:
        return signal
    
    for i in range(len(signal) - width + 1):
        window = signal[i:i+width]
        deviations = np.abs(window - signal_mean)
        if np.all(deviations > threshold * signal_std):
            cleaned_signal[i:i+width] = np.nan
    
    return cleaned_signal

def extract_features_with_outlier_removal(voltage_data, current_data, window_size=4096, step_size=4096):
    """Extract features with outlier removal for transient detection"""
    features = []
    window_starts = []
    
    for start in range(0, len(voltage_data) - window_size + 1, step_size):
        # Extract windows
        voltage_chunk = voltage_data[start:start+window_size]
        current_chunk = current_data[start:start+window_size]
        
        # Apply outlier removal
        voltage_clean = remove_outliers_from_chunk(voltage_chunk, 5, 2.5)
        current_clean = remove_outliers_from_chunk(current_chunk, 5, 2.5)
        
        # Remove NaN values for feature calculation
        voltage_valid = voltage_clean[~np.isnan(voltage_clean)]
        current_valid = current_clean[~np.isnan(current_clean)]
        
        if len(voltage_valid) == 0 or len(current_valid) == 0:
            features.append([0, 0, 0, 0, 0, 0])
        else:
            # Voltage features
            v_avg = np.mean(voltage_valid)
            v_std = np.std(voltage_valid)
            v_min = np.min(voltage_valid)
            
            # Current features (inverted)
            c_max = np.max(current_valid)
            c_avg = np.mean(current_valid)
            c_std = np.std(current_valid)
            c_min = np.min(current_valid)
            
            c_feature_1 = c_max - c_avg  # Inverted
            c_feature_3 = c_max - c_min  # Inverted
            
            features.append([v_avg, v_std, v_min, c_feature_1, c_std, c_feature_3])
        
        window_starts.append(start)
    
    return np.array(features), window_starts

def detect_transients_simple(features):
    """Simple transient detection based on feature thresholds"""
    if len(features) == 0:
        return []
    
    # Calculate product feature
    feature_products = np.prod(features, axis=1)
    
    # Simple threshold-based detection
    threshold = np.mean(feature_products) + 2 * np.std(feature_products)
    
    transient_windows = []
    for i, product in enumerate(feature_products):
        if product > threshold:
            transient_windows.append(i)
    
    return transient_windows

def progressive_transient_detection(voltage_data, current_data, initial_chunk=8192, final_chunk=128):
    """Progressive transient detection with specific progression: 8192, 1024, 128"""
    results = {}
    
    # Fixed progression as requested
    chunk_sizes = [8192, 1024, 128]
    
    # Initial detection with largest chunk size
    features, window_starts = extract_features_with_outlier_removal(
        voltage_data, current_data, chunk_sizes[0], chunk_sizes[0]
    )
    transient_windows = detect_transients_simple(features)
    
    if not transient_windows:
        return None, chunk_sizes
    
    # Get the first detected transient window for refinement
    first_transient_window = transient_windows[0]
    initial_start = window_starts[first_transient_window]
    initial_end = initial_start + chunk_sizes[0]
    
    # Define zoom region around initial detection (expand by 50% on each side)
    zoom_margin = chunk_sizes[0] // 2
    zoom_start = max(0, initial_start - zoom_margin)
    zoom_end = min(len(voltage_data), initial_end + zoom_margin)
    
    # Extract zoomed data
    zoom_voltage = voltage_data[zoom_start:zoom_end]
    zoom_current = current_data[zoom_start:zoom_end]
    
    results['zoom_region'] = (zoom_start, zoom_end)
    results['zoom_data'] = (zoom_voltage, zoom_current)
    results['progressions'] = []
    
    # Progressive refinement within zoom region
    for chunk_size in chunk_sizes:
        if len(zoom_voltage) < chunk_size:
            continue
            
        # COORDINATE FIX: Use non-overlapping windows to simplify coordinate tracking
        step_size = chunk_size  # Non-overlapping windows for clearer coordinates
        
        # Extract features for this chunk size
        features, window_starts = extract_features_with_outlier_removal(
            zoom_voltage, zoom_current, chunk_size, step_size
        )
        
        # Detect transients
        transient_windows = detect_transients_simple(features)
        
        if transient_windows:
            # Get the strongest transient (highest feature product)
            feature_products = np.prod(features, axis=1)
            best_window_idx = transient_windows[np.argmax([feature_products[i] for i in transient_windows])]
            
            # COORDINATE FIX: Calculate positions in downsampled coordinate system
            local_start = window_starts[best_window_idx]
            local_end = local_start + chunk_size
            # absolute positions are within the zoom region (downsampled coordinates)
            absolute_start = zoom_start + local_start
            absolute_end = zoom_start + local_end
            absolute_center = (absolute_start + absolute_end) / 2
            
            results['progressions'].append({
                'chunk_size': chunk_size,
                'local_start': local_start,
                'local_end': local_end,
                'absolute_start': absolute_start,
                'absolute_end': absolute_end,
                'absolute_center': absolute_center,
                'confidence': feature_products[best_window_idx]
            })
    
    return results, chunk_sizes

def save_transient_center(experiment_path, center_time, center_sample, initial_chunk, final_chunk, confidence):
    """Save fine-tuned transient center to database"""
    conn = sqlite3.connect(TRANSIENT_DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO transient_centers 
        (experiment_path, center_time, center_sample, initial_chunk_size, final_chunk_size, detection_confidence, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (experiment_path, center_time, center_sample, initial_chunk, final_chunk, confidence, time.time()))
    conn.commit()
    conn.close()

def get_saved_transient_center(experiment_path):
    """Retrieve saved transient center from database"""
    conn = sqlite3.connect(TRANSIENT_DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM transient_centers WHERE experiment_path = ?', (experiment_path,))
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return {
            'experiment_path': result[0],
            'center_time': result[1],
            'center_sample': result[2],
            'initial_chunk_size': result[3],
            'final_chunk_size': result[4],
            'detection_confidence': result[5],
            'timestamp': result[6]
        }
    return None

def get_database_transients(experiment_path):
    """Get transient indices from V3 database - these override ML predictions"""
    if not experiment_path or '/' not in experiment_path:
        return None
    
    # Extract filename from experiment path
    exp_name = experiment_path.split('/')[-1]
    
    try:
        conn = sqlite3.connect(V3_DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT transient1_index, transient2_index, transient3_index 
            FROM files WHERE original_filename = ?
        ''', (exp_name,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            transients = []
            # Convert sample indices to time (5 MSPS sampling rate)
            sampling_rate = 5.0e6
            for i, sample_index in enumerate(result):
                if sample_index is not None:
                    time_seconds = sample_index / sampling_rate
                    transients.append({
                        'index': i + 1,
                        'sample': sample_index,
                        'time': time_seconds,
                        'source': 'database'
                    })
            return transients if transients else None
    except Exception as e:
        print(f"Error getting database transients: {e}")
        return None
    
    return None

def save_database_transient(experiment_path, transient_sample, transient_index=1):
    """Save transient to V3 database"""
    if not experiment_path or '/' not in experiment_path:
        return False
    
    exp_name = experiment_path.split('/')[-1]
    
    try:
        conn = sqlite3.connect(V3_DATABASE_PATH)
        cursor = conn.cursor()
        
        # Map transient index to column
        transient_columns = {1: 'transient1_index', 2: 'transient2_index', 3: 'transient3_index'}
        column = transient_columns.get(transient_index, 'transient1_index')
        
        cursor.execute(f'''
            UPDATE files SET {column} = ? WHERE original_filename = ?
        ''', (int(transient_sample), exp_name))
        
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return success
    except Exception as e:
        print(f"Error saving database transient: {e}")
        return False

def create_output_files_subplot_20250526_125500_001_001_001_003(ax, time_data, final_transient_center, final_transient_sample, exp_name):
    """Create 5th subplot - Rev 20250526_125500_001_001_001_003 - FIXED data loading"""
    total_time = time_data[-1] if len(time_data) > 0 else 1.0
    
    # Use augmented data file length (262144 samples)
    augmented_file_length = 262144
    effective_sampling_rate = 5.0e6  # 5 MHz raw sampling rate
    file_duration = augmented_file_length / effective_sampling_rate  # Duration of each file in seconds
    
    # GENERAL APPROACH: Calculate maximum possible segments
    max_segments = int(total_time / file_duration)
    
    print(f"DEBUG SEGMENT FIX: total_time={total_time:.3f}s, file_duration={file_duration:.6f}s")
    print(f"DEBUG SEGMENT FIX: max_segments={max_segments} (should be ~9 for 0.5s data)")
    print(f"DEBUG SEGMENT FIX: final_transient_center={final_transient_center}")
    
    file_times = []
    
    if final_transient_center is not None:
        # SEGMENT FIX: Generate segments based on max_segments constraint
        if final_transient_center <= file_duration:
            print(f"DEBUG SEGMENT FIX: Transient in first segment - C is FIRST segment starting at 0")
            # C segment: 0.0 to file_duration (samples 0-262143)
            file_times.append((0.0, file_duration, 'center', 0))
            
            # Generate right files (R001, R002, etc.) up to max_segments
            for i in range(1, max_segments):
                start_time = i * file_duration
                end_time = (i + 1) * file_duration
                if start_time < total_time:  # Only add if within data range
                    file_times.append((start_time, end_time, 'right', i))
        else:
            # SEGMENT FIX: Determine which segment contains the transient
            transient_segment = int(final_transient_center / file_duration)
            
            # Generate all segments sequentially, marking the transient one as 'C'
            for i in range(max_segments):
                start_time = i * file_duration
                end_time = min((i + 1) * file_duration, total_time)
                
                if i < transient_segment:
                    file_times.append((start_time, end_time, 'left', i + 1))
                elif i == transient_segment:
                    file_times.append((start_time, end_time, 'center', 0))
                else:
                    file_times.append((start_time, end_time, 'right', i - transient_segment))
    else:
        # Fallback: sequential files up to max_segments
        for i in range(max_segments):
            start_time = i * file_duration
            end_time = min((i + 1) * file_duration, total_time)
            file_times.append((start_time, end_time, 'sequential', i))
    
    print(f"DEBUG REV 20250526_125000_002_001_001_001: Generated {len(file_times)} files")
    for i, (start, end, ftype, num) in enumerate(file_times):
        print(f"  File {i}: {start:.6f}-{end:.6f}s, type={ftype}, num={num}")
    
    # Create file rectangles
    rect_height = 1.0
    for file_start_time, file_end_time, file_type, file_number in file_times:
        # Determine file color and label
        if file_type == 'left':
            file_label = f'L{file_number:03d}'
            file_color = 'green'
        elif file_type == 'center':
            file_label = 'C'
            file_color = 'red'
        elif file_type == 'right':
            file_label = f'R{file_number:03d}'
            file_color = 'blue'
        else:
            file_label = f'S{file_number:03d}'
            file_color = 'orange'
        
        # Draw rectangle
        rect_width = file_end_time - file_start_time
        rect = plt.Rectangle((file_start_time, 0), rect_width, rect_height, 
                           facecolor=file_color, edgecolor='black', linewidth=1, alpha=0.7)
        ax.add_patch(rect)
        
        # Add label
        center_time = (file_start_time + file_end_time) / 2
        text_color = 'white' if file_color in ['red', 'blue'] else 'black'
        ax.text(center_time, rect_height/2, file_label, ha='center', va='center', 
               fontsize=10, fontweight='bold', color=text_color)
    
    # Configure subplot
    ax.set_xlim(0, total_time)
    ax.set_ylim(0, rect_height)
    ax.set_ylabel('Augmented Data Files')
    ax.set_title(f'{exp_name} - Data Segments Rev 20250526_125000_002_001_001_001\nC=samples 0-{augmented_file_length-1} when transient early')
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_yticks([])
    
    return ax

def get_current_experiment_from_main_tool():
    """Get current experiment being viewed in the main tool"""
    try:
        response = requests.get(f"{MAIN_TOOL_URL}/stats", timeout=2)
        if response.status_code == 200:
            sync_file = "/Volumes/ArcData/V3_database/current_experiment.sync"
            if os.path.exists(sync_file):
                with open(sync_file, 'r') as f:
                    return f.read().strip()
    except:
        pass
    
    # Fallback
    filtered_file = os.path.join(DB_DIR, "full_dataset_filtered.txt")
    if not os.path.exists(filtered_file):
        return None
    
    with open(filtered_file, 'r') as f:
        experiments = [line.strip() for line in f.readlines() 
                      if line.strip() and not line.strip().startswith('#')]
    
    return experiments[0] if experiments else None

def create_transient_prediction_plot_20250526_125000_002_001_001_001(experiment_path):
    """Create plot - Rev 20250526_125000_002_001_001_001 - ENTIRE data file with proper colors"""
    print(f"PLOT_FUNCTION: === PLOT FUNCTION CALLED ===")
    print(f"PLOT_FUNCTION: experiment_path = {experiment_path}")
    
    if not experiment_path or '/' not in experiment_path:
        print(f"PLOT_FUNCTION: ERROR - Invalid experiment_path")
        return None
    
    exp_type, exp_name = experiment_path.split('/')
    exp_full_path = os.path.join(RAW_DATA_DIR, exp_type, exp_name)
    
    if not os.path.exists(exp_full_path):
        return None
    
    # Find .mat files
    mat_files = [f for f in os.listdir(exp_full_path) if f.endswith('.mat')]
    if len(mat_files) == 0:
        return None
    
    # Load voltage and current data (corrected channels)
    voltage_data = None
    current_data = None
    
    for mat_file in mat_files:
        file_path = os.path.join(exp_full_path, mat_file)
        try:
            mat_data = scipy.io.loadmat(file_path)
            if 'data' in mat_data:
                data = mat_data['data'].flatten()
            elif 'y' in mat_data:
                data = mat_data['y'].flatten()
            else:
                continue
            
            # CRITICAL FIX: Limit to 2.5M samples for 0.5s duration at 5MSPS
            # This prevents loading 5M samples which gives wrong 1.0s duration
            MAX_SAMPLES = int(2.5e6)  # 2.5M samples = 0.5s at 5MSPS
            if len(data) > MAX_SAMPLES:
                data = data[:MAX_SAMPLES]
                print(f"DEBUG REV 20250526_125500_001_001_001_003: Truncated {len(data)} samples to {MAX_SAMPLES} for 0.5s duration")
            
            # Use truncated data file
            if 'ch1' in mat_file:  # Load voltage (ch1)
                if voltage_data is None:
                    voltage_data = data
            elif 'ch4' in mat_file:  # Load current (ch4)
                if current_data is None:
                    current_data = data
        except Exception as e:
            continue
    
    if voltage_data is None or current_data is None:
        return None
    
    # Make same length
    min_len = min(len(voltage_data), len(current_data))
    voltage_data = voltage_data[:min_len]
    current_data = current_data[:min_len]
    
    print(f"DEBUG REV 20250526_125500_001_001_001_003: Loaded {min_len} samples (should be ~2.5M)")
    
    # Define effective sampling rate (raw data is 5MSPS)
    effective_sampling_rate = 5.0e6  # 5 MHz
    
    # Create time array for ENTIRE data file
    time = np.linspace(0, min_len / effective_sampling_rate, min_len)
    
    print(f"DEBUG REV 20250526_125500_001_001_001_003: Total time = {time[-1]:.3f}s (should be ~0.5s)")
    
    # For detection, use downsampled data
    downsample_factor = 50
    voltage_detection = voltage_data[::downsample_factor]
    current_detection = current_data[::downsample_factor]
    
    # Extract features and detect transients on downsampled data
    features, window_starts = extract_features_with_outlier_removal(voltage_detection, current_detection)
    transient_windows = detect_transients_simple(features)
    
    # Run fine detection for precise center
    fine_results = None
    final_transient_center = None
    final_transient_sample = None
    transient_source = "none"
    
    # PRIORITY 1: Check for database transients (override everything)
    database_transients = get_database_transients(experiment_path)
    if database_transients:
        # Use the first database transient
        db_transient = database_transients[0]
        final_transient_center = db_transient['time']
        final_transient_sample = db_transient['sample']
        transient_source = "database"
        print(f"Using DATABASE transient: center at {final_transient_center:.6f}s (sample {final_transient_sample})")
    else:
        # PRIORITY 2: Check for cached ML predictions
        cached_result = get_saved_transient_center(experiment_path)
        if cached_result and cached_result['initial_chunk_size'] == 8192 and cached_result['final_chunk_size'] == 128:
            final_transient_center = cached_result['center_time']
            final_transient_sample = cached_result['center_sample']
            transient_source = "cached"
            print(f"Using cached ML result: center at {final_transient_center:.6f}s")
        else:
            # PRIORITY 3: Run new ML detection
            fine_results, fine_chunk_sizes = progressive_transient_detection(voltage_detection, current_detection)
            if fine_results and fine_results['progressions']:
                final_result = fine_results['progressions'][-1]
                # COORDINATE FIX: absolute_center is already in downsampled coordinates, convert correctly
                downsampled_center = final_result['absolute_center']
                # Convert downsampled coordinate to full resolution
                final_transient_sample = int(downsampled_center * downsample_factor)
                final_transient_center = final_transient_sample / effective_sampling_rate
                save_transient_center(experiment_path, final_transient_center, final_transient_sample, 8192, 128, final_result['confidence'])
                transient_source = "ml_detection"
                print(f"New ML detection: downsampled_center={downsampled_center:.1f}, full_sample={final_transient_sample}, time={final_transient_center:.6f}s")
    
    # MAJOR FIX: Create color arrays for ENTIRE data file
    colors_v = ['green'] * min_len
    colors_c = ['green'] * min_len
    
    if final_transient_center is not None:
        # COORDINATE FIX: Define transient region correctly based on detection chunk size
        # Detection chunk size was 8192 samples on downsampled data (downsample_factor=50)
        # So in full resolution, this represents 8192 * 50 = 409600 samples
        detection_chunk_full_res = 8192 * downsample_factor
        half_chunk = detection_chunk_full_res // 2
        
        transient_start_sample = max(0, final_transient_sample - half_chunk)
        transient_end_sample = min(min_len, final_transient_sample + half_chunk)
        
        print(f"DEBUG REV 20250526_125000_002_001_001_001: Transient region samples {transient_start_sample}-{transient_end_sample}")
        
        # Color the arrays: green=before, red=during, blue=after
        for i in range(min_len):
            if i < transient_start_sample:
                colors_v[i] = 'green'
                colors_c[i] = 'green'
            elif transient_start_sample <= i < transient_end_sample:
                colors_v[i] = 'red'
                colors_c[i] = 'red'
            else:
                colors_v[i] = 'blue'
                colors_c[i] = 'blue'
    
    # Downsample for plotting only (but maintain color alignment)
    plot_downsample = 100  # Plot every 100th point for performance
    time_plot = time[::plot_downsample]
    voltage_plot = voltage_data[::plot_downsample]
    current_plot = current_data[::plot_downsample]
    colors_v_plot = colors_v[::plot_downsample]
    colors_c_plot = colors_c[::plot_downsample]
    
    # STYLES GALLERY INTEGRATION: Apply style and get colors
    current_style = 'default'  # Could be made configurable via API
    styles_gallery.apply_style(current_style)
    colors = styles_gallery.get_colors(current_style)
    
    # Create figure with style-based sizing
    style_config = styles_gallery.get_style_config(current_style)
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=style_config['figure_size'])
    
    # MAJOR FIX: Plot ENTIRE voltage signal with proper color segments
    plot_colored_segments(ax1, time_plot, voltage_plot, colors_v_plot)
    ax1.set_ylabel('Load Voltage (V)')
    ax1.set_title(f'{exp_name} - ENTIRE Load Voltage Rev 20250526_125000_002_001_001_001\n(Green=Before, Red=During, Blue=After)')
    ax1.grid(True, alpha=0.3)
    
    # Add transient center line
    if final_transient_center is not None:
        ax1.axvline(final_transient_center, color='black', linestyle='--', linewidth=2, alpha=0.8, label=f'Center: {final_transient_sample}')
        ax1.legend(loc='upper right')
    
    # MAJOR FIX: Plot ENTIRE current signal with proper color segments  
    plot_colored_segments(ax2, time_plot, current_plot, colors_c_plot)
    ax2.set_ylabel('Load Current (A)')
    ax2.set_title(f'{exp_name} - ENTIRE Load Current Rev 20250526_125000_002_001_001_001\n(Green=Before, Red=During, Blue=After)')
    ax2.grid(True, alpha=0.3)
    
    if final_transient_center is not None:
        ax2.axvline(final_transient_center, color='black', linestyle='--', linewidth=2, alpha=0.8)
    
    # Enhanced Zoom plots (ax3, ax4) - focusing on transient region with color coding
    if final_transient_center is not None:
        # COORDINATE FIX: Create zoom region centered on transient with higher resolution
        # Ensure final_transient_sample is correctly in full resolution coordinates
        transient_sample_full_res = int(final_transient_sample)
        zoom_samples_before = 50000  # ~10ms before at full resolution (50000/5e6 = 0.01s)
        zoom_samples_after = 50000   # ~10ms after at full resolution
        
        zoom_start_full = max(0, transient_sample_full_res - zoom_samples_before)
        zoom_end_full = min(min_len, transient_sample_full_res + zoom_samples_after)
        
        # Extract full-resolution zoom data
        zoom_voltage_full = voltage_data[zoom_start_full:zoom_end_full]
        zoom_current_full = current_data[zoom_start_full:zoom_end_full]
        zoom_time_full = time[zoom_start_full:zoom_end_full]
        
        # Create color arrays for zoom region
        zoom_colors_v = ['green'] * len(zoom_voltage_full)
        zoom_colors_c = ['green'] * len(zoom_current_full)
        
        # COORDINATE FIX: Color the transient region within zoom using correct coordinate mapping
        # transient_start_sample and transient_end_sample are in full resolution coordinates
        # zoom_start_full is also in full resolution coordinates
        transient_start_in_zoom = max(0, transient_start_sample - zoom_start_full)
        transient_end_in_zoom = min(len(zoom_voltage_full), transient_end_sample - zoom_start_full)
        
        for i in range(len(zoom_voltage_full)):
            if i < transient_start_in_zoom:
                zoom_colors_v[i] = 'green'  # Before transient
                zoom_colors_c[i] = 'green'
            elif transient_start_in_zoom <= i < transient_end_in_zoom:
                zoom_colors_v[i] = 'red'    # During transient
                zoom_colors_c[i] = 'red'
            else:
                zoom_colors_v[i] = 'blue'   # After transient
                zoom_colors_c[i] = 'blue'
        
        # Downsample for plotting (every 10th point for performance)
        zoom_downsample = 10
        zoom_time_plot = zoom_time_full[::zoom_downsample]
        zoom_voltage_plot = zoom_voltage_full[::zoom_downsample]
        zoom_current_plot = zoom_current_full[::zoom_downsample]
        zoom_colors_v_plot = zoom_colors_v[::zoom_downsample]
        zoom_colors_c_plot = zoom_colors_c[::zoom_downsample]
        
        # Plot zoomed voltage with color segments
        plot_colored_segments(ax3, zoom_time_plot, zoom_voltage_plot, zoom_colors_v_plot)
        ax3.set_ylabel('Zoomed Voltage (V)')
        ax3.set_title(f'{exp_name} - ZOOMED Transient Region (±10ms)\\n(Green=Before, Red=During, Blue=After)')
        ax3.grid(True, alpha=0.3)
        ax3.axvline(final_transient_center, color='black', linestyle='--', linewidth=2, alpha=0.8, label=f'Center: {final_transient_sample}')
        ax3.legend(loc='upper right', fontsize=8)
        
        # Plot zoomed current with color segments
        plot_colored_segments(ax4, zoom_time_plot, zoom_current_plot, zoom_colors_c_plot)
        ax4.set_ylabel('Zoomed Current (A)')
        ax4.set_title(f'{exp_name} - ZOOMED Transient Region (±10ms)\\n(Green=Before, Red=During, Blue=After)')
        ax4.grid(True, alpha=0.3)
        ax4.axvline(final_transient_center, color='black', linestyle='--', linewidth=2, alpha=0.8)
        
        print(f"DEBUG ZOOM: Created zoom plots from {zoom_time_full[0]:.6f}s to {zoom_time_full[-1]:.6f}s")
        print(f"DEBUG ZOOM: Transient at {final_transient_center:.6f}s, zoom duration: {(zoom_time_full[-1] - zoom_time_full[0]):.6f}s")
        
    else:
        # No transient detected - show message
        ax3.text(0.5, 0.5, 'No transient detected\\nCannot create zoom view', ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title(f'{exp_name} - No Transient Detected')
        
        ax4.text(0.5, 0.5, 'No transient detected\\nCannot create zoom view', ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title(f'{exp_name} - No Transient Detected')
    
    # MAJOR FIX: Create 5th subplot with corrected C segment logic
    create_output_files_subplot_20250526_125500_001_001_001_003(ax5, time, final_transient_center, final_transient_sample, exp_name)
    
    plt.tight_layout()
    
    # UNIVERSAL IMAGE SAVE: Use universal saver with metadata
    transient_data = {
        'center_time': final_transient_center,
        'center_sample': final_transient_sample,
        'source': transient_source
    }
    
    save_result = universal_saver.save_transient_plot(
        filename=f"{exp_name}_transient_prediction_coordinate_fixed",
        experiment_path=experiment_path,
        transient_data=transient_data,
        style=current_style,
        formats=['png']  # Can be extended to ['png', 'pdf'] for publications
    )
    
    # For backward compatibility, also save to temp plots directory
    plot_filename = f"{exp_name}_transient_prediction_20250526_125000_002_001_001_001.png"
    plot_path = os.path.join(TEMP_PLOTS_DIR, plot_filename)
    print(f"PLOT_FUNCTION: Saving plot to: {plot_path}")
    plt.savefig(plot_path, dpi=style_config['dpi'], bbox_inches='tight')
    plt.close()
    
    print(f"UNIVERSAL SAVE: Primary file: {save_result.get('primary_file')}")
    print(f"UNIVERSAL SAVE: Metadata: {save_result.get('metadata', {}).get('timestamp')}")
    
    print(f"PLOT_FUNCTION: === PLOT FUNCTION COMPLETED ===")
    print(f"PLOT_FUNCTION: Returning plot_filename: {plot_filename}")
    return plot_filename

def plot_colored_segments(ax, time_data, signal_data, colors):
    """Plot signal as colored line segments"""
    current_color = colors[0]
    start_idx = 0
    
    for i in range(1, len(colors)):
        if colors[i] != current_color:
            # Plot the segment with current color
            ax.plot(time_data[start_idx:i+1], signal_data[start_idx:i+1], 
                   c=current_color, linewidth=1.5, alpha=0.8)
            start_idx = i
            current_color = colors[i]
    
    # Plot the final segment
    ax.plot(time_data[start_idx:], signal_data[start_idx:], 
           c=current_color, linewidth=1.5, alpha=0.8)

@app.route('/')
def index():
    """Main transient editor (like the working viewer)"""
    global current_file_info
    
    plot_filename = None
    experiment_name = None
    current_experiment = None
    
    # Check if we have synced file info first (PRIORITY 1)
    if current_file_info.get('file_id'):
        file_id = current_file_info['file_id']
        filename = current_file_info.get('filename', f'file_{file_id:08d}')
        experiment_name = f"File ID: {file_id} ({filename})"
        current_experiment = f"file_id_{file_id}"
        print(f"INDEX: Using synced file info: {experiment_name}")
        
        # Try to get the correct directory from the file's label in V3 database
        try:
            conn = sqlite3.connect(V3_DATABASE_PATH)
            cursor = conn.cursor()
            cursor.execute('SELECT selected_label FROM files WHERE file_id = ?', (file_id,))
            result = cursor.fetchone()
            conn.close()
            
            if result:
                label = result[0]
                # Use same mapping as enhanced_data_cleaning_tool
                label_to_dir = {
                    'arc': 'arc_matrix_experiment',
                    'parallel_motor_arc': 'arc_matrix_experiment_with_parallel_motor', 
                    'negative_transient': 'transient_negative_test',
                    'steady_state': 'transient_negative_test',
                    'other': 'transient_negative_test'
                }
                directory = label_to_dir.get(label, 'arc_matrix_experiment')
                experiment_path = f"{directory}/{filename}"
                print(f"INDEX: Mapped file_id {file_id} to {experiment_path}")
                
                try:
                    plot_filename = create_transient_prediction_plot_20250526_125000_002_001_001_001(experiment_path)
                    if plot_filename:
                        print(f"INDEX: Plot created successfully from file_id: {plot_filename}")
                    else:
                        print(f"INDEX: Plot generation returned None for file_id {file_id}")
                        experiment_name = f"File ID: {file_id} - Plot generation failed"
                except Exception as plot_error:
                    print(f"INDEX: Plot generation error for file_id {file_id}: {plot_error}")
                    experiment_name = f"File ID: {file_id} - Plot error: {str(plot_error)}"
                    plot_filename = None
            else:
                print(f"INDEX: No database record found for file_id {file_id}")
                plot_filename = None
                experiment_name = f"File ID {file_id} not found in database"
                
        except Exception as e:
            print(f"INDEX: Database error for file_id {file_id}: {e}")
            experiment_name = f"Error: {str(e)}"
            plot_filename = None
    else:
        # Fallback to original experiment logic (PRIORITY 2)
        print(f"INDEX: No synced file info, trying fallback experiment lookup")
        current_experiment = get_current_experiment_from_main_tool()
        
        if current_experiment:
            experiment_name = current_experiment.split('/')[-1] if '/' in current_experiment else current_experiment
            print(f"INDEX: Using experiment path fallback: {experiment_name}")
            try:
                plot_filename = create_transient_prediction_plot_20250526_125000_002_001_001_001(current_experiment)
                if plot_filename:
                    print(f"INDEX: Plot created successfully from experiment path: {plot_filename}")
                else:
                    print(f"INDEX: Plot generation returned None for experiment: {current_experiment}")
                    experiment_name = f"Plot generation failed for: {experiment_name}"
            except Exception as e:
                print(f"INDEX: Error creating plot from experiment path: {e}")
                experiment_name = f"Error: {str(e)}"
                plot_filename = None
        else:
            print(f"INDEX: No fallback experiment found")
            experiment_name = "No experiment available - waiting for sync"
    
    return render_template('transient_editor_20250602_011500_0_0_1_1.html', 
                         experiment_path=current_experiment,
                         experiment_name=experiment_name,
                         plot_filename=plot_filename)

@app.route('/plot/<filename>')
def serve_plot(filename):
    """Serve transient prediction plot images"""
    return send_from_directory(TEMP_PLOTS_DIR, filename)

@app.route('/sync', methods=['POST'])
def sync():
    """Sync with main data review tool - STANDARDIZED API FORMAT"""
    global current_file_info
    
    try:
        data = request.json
        
        # WORKER 3 COMPATIBILITY: Parse standardized database input
        parsed_data = parse_worker3_database_input(data)
        
        # Update global state
        current_file_info['file_id'] = parsed_data['file_id']
        current_file_info['filename'] = parsed_data['filename']
        current_file_info['experiment_path'] = parsed_data['experiment_path']
        
        print(f"SYNC: Received standardized input - file_id: {parsed_data['file_id']}, filename: {parsed_data['filename']}")
        
        # STANDARDIZED RESPONSE FORMAT
        return jsonify({
            'status': 'success',
            'service': 'worker4_transient_editor',
            'synced_file': current_file_info,
            'timestamp': time.time(),
            'api_version': '1.0'
        })
        
    except Exception as e:
        print(f"SYNC ERROR: {e}")
        return jsonify(standardize_error_response(str(e), 'worker4_transient_editor'))

@app.route('/refresh')
def refresh_current():
    """Refresh current experiment view - MUST use same logic as index route"""
    global current_file_info
    
    plot_filename = None
    experiment_name = None
    current_experiment = None
    
    print(f"DEBUG REFRESH: === REFRESH ENDPOINT CALLED ===")
    print(f"DEBUG REFRESH: Checking global current_file_info: {current_file_info}")
    print(f"DEBUG REFRESH: About to start plot generation process...")
    
    # Use EXACT SAME LOGIC as index route
    # Check if we have synced file info first (PRIORITY 1)
    if current_file_info.get('file_id'):
        file_id = current_file_info['file_id']
        filename = current_file_info.get('filename', f'file_{file_id:08d}')
        experiment_name = f"File ID: {file_id} ({filename})"
        current_experiment = f"file_id_{file_id}"
        print(f"REFRESH: Using synced file info: {experiment_name}")
        
        # Try to get the correct directory from the file's label in V3 database
        try:
            conn = sqlite3.connect(V3_DATABASE_PATH)
            cursor = conn.cursor()
            cursor.execute('SELECT selected_label FROM files WHERE file_id = ?', (file_id,))
            result = cursor.fetchone()
            conn.close()
            
            if result:
                label = result[0]
                # Use same mapping as enhanced_data_cleaning_tool
                label_to_dir = {
                    'arc': 'arc_matrix_experiment',
                    'parallel_motor_arc': 'arc_matrix_experiment_with_parallel_motor', 
                    'negative_transient': 'transient_negative_test',
                    'steady_state': 'transient_negative_test',
                    'other': 'transient_negative_test'
                }
                directory = label_to_dir.get(label, 'arc_matrix_experiment')
                experiment_path = f"{directory}/{filename}"
                print(f"REFRESH: Mapped file_id {file_id} to {experiment_path}")
                
                print(f"REFRESH: Calling plot function with experiment_path: {experiment_path}")
                plot_filename = create_transient_prediction_plot_20250526_125000_002_001_001_001(experiment_path)
                print(f"REFRESH: Plot function returned: {plot_filename}")
                if plot_filename:
                    print(f"REFRESH: SUCCESS - Plot created from file_id: {plot_filename}")
                else:
                    print(f"REFRESH: ERROR - Plot function returned None")
            else:
                print(f"REFRESH: No database record found for file_id {file_id}")
                plot_filename = None
                experiment_name = f"File ID {file_id} not found in database"
                
        except Exception as e:
            print(f"REFRESH: Error creating plot from file_id: {e}")
            experiment_name = f"Error: {str(e)}"
            plot_filename = None
    else:
        # Fallback to original experiment logic (PRIORITY 2)
        current_experiment = get_current_experiment_from_main_tool()
        print(f"REFRESH: Using fallback experiment path: {current_experiment}")
        
        if current_experiment:
            experiment_name = current_experiment.split('/')[-1] if '/' in current_experiment else current_experiment
            print(f"REFRESH: Creating plot for experiment path: {experiment_name}")
            try:
                plot_filename = create_transient_prediction_plot_20250526_125000_002_001_001_001(current_experiment)
                print(f"REFRESH: Plot created from experiment path: {plot_filename}")
            except Exception as e:
                print(f"REFRESH: Error creating plot from experiment path: {e}")
                experiment_name = f"Error: {str(e)}"
                plot_filename = None
    
    if plot_filename:
        return jsonify({
            'success': True,
            'experiment_path': current_experiment,
            'experiment_name': experiment_name,
            'plot_filename': plot_filename
        })
    else:
        print("REFRESH: No plot generated")
        return jsonify({'success': False, 'error': 'No plot generated'})

# Stub routes for compatibility
@app.route('/get_transient_data', methods=['POST'])
def get_transient_data():
    return jsonify({'success': True, 'transients': []})

@app.route('/update_transient_plot', methods=['POST'])
def update_transient_plot():
    return jsonify({'success': False, 'error': 'Not implemented in this revision'})

@app.route('/fine_detection', methods=['POST'])
def fine_detection():
    data = request.json
    experiment_path = data.get('experiment_path')
    if not experiment_path:
        return jsonify({'success': False, 'error': 'No experiment path provided'})
    
    plot_filename = create_transient_prediction_plot_20250526_125000_002_001_001_001(experiment_path)
    return jsonify({'success': True, 'plot_filename': plot_filename})

@app.route('/get_cached_center', methods=['POST'])
def get_cached_center():
    data = request.json
    experiment_path = data.get('experiment_path')
    if not experiment_path:
        return jsonify({'success': False, 'error': 'No experiment path provided'})
    
    saved_result = get_saved_transient_center(experiment_path)
    if saved_result:
        return jsonify({'success': True, 'cached': True, 'result': saved_result})
    else:
        return jsonify({'success': True, 'cached': False, 'result': None})

@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    """Clear cached transient center for experiment (Rev 20250526_125500_001_001_001_002)"""
    data = request.json
    experiment_path = data.get('experiment_path')
    if not experiment_path:
        return jsonify({'success': False, 'error': 'No experiment path provided'})
    
    try:
        conn = sqlite3.connect(TRANSIENT_DB_PATH)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM transient_centers WHERE experiment_path = ?', (experiment_path,))
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True, 
            'message': f'Cleared {deleted_count} cached entries for {experiment_path}'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/search_transients', methods=['POST'])
def search_transients():
    """Search for database transients for current experiment"""
    current_experiment = get_current_experiment_from_main_tool()
    if not current_experiment:
        return jsonify({'success': False, 'error': 'No current experiment'})
    
    database_transients = get_database_transients(current_experiment)
    cached_result = get_saved_transient_center(current_experiment)
    
    return jsonify({
        'success': True,
        'experiment_path': current_experiment,
        'database_transients': database_transients or [],
        'cached_transient': cached_result,
        'has_database_transients': database_transients is not None
    })

@app.route('/set_transient', methods=['POST'])
def set_transient():
    """Set a manual transient override in the database"""
    data = request.json
    transient_time = data.get('transient_time')
    experiment_path = data.get('experiment_path') or get_current_experiment_from_main_tool()
    transient_index = data.get('transient_index', 1)
    
    if not transient_time or not experiment_path:
        return jsonify({'success': False, 'error': 'Missing transient_time or experiment_path'})
    
    try:
        # Convert time to sample index (5 MSPS)
        sampling_rate = 5.0e6
        transient_sample = int(float(transient_time) * sampling_rate)
        
        # Save to database
        success = save_database_transient(experiment_path, transient_sample, transient_index)
        
        if success:
            # Clear any cached ML predictions to force regeneration
            conn = sqlite3.connect(TRANSIENT_DB_PATH)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM transient_centers WHERE experiment_path = ?', (experiment_path,))
            conn.commit()
            conn.close()
            
            return jsonify({
                'success': True,
                'message': f'Transient {transient_index} set at {transient_time}s (sample {transient_sample})',
                'transient_time': transient_time,
                'transient_sample': transient_sample
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to save transient to database'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/clear_database_transients', methods=['POST'])
def clear_database_transients():
    """Clear database transients for current experiment"""
    current_experiment = get_current_experiment_from_main_tool()
    if not current_experiment:
        return jsonify({'success': False, 'error': 'No current experiment'})
    
    exp_name = current_experiment.split('/')[-1]
    
    try:
        conn = sqlite3.connect(V3_DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE files 
            SET transient1_index = NULL, transient2_index = NULL, transient3_index = NULL 
            WHERE original_filename = ?
        ''', (exp_name,))
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': success,
            'message': f'Cleared database transients for {exp_name}' if success else 'No transients found to clear'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_current_transient_info', methods=['POST'])
def get_current_transient_info():
    """Get current transient center information for zoom plot coordinate calculation"""
    data = request.json
    experiment_path = data.get('experiment_path') or get_current_experiment_from_main_tool()
    
    if not experiment_path:
        return jsonify({'success': False, 'error': 'No current experiment'})
    
    transient_center = None
    transient_sample = None
    transient_source = "none"
    
    # Use same priority system as plot generation
    # PRIORITY 1: Check for database transients
    database_transients = get_database_transients(experiment_path)
    if database_transients:
        db_transient = database_transients[0]
        transient_center = db_transient['time']
        transient_sample = db_transient['sample']
        transient_source = "database"
    else:
        # PRIORITY 2: Check for cached ML predictions
        cached_result = get_saved_transient_center(experiment_path)
        if cached_result:
            transient_center = cached_result['center_time']
            transient_sample = cached_result['center_sample']
            transient_source = "cached"
    
    return jsonify({
        'success': True,
        'experiment_path': experiment_path,
        'transient_center': transient_center,
        'transient_sample': transient_sample,
        'transient_source': transient_source,
        'has_transient': transient_center is not None
    })

@app.route('/status')
def status():
    """Status endpoint for tool communication - STANDARDIZED FORMAT"""
    return jsonify({
        'service': 'worker4_transient_editor',
        'status': 'running',
        'version': '20250602_015000_0_0_1_4',
        'api_version': '1.0',
        'universal_styling': True,
        'worker3_compatible': True,
        'worker5_compatible': True,
        'demo_ready': True
    })

@app.route('/api/analysis_result', methods=['GET'])
def get_analysis_result():
    """WORKER 5 COMPATIBILITY: Get standardized analysis result"""
    global current_file_info
    
    try:
        if not current_file_info.get('file_id'):
            return jsonify(standardize_error_response("No current experiment", 'worker4_transient_editor'))
        
        # Get current transient data
        experiment_path = current_file_info.get('experiment_path')
        if not experiment_path:
            return jsonify(standardize_error_response("No experiment path", 'worker4_transient_editor'))
        
        # Get transient information
        transient_info = get_current_transient_info_data(experiment_path)
        
        # Standardize for Worker 5
        standardized_result = standardize_transient_output({
            'experiment_path': experiment_path,
            'center_time': transient_info.get('transient_center'),
            'center_sample': transient_info.get('transient_sample'),
            'source': transient_info.get('transient_source', 'unknown'),
            'confidence': 0.8,  # Default confidence
            'total_time': 0.5,  # Standard file duration
            'data_length': 2500000,  # Standard samples
            'style': 'default',
            'plot_filename': f"transient_analysis_{current_file_info['file_id']}.png",
            'timestamp': time.time()
        })
        
        return jsonify(standardized_result)
        
    except Exception as e:
        return jsonify(standardize_error_response(str(e), 'worker4_transient_editor'))

def get_current_transient_info_data(experiment_path):
    """Helper function to get transient info without HTTP request"""
    transient_center = None
    transient_sample = None
    transient_source = "none"
    
    # PRIORITY 1: Check for database transients
    database_transients = get_database_transients(experiment_path)
    if database_transients:
        db_transient = database_transients[0]
        transient_center = db_transient['time']
        transient_sample = db_transient['sample']
        transient_source = "database"
    else:
        # PRIORITY 2: Check for cached ML predictions
        cached_result = get_saved_transient_center(experiment_path)
        if cached_result:
            transient_center = cached_result['center_time']
            transient_sample = cached_result['center_sample']
            transient_source = "cached"
    
    return {
        'transient_center': transient_center,
        'transient_sample': transient_sample,
        'transient_source': transient_source,
        'has_transient': transient_center is not None
    }

@app.route('/adjust_transient', methods=['POST'])
def adjust_transient():
    """Fine-tune transient position with coarse/fine adjustments"""
    data = request.json
    direction = data.get('direction')  # 'left' or 'right'
    adjustment_type = data.get('type')  # 'coarse' or 'fine'
    experiment_path = data.get('experiment_path') or get_current_experiment_from_main_tool()
    transient_index = data.get('transient_index', 1)
    
    if not direction or not adjustment_type or not experiment_path:
        return jsonify({'success': False, 'error': 'Missing required parameters'})
    
    try:
        # Get current transient position
        database_transients = get_database_transients(experiment_path)
        current_sample = None
        
        if database_transients:
            current_sample = database_transients[0]['sample']
        else:
            # Try to get from the current plot generation system
            # Default to center if no transient found
            current_sample = int(2.5e6 / 2)  # Middle of 0.5s file
        
        # Calculate adjustment amount
        if adjustment_type == 'coarse':
            adjustment = COARSE_ADJUSTMENT  # 2048 samples
        else:  # fine
            adjustment = FINE_ADJUSTMENT   # 256 samples
        
        # Apply direction
        if direction == 'left':
            new_sample = max(0, current_sample - adjustment)
        else:  # right
            new_sample = min(int(2.5e6 - 1), current_sample + adjustment)
        
        # Convert to time
        new_time = new_sample / SAMPLING_RATE
        
        # Save to database
        success = save_database_transient(experiment_path, new_sample, transient_index)
        
        if success:
            # Clear cached ML predictions
            conn = sqlite3.connect(TRANSIENT_DB_PATH)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM transient_centers WHERE experiment_path = ?', (experiment_path,))
            conn.commit()
            conn.close()
            
            return jsonify({
                'success': True,
                'message': f'{adjustment_type.title()} adjustment {direction}: {adjustment} samples',
                'new_time': new_time,
                'new_sample': new_sample,
                'adjustment_samples': adjustment if direction == 'right' else -adjustment,
                'adjustment_type': adjustment_type,
                'direction': direction
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to save adjusted transient'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_adjustment_info')
def get_adjustment_info():
    """Get information about adjustment values for UI display"""
    return jsonify({
        'coarse_adjustment': {
            'samples': COARSE_ADJUSTMENT,
            'time_ms': (COARSE_ADJUSTMENT / SAMPLING_RATE) * 1000,
            'description': f'1/4 of smallest segment ({COARSE_ADJUSTMENT} samples)'
        },
        'fine_adjustment': {
            'samples': FINE_ADJUSTMENT, 
            'time_ms': (FINE_ADJUSTMENT / SAMPLING_RATE) * 1000,
            'description': f'1/32 of smallest segment ({FINE_ADJUSTMENT} samples)'
        },
        'keyboard_controls': {
            'coarse_left': ['7', 'Home', '['],
            'coarse_right': ['9', 'PageUp', ']'],
            'fine_left': ['ArrowLeft', 'Numpad4', 'NumpadLeft'],
            'fine_right': ['ArrowRight', 'Numpad6', 'NumpadRight']
        }
    })

if __name__ == '__main__':
    print("=== TRANSIENT EDITOR - 20250602_011500_0_0_1_1 ===")
    print("FINE-TUNING: Keyboard controls for precise transient positioning")
    print("ADJUSTMENTS: Coarse (2048 samples) and Fine (256 samples)")
    print("KEYBOARD: 7/9 or [/] (coarse), Arrow/Numpad 4/6 (fine)")
    print("MONITORING: 250ms real-time updates")
    print(f"Raw data directory: {RAW_DATA_DIR}")
    print(f"Temp plots directory: {TEMP_PLOTS_DIR}")
    print("Starting server on http://localhost:5031")
    
    app.run(debug=True, host='0.0.0.0', port=5031)