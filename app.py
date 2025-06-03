#!/usr/bin/env python3
"""
Transient Prediction Viewer - Version 20250528_200000_001_001_001_002
Synchronized with Enhanced Data Cleaning Tool
Shows predicted transient locations with color-coded data points.

CHANGES in 002:
- Fixed UnboundLocalError for current_experiment variable
- Enhanced sync functionality to generate plots for file_id
- Fixed duplicate /sync routes
- Added proper V3 database support for file_id based syncing
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

app = Flask(__name__)

# Configuration
V3_DATABASE_PATH = "/Volumes/ArcData/V3_database/arc_detection.db"
BINARY_DATA_DIR = "/Volumes/ArcData/V3_database/fileset"
TEMP_PLOTS_DIR = "/Users/kjensen/Documents/GitHub/data_processor_project/arc_detection_project/temp_transient_plots"
DB_DIR = "/Volumes/ArcData/V3_database"
MAIN_TOOL_URL = "http://localhost:5030"

# Ensure temp plots directory exists
os.makedirs(TEMP_PLOTS_DIR, exist_ok=True)

# Initialize transient centers database
TRANSIENT_DB_PATH = os.path.join(DB_DIR, "transient_centers.db")

def init_transient_database():
    """Initialize database for storing fine-tuned transient centers"""
    conn = sqlite3.connect(TRANSIENT_DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transient_centers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_path TEXT NOT NULL,
            center_sample INTEGER NOT NULL,
            center_time REAL NOT NULL,
            detection_method TEXT,
            initial_chunk INTEGER,
            final_chunk INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(experiment_path, detection_method, initial_chunk, final_chunk)
        )
    ''')
    
    conn.commit()
    conn.close()

# Initialize the database
init_transient_database()

def get_local_ip():
    """Get the local IP address for network access"""
    import socket
    try:
        # Connect to a remote address to determine local IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except:
        return "localhost"

def load_experiment_data(experiment_path):
    """Load voltage and current data from experiment directory"""
    if not experiment_path or '/' not in experiment_path:
        return None, None, None
    
    exp_type, exp_name = experiment_path.split('/')
    exp_full_path = os.path.join(RAW_DATA_DIR, exp_type, exp_name)
    
    if not os.path.exists(exp_full_path):
        return None, None, None
    
    # Find .mat files
    mat_files = [f for f in os.listdir(exp_full_path) if f.endswith('.mat')]
    if len(mat_files) == 0:
        return None, None, None
    
    voltage_data = None
    current_data = None
    
    for mat_file in mat_files:
        file_path = os.path.join(exp_full_path, mat_file)
        try:
            mat_data = scipy.io.loadmat(file_path)
            # Try both 'data' and 'y' keys
            if 'data' in mat_data:
                data = mat_data['data'].flatten()
            elif 'y' in mat_data:
                data = mat_data['y'].flatten()
            else:
                continue
            
            # Check if this is voltage or current based on filename
            if 'CH1' in mat_file.upper() or 'VOLTAGE' in mat_file.upper():
                voltage_data = data
            elif 'CH4' in mat_file.upper() or 'CURRENT' in mat_file.upper():
                current_data = data
        except Exception as e:
            print(f"Error loading {mat_file}: {e}")
            continue
    
    return voltage_data, current_data, exp_full_path

def get_cached_transient_center(experiment_path, method="fine", initial_chunk=8192, final_chunk=128):
    """Get cached transient center for experiment"""
    conn = sqlite3.connect(TRANSIENT_DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT center_sample, center_time, timestamp 
        FROM transient_centers 
        WHERE experiment_path = ? AND detection_method = ? 
              AND initial_chunk = ? AND final_chunk = ?
        ORDER BY timestamp DESC LIMIT 1
    ''', (experiment_path, method, initial_chunk, final_chunk))
    
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return {
            'center_sample': result[0],
            'center_time': result[1],
            'timestamp': result[2]
        }
    return None

def cache_transient_center(experiment_path, center_sample, center_time, method="fine", initial_chunk=8192, final_chunk=128):
    """Cache transient center result"""
    conn = sqlite3.connect(TRANSIENT_DB_PATH)
    cursor = conn.cursor()
    
    # Insert or replace the cached result
    cursor.execute('''
        INSERT OR REPLACE INTO transient_centers 
        (experiment_path, center_sample, center_time, detection_method, initial_chunk, final_chunk)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (experiment_path, center_sample, center_time, method, initial_chunk, final_chunk))
    
    conn.commit()
    conn.close()

def clear_transient_cache(experiment_path):
    """Clear all cached results for an experiment"""
    conn = sqlite3.connect(TRANSIENT_DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('DELETE FROM transient_centers WHERE experiment_path = ?', (experiment_path,))
    rows_deleted = cursor.rowcount
    
    conn.commit()
    conn.close()
    
    return rows_deleted

def detect_transient_center_progressive(voltage_data, initial_chunk=8192, final_chunk=128):
    """Progressive transient detection with decreasing chunk sizes"""
    if len(voltage_data) < initial_chunk:
        return None
    
    current_chunk = initial_chunk
    search_start = 0
    search_end = len(voltage_data)
    
    while current_chunk >= final_chunk:
        print(f"Processing with chunk size: {current_chunk}")
        
        # Calculate variance for each chunk
        chunks = []
        for i in range(search_start, min(search_end, len(voltage_data) - current_chunk), current_chunk // 4):
            chunk = voltage_data[i:i + current_chunk]
            variance = np.var(chunk)
            chunks.append((i, variance))
        
        if not chunks:
            break
        
        # Find chunk with highest variance
        max_chunk = max(chunks, key=lambda x: x[1])
        center_sample = max_chunk[0] + current_chunk // 2
        
        # Narrow search window for next iteration
        window_size = current_chunk * 2
        search_start = max(0, center_sample - window_size // 2)
        search_end = min(len(voltage_data), center_sample + window_size // 2)
        
        # Reduce chunk size
        current_chunk = current_chunk // 2
        
        if current_chunk < final_chunk:
            break
    
    return center_sample

def detect_transient_with_center_preference(voltage_data, target_transient_sample):
    """
    Detect transient with preference for center location
    Enhanced logic for handling transients in different segments with proper bounds checking
    """
    
    data_length = len(voltage_data)
    segment_size = data_length // 5  # 5 segments: A, B, C, D, E
    
    # Determine which segment the target transient is in
    segment = min(4, target_transient_sample // segment_size)  # Ensure segment <= 4
    
    print(f"Target transient at sample {target_transient_sample}, segment {segment}")
    print(f"Data length: {data_length}, segment size: {segment_size}")
    
    if segment <= 1:  # A or B segment - transient in first 40%
        # NEW LOGIC 20250526_124000: If transient is in first segment, make C start at 0
        if segment == 0:
            print(f"DEBUG BOTTOM 20250526_124000: Transient in first segment, C starts at 0 (samples 0-{segment_size-1})")
        
        A_start, A_end = 0, segment_size
        B_start, B_end = segment_size, 2 * segment_size  
        C_start, C_end = 2 * segment_size, 4 * segment_size
        D_start, D_end = 4 * segment_size, data_length
        
        # Calculate center sample, ensuring it doesn't go negative
        center_sample = max(0, target_transient_sample - segment_size)
        center_file_start = center_sample / 5000000  # Convert to time
        
        if center_file_start < 0:
            center_file_start = 0
            center_sample = 0
            print(f"DEBUG BOTTOM 20250526_124000: Center file start {center_file_start:.3f}s is negative, starting at 0")
    
    elif segment >= 3:  # D or E segment - transient in last 40%
        # Standard logic for rear transients
        A_start, A_end = 0, 2 * segment_size
        B_start, B_end = 2 * segment_size, target_transient_sample - segment_size//2
        C_start, C_end = target_transient_sample - segment_size//2, target_transient_sample + segment_size//2
        D_start, D_end = target_transient_sample + segment_size//2, data_length
        
        # Place center file to capture transient in middle
        center_sample = max(0, target_transient_sample - 2*segment_size)
        center_file_start = center_sample / 5000000
    
    else:  # C segment - transient in middle 20%
        # Standard balanced segmentation
        A_start, A_end = 0, segment_size
        B_start, B_end = segment_size, target_transient_sample - segment_size//2  
        C_start, C_end = target_transient_sample - segment_size//2, target_transient_sample + segment_size//2
        D_start, D_end = target_transient_sample + segment_size//2, 4 * segment_size
        E_start, E_end = 4 * segment_size, data_length
        
        center_sample = max(0, target_transient_sample - 2*segment_size)
        center_file_start = center_sample / 5000000
    
    # Ensure all bounds are within data limits
    A_end = min(A_end, data_length)
    B_end = min(B_end, data_length) 
    C_end = min(C_end, data_length)
    D_end = min(D_end, data_length)
    if 'E_end' in locals():
        E_end = min(E_end, data_length)
    
    # Convert to time for result
    center_time = center_sample / 5000000
    
    segments = {
        'A': (A_start, A_end),
        'B': (B_start, B_end), 
        'C': (C_start, C_end),
        'D': (D_start, D_end)
    }
    
    if 'E_start' in locals():
        segments['E'] = (E_start, E_end)
    
    return {
        'center_sample': center_sample,
        'center_time': center_time,
        'center_file_start': center_file_start,
        'segments': segments,
        'target_segment': chr(65 + segment)  # A, B, C, D, E
    }

def get_current_experiment_from_main_tool():
    """Get current experiment from the main data cleaning tool"""
    try:
        response = requests.get(f"{MAIN_TOOL_URL}/current_experiment", timeout=2)
        if response.status_code == 200:
            data = response.json()
            return data.get('experiment_path')
    except:
        pass
    
    # Fallback: check sync file
    sync_file = os.path.join(DB_DIR, "current_experiment.sync")
    if os.path.exists(sync_file):
        with open(sync_file, 'r') as f:
            return f.read().strip()
    
    # Final fallback: scan directory for experiments
    experiments = []
    try:
        if os.path.exists(RAW_DATA_DIR):
            for exp_type in os.listdir(RAW_DATA_DIR):
                exp_type_path = os.path.join(RAW_DATA_DIR, exp_type)
                if os.path.isdir(exp_type_path):
                    for exp_name in os.listdir(exp_type_path):
                        exp_path = os.path.join(exp_type_path, exp_name)
                        if os.path.isdir(exp_path):
                            experiments.append(f"{exp_type}/{exp_name}")
        
        # Sort and return most recent
        experiments.sort(reverse=True)
    except Exception as e:
        print(f"Error scanning experiments: {e}")
    
    # Return first experiment as fallback
    return experiments[0] if experiments else None

def create_transient_plot_by_file_id(file_id, filename):
    """Create transient plot for a specific file_id from V3 database with transient detection"""
    try:
        # Load binary data
        binary_path = os.path.join(BINARY_DATA_DIR, f"{file_id:08d}.npy")
        if not os.path.exists(binary_path):
            return None
            
        data = np.load(binary_path)
        if len(data.shape) == 2:
            if data.shape[1] == 2:
                voltage, current = data[:, 0], data[:, 1]
            elif data.shape[0] == 2:
                voltage, current = data[0], data[1]
            else:
                return None
        else:
            return None
        
        # Create time array (assuming 5MHz sampling rate)
        time = np.linspace(0, len(voltage)/5000000, len(voltage))
        
        # Detect transient center using progressive detection
        print(f"Running transient detection for file_id {file_id}...")
        transient_center = detect_transient_center_progressive(voltage)
        
        if transient_center is None:
            print(f"No transient detected in file_id {file_id}")
            # Still create plot but without transient segmentation
            fig, axes = plt.subplots(2, 1, figsize=(20, 12))
            fig.suptitle(f'Transient Analysis - File ID: {file_id} - NO TRANSIENT DETECTED', fontsize=18, fontweight='bold', color='red', y=0.95)
            
            axes[0].plot(time, voltage, color='gray', linewidth=0.5)
            axes[0].set_ylabel('Load Voltage (V)\n(CH1)', fontsize=12)
            axes[0].grid(True, alpha=0.3)
            axes[0].set_title(f'Filename: {filename}', fontsize=10)
            
            axes[1].plot(time, current, color='gray', linewidth=0.5)
            axes[1].set_ylabel('Source Current (A)\n(CH4)', fontsize=12)
            axes[1].set_xlabel('Time (s)', fontsize=12)
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout(rect=[0, 0.02, 1, 0.94])
        else:
            print(f"Transient detected at sample {transient_center} ({transient_center/5000000:.4f}s)")
            
            # Get center-preference segmentation
            center_result = detect_transient_with_center_preference(voltage, transient_center)
            segments = center_result['segments']
            
            # Create plot with color-coded segments
            fig, axes = plt.subplots(2, 1, figsize=(20, 12))
            fig.suptitle(f'Transient Prediction - File ID: {file_id} - Transient at {transient_center/5000000:.4f}s', 
                        fontsize=18, fontweight='bold', y=0.95)
            
            # Color mapping for segments
            colors = {'A': 'green', 'B': 'green', 'C': 'red', 'D': 'blue', 'E': 'orange'}
            
            # Plot voltage with color coding
            for segment_name, (start, end) in segments.items():
                if start < len(time) and end <= len(time) and start < end:
                    color = colors.get(segment_name, 'gray')
                    axes[0].plot(time[start:end], voltage[start:end], color=color, linewidth=0.8,
                               label=f'Segment {segment_name}' if segment_name == 'C' else None)
            
            axes[0].set_ylabel('Load Voltage (V)\n(CH1)', fontsize=12)
            axes[0].grid(True, alpha=0.3)
            axes[0].set_title(f'Filename: {filename} - Segment: {center_result.get("target_segment", "?")}', fontsize=10)
            
            # Plot current with color coding
            for segment_name, (start, end) in segments.items():
                if start < len(time) and end <= len(time) and start < end:
                    color = colors.get(segment_name, 'gray')
                    axes[1].plot(time[start:end], current[start:end], color=color, linewidth=0.8)
            
            axes[1].set_ylabel('Source Current (A)\n(CH4)', fontsize=12)
            axes[1].set_xlabel('Time (s)', fontsize=12)
            axes[1].grid(True, alpha=0.3)
            
            # Add transient marker
            transient_time = transient_center / 5000000
            for i, ax in enumerate(axes):
                ax.axvline(x=transient_time, color='black', linestyle='--', linewidth=2, alpha=0.8)
            
            # Add legend to voltage plot
            axes[0].legend(loc='upper right', fontsize=10)
            
            plt.tight_layout(rect=[0, 0.02, 1, 0.94])
        
        # Save plot
        plot_filename = f"file_{file_id:08d}_transient_v002.png"
        plot_path = os.path.join(TEMP_PLOTS_DIR, plot_filename)
        plt.savefig(plot_path, dpi=200, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
        print(f"Generated transient prediction plot: {plot_filename}")
        return plot_filename
        
    except Exception as e:
        print(f"Error creating transient plot for file_id {file_id}: {e}")
        return None

def create_transient_prediction_plot(experiment_path):
    """Create plot showing transient predictions with color coding"""
    if not experiment_path or '/' not in experiment_path:
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
            # Try both 'data' and 'y' keys
            if 'data' in mat_data:
                data = mat_data['data'].flatten()
            elif 'y' in mat_data:
                data = mat_data['y'].flatten()
            else:
                continue
            
            # Check if this is voltage or current based on filename
            if 'CH1' in mat_file.upper() or 'VOLTAGE' in mat_file.upper():
                voltage_data = data
            elif 'CH4' in mat_file.upper() or 'CURRENT' in mat_file.upper():
                current_data = data
        except Exception as e:
            print(f"Error loading {mat_file}: {e}")
            continue
    
    if voltage_data is None or current_data is None:
        return None
    
    # Ensure both datasets are same length and handle 2.5M samples (0.5s duration)
    min_length = min(len(voltage_data), len(current_data))
    
    # Limit to 2.5M samples for 0.5s duration at 5MSPS
    target_samples = 2500000
    if min_length > target_samples:
        voltage_data = voltage_data[:target_samples]
        current_data = current_data[:target_samples]
        min_length = target_samples
        print(f"Trimmed data to {target_samples} samples (0.5s @ 5MSPS)")
    else:
        voltage_data = voltage_data[:min_length]
        current_data = current_data[:min_length]
        print(f"Using {min_length} samples ({min_length/5000000:.3f}s @ 5MSPS)")
    
    # Create time array (5MSPS sampling rate)
    time = np.linspace(0, len(voltage_data)/5000000, len(voltage_data))
    
    # Detect transient center
    transient_center = detect_transient_center_progressive(voltage_data)
    
    if transient_center is None:
        print("No transient detected")
        return None
    
    # Get center-preference segmentation
    center_result = detect_transient_with_center_preference(voltage_data, transient_center)
    segments = center_result['segments']
    
    # Create plot
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    
    # Color mapping for segments
    colors = {'A': 'green', 'B': 'green', 'C': 'red', 'D': 'blue', 'E': 'orange'}
    
    # Plot voltage with color coding
    for segment_name, (start, end) in segments.items():
        if start < len(time) and end <= len(time):
            color = colors.get(segment_name, 'gray')
            axes[0].plot(time[start:end], voltage_data[start:end], color=color, linewidth=0.5)
    
    axes[0].set_ylabel('Load Voltage (V) - CH1')
    axes[0].set_title(f'Transient Prediction: {exp_name} (Rev: 20250528_200000_001_001_001_002)')
    axes[0].grid(True, alpha=0.3)
    
    # Plot current with color coding
    for segment_name, (start, end) in segments.items():
        if start < len(time) and end <= len(time):
            color = colors.get(segment_name, 'gray')
            axes[1].plot(time[start:end], current_data[start:end], color=color, linewidth=0.5)
    
    axes[1].set_ylabel('Source Current (A) - CH4')
    axes[1].set_xlabel('Time (s)')
    axes[1].grid(True, alpha=0.3)
    
    # Add transient marker
    transient_time = transient_center / 5000000
    for ax in axes:
        ax.axvline(x=transient_time, color='black', linestyle='--', linewidth=2, alpha=0.7)
        ax.text(transient_time, ax.get_ylim()[1] * 0.9, f'Transient: {transient_time:.4f}s', 
                rotation=90, fontsize=8, ha='right', va='top')
    
    plt.tight_layout()
    
    # Save plot
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    plot_filename = f"{exp_name}_transient_prediction_{timestamp}_002_001_001_002.png"
    plot_path = os.path.join(TEMP_PLOTS_DIR, plot_filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_filename

@app.route('/')
def index():
    """Main transient prediction viewer"""
    global current_file_info
    
    plot_filename = None
    experiment_name = None
    current_experiment = None
    
    # Check if we have synced file info first
    if current_file_info.get('file_id'):
        file_id = current_file_info['file_id']
        filename = current_file_info.get('filename', f'file_{file_id:08d}')
        experiment_name = f"File ID: {file_id} ({filename})"
        current_experiment = f"file_id_{file_id}"  # Set a value for template
        print(f"Creating transient plot for synced file: {experiment_name}")
        try:
            plot_filename = create_transient_plot_by_file_id(file_id, filename)
            print(f"Plot created: {plot_filename}")
        except Exception as e:
            print(f"Error creating plot for synced file: {e}")
    else:
        # Fallback to original experiment logic
        current_experiment = get_current_experiment_from_main_tool()
        
        if current_experiment:
            experiment_name = current_experiment.split('/')[-1] if '/' in current_experiment else current_experiment
            print(f"Creating transient plot for: {experiment_name}")
            try:
                plot_filename = create_transient_prediction_plot(current_experiment)
                print(f"Plot created: {plot_filename}")
            except Exception as e:
                print(f"Error creating plot: {e}")
                experiment_name = f"Error: {str(e)}"
    
    return render_template('transient_viewer_20250528_200000_001_001_001_002.html', 
                         experiment_path=current_experiment,
                         experiment_name=experiment_name,
                         plot_filename=plot_filename)

@app.route('/plot/<filename>')
def serve_plot(filename):
    """Serve transient prediction plot images"""
    return send_from_directory(TEMP_PLOTS_DIR, filename)

@app.route('/sync_experiment', methods=['POST'])
def sync_experiment():
    """Sync to a specific experiment (called by main tool)"""
    data = request.json
    experiment_path = data.get('experiment_path') if data else None
    
    if not experiment_path:
        return jsonify({'success': False, 'error': 'No experiment path provided'})
    # Save current experiment to sync file
    sync_file = os.path.join(DB_DIR, "current_experiment.sync")
    with open(sync_file, 'w') as f:
        f.write(experiment_path)
    
    # Generate new plot
    plot_filename = create_transient_prediction_plot(experiment_path)
    experiment_name = experiment_path.split('/')[-1] if '/' in experiment_path else experiment_path
    
    return jsonify({
        'success': True,
        'experiment_path': experiment_path,
        'experiment_name': experiment_name,
        'plot_filename': plot_filename
    })

@app.route('/refresh')
def refresh():
    """Refresh current plot"""
    current_experiment = get_current_experiment_from_main_tool()
    
    if not current_experiment:
        return jsonify({'success': False, 'error': 'No current experiment'})
    
    plot_filename = create_transient_prediction_plot(current_experiment)
    experiment_name = current_experiment.split('/')[-1] if '/' in current_experiment else current_experiment
    
    return jsonify({
        'success': True,
        'experiment_path': current_experiment,
        'experiment_name': experiment_name,
        'plot_filename': plot_filename
    })

@app.route('/fine_detection', methods=['POST'])
def fine_detection():
    """Run fine transient detection with custom parameters"""
    data = request.json
    experiment_path = data.get('experiment_path')
    initial_chunk = data.get('initial_chunk', 8192)
    final_chunk = data.get('final_chunk', 128)
    
    if not experiment_path:
        return jsonify({'success': False, 'error': 'No experiment path provided'})
    
    # Check cache first
    cached_result = get_cached_transient_center(experiment_path, "fine", initial_chunk, final_chunk)
    if cached_result:
        print(f"Found cached result: {cached_result}")
        # Generate new plot with cached data
        plot_filename = create_transient_prediction_plot(experiment_path)
        return jsonify({
            'success': True,
            'cached': True,
            'result': cached_result,
            'plot_filename': plot_filename
        })
    
    # Load experiment data
    voltage_data, current_data, exp_path = load_experiment_data(experiment_path)
    
    if voltage_data is None:
        return jsonify({'success': False, 'error': 'Could not load experiment data'})
    
    # Run fine detection
    center_sample = detect_transient_center_progressive(voltage_data, initial_chunk, final_chunk)
    
    if center_sample is None:
        return jsonify({'success': False, 'error': 'No transient detected'})
    
    center_time = center_sample / 5000000  # Convert to time assuming 5MSPS
    
    # Cache the result
    cache_transient_center(experiment_path, center_sample, center_time, "fine", initial_chunk, final_chunk)
    
    # Generate new plot
    plot_filename = create_transient_prediction_plot(experiment_path)
    
    return jsonify({
        'success': True,
        'cached': False,
        'center_sample': center_sample,
        'center_time': center_time,
        'plot_filename': plot_filename
    })

@app.route('/get_cached_center', methods=['POST'])
def get_cached_center():
    """Get cached transient center result"""
    data = request.json
    experiment_path = data.get('experiment_path')
    
    if not experiment_path:
        return jsonify({'success': False, 'error': 'No experiment path provided'})
    
    cached_result = get_cached_transient_center(experiment_path)
    
    if cached_result:
        return jsonify({
            'success': True,
            'cached': True,
            'result': cached_result
        })
    else:
        return jsonify({
            'success': True,
            'cached': False,
            'message': 'No cached result found'
        })

@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    """Clear cached results for experiment"""
    data = request.json
    experiment_path = data.get('experiment_path')
    
    if not experiment_path:
        return jsonify({'success': False, 'error': 'No experiment path provided'})
    
    rows_deleted = clear_transient_cache(experiment_path)
    
    return jsonify({
        'success': True,
        'rows_deleted': rows_deleted,
        'message': f'Cleared {rows_deleted} cached results'
    })

@app.route('/status')
def status():
    """Status endpoint for health checks"""
    return jsonify({'status': 'running', 'service': 'transient_prediction_viewer', 'version': '20250528_200000_001_001_001_002'})

@app.route('/sync', methods=['POST'])
def sync():
    """Sync with main data review tool"""
    global current_file_info
    data = request.json
    file_id = data.get('file_id')
    filename = data.get('filename')
    
    current_file_info['file_id'] = file_id
    current_file_info['filename'] = filename
    
    # Generate plot for the synced file
    plot_filename = None
    if file_id:
        try:
            plot_filename = create_transient_plot_by_file_id(file_id, filename)
        except Exception as e:
            print(f"Error creating plot for file_id {file_id}: {e}")
    
    return jsonify({
        'success': True, 
        'synced_file': current_file_info,
        'plot_filename': plot_filename
    })

@app.route('/shutdown', methods=['POST'])
def shutdown():
    """Shutdown endpoint"""
    import threading
    import time
    
    def delayed_shutdown():
        time.sleep(1)
        os._exit(0)
    
    threading.Thread(target=delayed_shutdown).start()
    return jsonify({'success': True, 'message': 'Shutting down...'})

# Global variable for current file info
current_file_info = {'file_id': None, 'filename': None}

if __name__ == '__main__':
    local_ip = get_local_ip()
    
    print("=== TRANSIENT PREDICTION VIEWER ===")
    print(f"V3 Database: {V3_DATABASE_PATH}")
    print(f"Binary data directory: {BINARY_DATA_DIR}")
    print(f"Temp plots directory: {TEMP_PLOTS_DIR}")
    print(f"Syncing with main tool at: {MAIN_TOOL_URL}")
    print()
    print("🌐 Web Interface Available At:")
    print(f"   Local:   http://localhost:5031")
    print(f"   Network: http://{local_ip}:5031")
    print()
    print("📱 Access from other devices on your network using the Network URL")
    
    app.run(debug=True, host='0.0.0.0', port=5031)