# Arc Detection Transient Analysis Service

A specialized service for detecting and analyzing electrical transients in arc detection data. This service provides ML-based transient detection, coordinate mapping, and visualization capabilities.

## Features

- **Transient Detection**: Progressive ML-based detection with configurable chunk sizes
- **Coordinate Mapping**: Fixed coordinate system handling between downsampled detection and full-resolution visualization
- **Interactive Editor**: Fine-tuning interface for precise transient positioning
- **Styles Gallery**: Consistent styling across visualizations
- **Universal Image Save**: Enhanced image saving with metadata and multiple format support

## Services

### Main Viewer (`app.py`)
- Port: 5031 (default)
- Synchronized transient prediction viewer
- ML model integration with caching
- V3 database support

### Transient Editor (`services/editor.py`)
- Interactive transient position fine-tuning
- Keyboard controls for coarse/fine adjustments
- Real-time coordinate mapping fixes
- Database integration for manual overrides

## Quick Start

```bash
# Install dependencies
pip install flask numpy scipy matplotlib requests

# Start the main service
python app.py

# Start the editor service (separate terminal)
python services/editor.py
```

## Configuration

Key configuration variables in both services:
- `V3_DATABASE_PATH`: Path to V3 database
- `RAW_DATA_DIR`: Raw data directory
- `TEMP_PLOTS_DIR`: Temporary plots output directory

## API Endpoints

### Main Service
- `GET /`: Main viewer interface
- `POST /sync`: Sync with data review tool
- `GET /refresh`: Refresh current experiment view
- `GET /status`: Service status

### Editor Service
- `GET /`: Transient editor interface
- `POST /adjust_transient`: Fine-tune transient position
- `POST /set_transient`: Set manual transient override
- `POST /clear_cache`: Clear cached ML predictions

## Models

Includes pre-trained ML models in `models/` directory:
- 6-feature models for transient detection
- Balanced and improved variants
- Chunking-based detection models

## Version

Current version: `20250602_013000_0_0_1_1`

## Dependencies

- Flask
- NumPy
- SciPy
- Matplotlib
- SQLite3
- Requests

## Integration

This service integrates with:
- Enhanced Data Cleaning Tool (sync protocol)
- V3 Database system
- Arc Detection main coordination service