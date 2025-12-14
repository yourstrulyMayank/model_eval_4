"""
FRTB-SA Complete Runner Script
Comprehensive solution for FRTB-SA capital calculation and reporting
"""
from flask import Blueprint, request, jsonify, render_template, current_app
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import argparse
import logging
from pathlib import Path
import json
import threading
from pathlib import Path
frtbsa_bp = Blueprint('frtbsa', __name__, url_prefix='/risk/frtbsa')
latest_results = {
    'status': 'not_run',
    'capital_charges': None,
    'summary_data': None,
    'risk_breakdown': None,
    'bucket_analysis': None,
    'timestamp': None,
    'error': None
}
# Global lock to prevent multiple simultaneous calculations
calc_lock = threading.Lock()


@frtbsa_bp.route('/')
def frtbsa_home():
    """Legacy route for direct access"""
    # Trigger if needed
    if latest_results['status'] == 'not_run':
        thread = threading.Thread(target=run_calculation_task, args=(current_app.root_path,))
        thread.daemon = True
        thread.start()
    return render_template('risk_frtbsa.html', model_name='FRTB-SA')

@frtbsa_bp.route('/run_frtbsa', methods=['POST'])
def run_frtbsa():
    # Example: expects 'input_file' in POST data
    input_file = request.form.get('input_file')
    output_prefix = request.form.get('output_prefix')
    config_file = request.form.get('config_file')
    # You can adapt this logic as needed
    runner = FRTBSARunner(config_file=config_file)
    success = runner.run(input_file, output_prefix=output_prefix)
    return jsonify({'status': 'success' if success else 'error'})



@frtbsa_bp.route('/get_results', methods=['GET'])
def get_results():
    """Get the latest calculation results for dashboard"""
    global latest_results
    
    # --- AUTO-TRIGGER LOGIC ---
    # If the UI calls this and we haven't run yet, start the calculation immediately
    if latest_results['status'] == 'not_run':
        # Start background thread
        thread = threading.Thread(target=run_calculation_task, args=(current_app.root_path,))
        thread.daemon = True
        thread.start()
        
        # Return running status immediately so UI waits
        return jsonify({'status': 'running', 'message': 'Starting calculation engine...'})

    # --- STATUS HANDLING ---
    if latest_results['status'] == 'running':
        return jsonify({'status': 'running', 'message': 'Calculating capital charges...'})
    
    if latest_results['status'] == 'error':
        return jsonify({
            'status': 'error', 
            'message': latest_results.get('error', 'Unknown error occurred')
        })
    
    if latest_results['status'] == 'complete' and latest_results['summary_data']:
        return jsonify({
            'status': 'success',
            'summary': latest_results['summary_data'],
            'risk_breakdown': latest_results['risk_breakdown'],
            'bucket_analysis': latest_results['bucket_analysis'],
            'files': latest_results.get('files', {})
        })
    
    return jsonify({'status': 'running', 'message': 'Initializing...'})


@frtbsa_bp.route('/download/<filename>')
def download_file(filename):
    """Download generated output files"""
    try:
        output_dir = os.path.join(current_app.root_path, 'frtbsa_output')
        file_path = os.path.join(output_dir, filename)
        
        if os.path.exists(file_path):
            from flask import send_file
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({'error': 'File not found'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@frtbsa_bp.route('/start_evaluation', methods=['POST'])
def start_evaluation():
    """Start custom evaluation - placeholder for future implementation"""
    # TODO: Implement custom evaluation logic
    # Will compare runner results vs uploaded Excel
    return jsonify({
        'status': 'pending',
        'message': 'Custom evaluation feature coming soon'
    })

@frtbsa_bp.route('/upload_evaluation_file', methods=['POST'])
def upload_evaluation_file():
    """Upload Excel file for custom evaluation - placeholder"""
    # TODO: Implement file upload and comparison logic
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file uploaded'})
    
    file = request.files['file']
    # Placeholder for future implementation
    return jsonify({
        'status': 'pending',
        'message': 'File upload received. Evaluation logic to be implemented.'
    })


# Import the FRTB-SA modules (ensure they are in the same directory or in Python path)
try:
    from .frtbsa_engine import FRTBSAEngine, FRTBSAConfig
    from .frtbsa_data_processor import CRIFFormatter, DataValidator, RiskAggregator, StreamingProcessor
except ImportError:
    print("Please ensure frtbsa_engine.py and frtbsa_data_processor.py are in the same directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        # logging.FileHandler(f'frtbsa_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_calculation_task(app_root_path):
    """Background task to run the calculation"""
    global latest_results
    
    with calc_lock:
        # Double check status inside lock
        if latest_results['status'] == 'complete' or latest_results['status'] == 'running':
            return

        try:
            latest_results['status'] = 'running'
            logger.info('Starting automatic FRTB-SA calculation...')
            
            # --- 1. Smart File Detection ---
            # Look for data file in common locations
            possible_paths = [
                os.path.join(app_root_path, 'FRTBSA data.xlsx'),
                os.path.join(app_root_path, 'uploads', 'FRTBSA data.xlsx'),
                os.path.join(app_root_path, 'routes', 'risk', 'frtb_sa', 'frtbsa_data.xlsx'),
                'FRTBSA data.xlsx' # CWD fallback
            ]
            
            input_file = None
            for path in possible_paths:
                if os.path.exists(path):
                    input_file = path
                    break
            
            if not input_file:
                error_msg = f'Input file "FRTBSA data.xlsx" not found. Checked: {[p for p in possible_paths]}'
                logger.error(error_msg)
                latest_results['status'] = 'error'
                latest_results['error'] = error_msg
                return

            # --- 2. Run Calculation ---
            runner = FRTBSARunner()
            success = runner.run(input_file, output_prefix=f"AUTO_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            if not success:
                # Error details are handled inside runner.run() -> latest_results
                if latest_results['status'] != 'error':
                    latest_results['status'] = 'error'
                    latest_results['error'] = 'Calculation failed during execution'
                
        except Exception as e:
            logger.error(f"Auto-calculation critical error: {e}", exc_info=True)
            latest_results['status'] = 'error'
            latest_results['error'] = str(e)


class FRTBSARunner:
    def __init__(self, config_file: str = None):
        self.engine = FRTBSAEngine()
        self.validator = DataValidator()
        self.crif_formatter = CRIFFormatter()
        self.aggregator = RiskAggregator()
        self.config = self._default_config()
        
        # Ensure output dir exists
        # Use current_app path if available to ensure correct location
        try:
            base_path = current_app.root_path
        except:
            base_path = os.getcwd()
            
        self.output_dir = Path(os.path.join(base_path, 'frtbsa_output'))
        self.output_dir.mkdir(exist_ok=True)

    def _default_config(self) -> dict:
        return {
            'generate_crif': True,
            'validate_data': True,
            'generate_detailed_reports': True,
            'valuation_date': datetime.now().strftime('%Y-%m-%d')
        }

    def run(self, input_file: str, output_prefix: str = None):
        global latest_results
        try:
            if output_prefix is None:
                output_prefix = f"FRTBSA_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # 1. Load Data
            data = self.engine.load_data(input_file)
            
            # 2. Calculate
            capital_charges = self.engine.calculate_capital_charges(data)
            
            # 3. Generate Reports (Silent/Background)
            qis_file = self.output_dir / f"{output_prefix}_QIS.xlsx"
            self.engine.generate_qis_report(capital_charges, str(qis_file))
            
            summary_file = self.output_dir / f"{output_prefix}_Summary.xlsx"
            risk_file = self.output_dir / f"{output_prefix}_Risk_Breakdown.csv"
            
            # 4. Prepare Results for UI
            results_dict = self.get_results_dict(capital_charges, data)
            
            # Add file paths for download
            results_dict['files'] = {
                'qis': qis_file.name,
                'summary': summary_file.name,
                'risk_breakdown': risk_file.name
            }

            # Update Global State
            latest_results.update({
                'status': 'complete',
                'capital_charges': capital_charges,
                'summary_data': results_dict,
                'risk_breakdown': results_dict['risk_breakdown'],
                'bucket_analysis': results_dict['bucket_analysis'],
                'files': results_dict['files'],
                'timestamp': datetime.now().isoformat()
            })
            
            return True

        except Exception as e:
            logger.error(f"Runner Execution Error: {e}", exc_info=True)
            latest_results['status'] = 'error'
            latest_results['error'] = str(e)
            return False

    def get_results_dict(self, capital_charges: dict, data: pd.DataFrame) -> dict:
        """Format results for JSON response"""
        
        # Risk Breakdown
        risk_breakdown = []
        for risk_class in data['RiskClass'].unique():
            class_data = data[data['RiskClass'] == risk_class]
            # Handle charge mapping safely
            charge_key = risk_class.replace(' ', '_').upper()
            # Map common variations
            if 'CREDIT' in charge_key: charge_key = 'CSR_NON_SEC' # Simplified mapping
            
            breakdown = {
                'Risk Class': risk_class,
                'Trade Count': int(len(class_data)),
                'Total Sensitivity': float(class_data['FS Amount USD'].sum() if 'FS Amount USD' in class_data.columns else 0),
                'Capital Charge': float(capital_charges.get(charge_key, 0))
            }
            risk_breakdown.append(breakdown)
            
        # Clean Capital Charges (handle numpy types)
        clean_charges = {k: float(v) for k, v in capital_charges.items()}
        
        return {
            'Capital_Charges': clean_charges,
            'risk_breakdown': risk_breakdown,
            'bucket_analysis': [], # Simplified for now
            'TOTAL': clean_charges.get('TOTAL', 0)
        }

def create_sample_config():
    """Create a sample configuration file"""

    config = {
        "output_dir": "frtbsa_output",
        "generate_crif": True,
        "validate_data": True,
        "generate_detailed_reports": True,
        "chunk_size": 10000,
        "valuation_date": "2025-10-16",
        "risk_weights": {
            "override_defaults": False,
            "custom_weights": {}
        },
        "correlation_parameters": {
            "use_custom": False,
            "custom_correlations": {}
        }
    }

    with open("frtbsa_config.json", 'w') as f:
        json.dump(config, f, indent=2)

    print("Sample configuration file created: frtbsa_config.json")

def main():
    """Main entry point"""

    parser = argparse.ArgumentParser(
        description='FRTB-SA Capital Calculation Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python frtbsa_runner.py input.xlsx
  python frtbsa_runner.py input.csv --output-prefix Q4_2025
  python frtbsa_runner.py input.xlsx --config config.json
  python frtbsa_runner.py --create-config
        """
    )

    parser.add_argument('input_file', nargs='?', help='Input data file (Excel or CSV)')
    parser.add_argument('--output-prefix', help='Prefix for output files')
    parser.add_argument('--config', help='Configuration file (JSON)')
    parser.add_argument('--create-config', action='store_true',
                       help='Create sample configuration file')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create sample config if requested
    if args.create_config:
        create_sample_config()
        return

    # Check input file
    if not args.input_file:
        parser.print_help()
        return

    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found")
        return

    # Initialize and run
    runner = FRTBSARunner(config_file=args.config)
    success = runner.run(args.input_file, output_prefix=args.output_prefix)

    if success:
        print("\n✓ FRTB-SA calculation completed successfully!")
    else:
        print("\n✗ FRTB-SA calculation failed. Check logs for details.")
        sys.exit(1)

# ============================================================================
# USAGE GUIDE
# ============================================================================

"""
FRTB-SA IMPLEMENTATION USAGE GUIDE
===================================

1. INSTALLATION
---------------
Required packages:
pip install pandas numpy openpyxl xlsxwriter

2. FILE STRUCTURE
-----------------
Ensure these files are in the same directory:
- frtbsa_engine.py          : Main calculation engine
- frtbsa_data_processor.py  : Data processing and validation
- frtbsa_runner.py          : This file - main runner script

3. INPUT DATA FORMAT
--------------------
Your input Excel/CSV file should have these columns:
- RiskClass: Risk classification (GIRR, Credit Spread, Equity, etc.)
- Risk_Type: Specific risk type
- Bucket: Risk bucket identifier
- Label1: Primary label (e.g., tenor for GIRR)
- Label2: Secondary label (e.g., curve type)
- FS Amount USD: Sensitivity amount in USD
- Trade_ID: Trade identifier
- Book_ID: Book/Portfolio identifier
- And other columns as per FRTB-SA requirements

4. BASIC USAGE
--------------
# Simple run with Excel input:
python frtbsa_runner.py "FRTBSA data.xlsx"

# With custom output prefix:
python frtbsa_runner.py "FRTBSA data.xlsx" --output-prefix "Q4_2025"

# With configuration file:
python frtbsa_runner.py "FRTBSA data.xlsx" --config my_config.json

5. CONFIGURATION
----------------
Create a configuration file:
python frtbsa_runner.py --create-config

This creates frtbsa_config.json which you can customize.

6. OUTPUT FILES
---------------
The script generates:
- *_QIS.xlsx           : QIS format report (main regulatory output)
- *_Summary.xlsx       : Executive summary
- *_CRIF.txt          : CRIF format for data exchange
- *_Risk_Breakdown.csv : Detailed risk class analysis
- validation_report.json : Data validation results

7. PROGRAMMATIC USAGE
---------------------
from frtbsa_runner import FRTBSARunner

# Initialize runner
runner = FRTBSARunner()

# Run calculation
runner.run("input_data.xlsx", output_prefix="Q4_2025")

8. LARGE FILE PROCESSING
------------------------
For files with >100K rows, use streaming:

from frtbsa_data_processor import StreamingProcessor
processor = StreamingProcessor(chunk_size=10000)
processor.process_large_file("large_input.csv", "output.csv", process_func)

9. TROUBLESHOOTING
------------------
- Check validation_report.json for data issues
- Review log files for detailed error messages
- Ensure all required columns are present
- Verify numeric fields don't contain text

10. SUPPORT
-----------
For regulatory questions, refer to:
- BCBS-D457 documentation
- ISDA SIMM/FRTB specifications

"""

if __name__ == "__main__":
    main()
