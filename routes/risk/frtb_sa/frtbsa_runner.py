"""
FRTB-SA Complete Runner Script
Comprehensive solution for FRTB-SA capital calculation and reporting
"""
from flask import Blueprint, request, jsonify, render_template
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
@frtbsa_bp.route('/')
def frtbsa_home():
    # Trigger automatic calculation in background
    def run_calculation():
        global latest_results
        try:
            input_file = r'routes\risk\frtb_sa\frtbsa_data.xlsx'  # Default input file
            if not os.path.exists(input_file):
                latest_results['status'] = 'error'
                latest_results['error'] = 'Input file "FRTBSA data.xlsx" not found'
                logger.error(f'Input file not found: {input_file}')
                return
            
            latest_results['status'] = 'running'
            logger.info('Starting automatic FRTB-SA calculation...')
            
            runner = FRTBSARunner()
            
            # Run calculation
            success = runner.run(input_file, output_prefix=f"AUTO_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            if not success:
                latest_results['status'] = 'error'
                latest_results['error'] = 'Calculation failed - check logs for details'
                logger.error('FRTB-SA calculation failed')
                
        except Exception as e:
            logger.error(f"Auto-calculation error: {e}", exc_info=True)
            latest_results['status'] = 'error'
            latest_results['error'] = str(e)
    
    # Start calculation in background thread
    thread = threading.Thread(target=run_calculation)
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
    try:
        global latest_results
        
        if latest_results['status'] == 'not_run':
            return jsonify({'status': 'not_run', 'message': 'Calculation not started'})
        
        if latest_results['status'] == 'running':
            return jsonify({'status': 'running', 'message': 'Calculation in progress...'})
        
        if latest_results['status'] == 'error':
            return jsonify({
                'status': 'error', 
                'message': latest_results.get('error', 'Unknown error occurred')
            })
        
        if latest_results['status'] == 'complete' and latest_results['summary_data']:
            response_data = {
                'status': 'success',
                'summary': {
                    'Capital_Charges': latest_results['capital_charges'],
                    'Calculation_Date': latest_results.get('timestamp'),
                    'Total_Capital': latest_results['capital_charges'].get('TOTAL', 0)
                },
                'risk_breakdown': latest_results['risk_breakdown'],
                'bucket_analysis': latest_results['bucket_analysis'],
                'has_risk_breakdown': True,
                'has_bucket_analysis': True
            }
            return jsonify(response_data)
        
        return jsonify({'status': 'running', 'message': 'Processing results...'})
        
    except Exception as e:
        logger.error(f"Error getting results: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)})


@frtbsa_bp.route('/download/<filename>')
def download_file(filename):
    """Download generated output files"""
    try:
        output_dir = Path('frtbsa_output')
        file_path = output_dir / filename
        
        if file_path.exists():
            from flask import send_file
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({'error': 'File not found'}), 404
            
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
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
        logging.FileHandler(f'frtbsa_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FRTBSARunner:
    """
    Main runner class for FRTB-SA calculations
    """

    def __init__(self, config_file: str = None):
        """
        Initialize runner with optional configuration file
        """
        self.engine = FRTBSAEngine()
        self.validator = DataValidator()
        self.crif_formatter = CRIFFormatter()
        self.aggregator = RiskAggregator()

        # Load configuration if provided
        if config_file:
            self.config = self._load_config(config_file)
        else:
            self.config = self._default_config()

        # Setup output directory
        self.output_dir = Path(self.config.get('output_dir', 'frtbsa_output'))
        self.output_dir.mkdir(exist_ok=True)

        logger.info(f"FRTB-SA Runner initialized. Output directory: {self.output_dir}")

    def _default_config(self) -> dict:
        """Default configuration"""
        return {
            'output_dir': 'frtbsa_output',
            'generate_crif': True,
            'validate_data': True,
            'generate_detailed_reports': True,
            'chunk_size': 10000,
            'valuation_date': datetime.now().strftime('%Y-%m-%d')
        }

    def _load_config(self, config_file: str) -> dict:
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {config_file}")
            return config
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            return self._default_config()

    def run(self, input_file: str, output_prefix: str = None):
        """
        Run complete FRTB-SA calculation pipeline
        """
        global latest_results
        print("\n" + "="*80)
        print("FRTB-SA CAPITAL CALCULATION PIPELINE")
        print("="*80)

        start_time = datetime.now()

        if output_prefix is None:
            output_prefix = f"FRTBSA_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        try:
            # Step 1: Load and validate data
            print("\n[Step 1/6] Loading and validating data...")
            data = self._load_and_validate_data(input_file)

            if data is None:
                logger.error("Data validation failed. Exiting.")
                return False

            # Step 2: Generate CRIF format (optional)
            if self.config.get('generate_crif', True):
                print("\n[Step 2/6] Generating CRIF format...")
                self._generate_crif(data, output_prefix)
            else:
                print("\n[Step 2/6] Skipping CRIF generation (disabled in config)")

            # Step 3: Calculate capital charges
            print("\n[Step 3/6] Calculating FRTB-SA capital charges...")
            capital_charges = self.engine.calculate_capital_charges(data)

            # Step 4: Generate QIS report
            print("\n[Step 4/6] Generating QIS report...")
            qis_file = self.output_dir / f"{output_prefix}_QIS.xlsx"
            self.engine.generate_qis_report(capital_charges, str(qis_file))

            # Step 5: Generate detailed reports
            if self.config.get('generate_detailed_reports', True):
                print("\n[Step 5/6] Generating detailed reports...")
                self._generate_detailed_reports(data, capital_charges, output_prefix)
            else:
                print("\n[Step 5/6] Skipping detailed reports (disabled in config)")

            # Step 6: Generate summary
            print("\n[Step 6/6] Generating summary...")
            self._generate_summary(capital_charges, output_prefix)

            # Calculate execution time
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()

            # Print results
            self._print_results(capital_charges, execution_time)
            # Store results globally for API access
            
            latest_results['status'] = 'complete'
            latest_results['capital_charges'] = capital_charges
            latest_results['summary_data'] = self.get_results_dict(capital_charges, data)
            latest_results['risk_breakdown'] = self.get_results_dict(capital_charges, data)['risk_breakdown']
            latest_results['bucket_analysis'] = self.get_results_dict(capital_charges, data)['bucket_analysis']
            latest_results['timestamp'] = datetime.now().isoformat()
            
            return True

        except Exception as e:
            logger.error(f"Error during FRTB-SA calculation: {e}", exc_info=True)
            
            latest_results['status'] = 'error'
            latest_results['error'] = str(e)
            return False


    def _load_and_validate_data(self, input_file: str) -> pd.DataFrame:
        """Load and validate input data"""

        # Load data
        try:
            data = self.engine.load_data(input_file)
            logger.info(f"Loaded {len(data)} records from {input_file}")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None

        # Validate if enabled
        if self.config.get('validate_data', True):
            is_valid, cleaned_data, validation_report = self.validator.validate(data)

            # Save validation report
            validation_file = self.output_dir / "validation_report.json"
            with open(validation_file, 'w') as f:
                json.dump(validation_report, f, indent=2)

            if not is_valid:
                logger.error(f"Data validation failed: {validation_report['errors']}")
                return None

            if validation_report['warnings']:
                logger.warning(f"Data validation warnings: {validation_report['warnings']}")

            return cleaned_data
        else:
            return data

    def _generate_crif(self, data: pd.DataFrame, output_prefix: str):
        """Generate CRIF format output"""

        crif_df = self.crif_formatter.to_crif(
            data,
            valuation_date=self.config.get('valuation_date')
        )

        # Save CRIF file
        crif_file = self.output_dir / f"{output_prefix}_CRIF.txt"
        crif_df.to_csv(crif_file, sep='\t', index=False)
        logger.info(f"CRIF file saved: {crif_file}")

    def _generate_detailed_reports(self, data: pd.DataFrame,
                                  capital_charges: dict, output_prefix: str):
        """Generate detailed analysis reports"""

        # Risk class breakdown
        risk_breakdown = []
        for risk_class in data['RiskClass'].unique():
            class_data = data[data['RiskClass'] == risk_class]
            breakdown = {
                'Risk Class': risk_class,
                'Trade Count': len(class_data['Trade_ID'].unique()) if 'Trade_ID' in class_data.columns else len(class_data),
                'Total Sensitivity': class_data['FS Amount USD'].sum() if 'FS Amount USD' in class_data.columns else 0,
                'Capital Charge': capital_charges.get(risk_class.replace(' ', '_').upper(), 0)
            }
            risk_breakdown.append(breakdown)

        risk_df = pd.DataFrame(risk_breakdown)
        risk_file = self.output_dir / f"{output_prefix}_Risk_Breakdown.csv"
        risk_df.to_csv(risk_file, index=False)

        # Bucket analysis
        bucket_analysis = []
        for risk_class in data['RiskClass'].unique():
            class_data = data[data['RiskClass'] == risk_class]
            if 'Bucket' in class_data.columns:
                for bucket in class_data['Bucket'].unique():
                    bucket_data = class_data[class_data['Bucket'] == bucket]
                    analysis = {
                        'Risk Class': risk_class,
                        'Bucket': bucket,
                        'Count': len(bucket_data),
                        'Total Sensitivity': bucket_data['FS Amount USD'].sum() if 'FS Amount USD' in bucket_data.columns else 0
                    }
                    bucket_analysis.append(analysis)

        if bucket_analysis:
            bucket_df = pd.DataFrame(bucket_analysis)
            bucket_file = self.output_dir / f"{output_prefix}_Bucket_Analysis.csv"
            bucket_df.to_csv(bucket_file, index=False)

        logger.info("Detailed reports generated")

    def _generate_summary(self, capital_charges: dict, output_prefix: str):
        """Generate executive summary"""

        summary = {
            'Calculation Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Valuation Date': self.config.get('valuation_date'),
            'Capital Charges': capital_charges,
            'Total Capital': capital_charges.get('TOTAL', 0),
            'Top Risk Classes': sorted(
                [(k, v) for k, v in capital_charges.items() if k != 'TOTAL'],
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }

        # Save summary
        summary_file = self.output_dir / f"{output_prefix}_Summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        # Generate Excel summary
        summary_df = pd.DataFrame([
            {'Metric': k, 'Value': v}
            for k, v in capital_charges.items()
        ])
        excel_file = self.output_dir / f"{output_prefix}_Summary.xlsx"
        summary_df.to_excel(excel_file, index=False)

        logger.info("Summary reports generated")

    def _print_results(self, capital_charges: dict, execution_time: float):
        """Print calculation results"""

        print("\n" + "="*80)
        print("CALCULATION RESULTS")
        print("="*80)

        # Capital charges
        print("\nCapital Charges by Risk Class:")
        print("-"*50)

        for risk_class, charge in sorted(capital_charges.items()):
            if risk_class != 'TOTAL':
                print(f"{risk_class:.<40} {charge:>15,.2f}")

        print("-"*50)
        print(f"{'TOTAL CAPITAL CHARGE':.<40} {capital_charges.get('TOTAL', 0):>15,.2f}")

        # Output files
        print("\n" + "="*80)
        print("OUTPUT FILES")
        print("="*80)

        for file in sorted(self.output_dir.glob("*")):
            if file.is_file():
                size = file.stat().st_size / 1024  # KB
                print(f"✓ {file.name:<50} ({size:.1f} KB)")

        print(f"\nAll files saved to: {self.output_dir.absolute()}")

        # Execution time
        print(f"\nExecution time: {execution_time:.2f} seconds")
        print("="*80)
    
    def get_results_dict(self, capital_charges: dict, data: pd.DataFrame) -> dict:
        """Convert calculation results to dictionary for API response"""
        
        # Risk breakdown
        risk_breakdown = []
        for risk_class in data['RiskClass'].unique():
            class_data = data[data['RiskClass'] == risk_class]
            breakdown = {
                'Risk Class': risk_class,
                'Trade Count': len(class_data['Trade_ID'].unique()) if 'Trade_ID' in class_data.columns else len(class_data),
                'Total Sensitivity': float(class_data['FS Amount USD'].sum() if 'FS Amount USD' in class_data.columns else 0),
                'Capital Charge': float(capital_charges.get(risk_class.replace(' ', '_').upper(), 0))
            }
            risk_breakdown.append(breakdown)
        
        # Bucket analysis
        bucket_analysis = []
        for risk_class in data['RiskClass'].unique():
            class_data = data[data['RiskClass'] == risk_class]
            if 'Bucket' in class_data.columns:
                for bucket in class_data['Bucket'].unique():
                    bucket_data = class_data[class_data['Bucket'] == bucket]
                    analysis = {
                        'Risk Class': risk_class,
                        'Bucket': str(bucket),
                        'Count': int(len(bucket_data)),
                        'Total Sensitivity': float(bucket_data['FS Amount USD'].sum() if 'FS Amount USD' in bucket_data.columns else 0)
                    }
                    bucket_analysis.append(analysis)
        
        # Convert all numpy types to native Python types
        capital_charges_clean = {}
        for k, v in capital_charges.items():
            capital_charges_clean[k] = float(v) if isinstance(v, (np.integer, np.floating)) else v
        
        return {
            'capital_charges': capital_charges_clean,
            'risk_breakdown': risk_breakdown,
            'bucket_analysis': bucket_analysis,
            'calculation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_records': len(data)
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
