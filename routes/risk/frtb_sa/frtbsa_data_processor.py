"""
FRTB-SA Data Processor and CRIF Format Handler
Handles data validation, transformation, and CRIF format conversion
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class CRIFFormatter:
    """
    Common Risk Interchange Format (CRIF) handler for FRTB-SA
    Based on ISDA SIMM/FRTB standards
    """

    # CRIF column definitions
    CRIF_COLUMNS = [
        'ValuationDate',
        'TradeID',
        'PortfolioID',
        'ProductClass',
        'RiskType',
        'Qualifier',
        'Bucket',
        'Label1',
        'Label2',
        'Amount',
        'AmountCurrency',
        'AmountUSD',
        'PostRegulations',
        'EndDate'
    ]

    # Risk type mappings for CRIF
    RISK_TYPE_MAP = {
        'GIRR': 'Risk_IRCurve',
        'CSR_NON_SEC': 'Risk_CreditQ',
        'CSR_SEC': 'Risk_CreditQ_Sec',
        'EQUITY': 'Risk_Equity',
        'FX': 'Risk_FX',
        'COMMODITY': 'Risk_Commodity'
    }

    # Product class mappings
    PRODUCT_CLASS_MAP = {
        'GIRR': 'RatesFX',
        'CSR': 'Credit',
        'EQUITY': 'Equity',
        'FX': 'RatesFX',
        'COMMODITY': 'Commodity'
    }

    def __init__(self):
        self.validation_errors = []

    def to_crif(self, df: pd.DataFrame, valuation_date: str = None) -> pd.DataFrame:
        """
        Convert FRTB-SA data to CRIF format
        """
        logger.info("Converting data to CRIF format")

        if valuation_date is None:
            valuation_date = datetime.now().strftime('%Y-%m-%d')

        crif_data = []

        for _, row in df.iterrows():
            crif_row = self._convert_row_to_crif(row, valuation_date)
            if crif_row:
                crif_data.append(crif_row)

        crif_df = pd.DataFrame(crif_data, columns=self.CRIF_COLUMNS)

        logger.info(f"Created CRIF with {len(crif_df)} records")
        return crif_df

    def _convert_row_to_crif(self, row: pd.Series, valuation_date: str) -> Dict:
        """Convert single row to CRIF format"""

        risk_class = row.get('RiskClass', '')

        # Map to CRIF risk type
        if 'GIRR' in risk_class or 'General Interest' in risk_class:
            risk_type = 'Risk_IRCurve'
            product_class = 'RatesFX'
        elif 'Credit Spread' in risk_class or 'CSR' in risk_class:
            risk_type = 'Risk_CreditQ'
            product_class = 'Credit'
        elif 'Equity' in risk_class or 'EQ' in risk_class:
            risk_type = 'Risk_Equity'
            product_class = 'Equity'
        elif 'FX' in risk_class or 'Foreign Exchange' in risk_class:
            risk_type = 'Risk_FX'
            product_class = 'RatesFX'
        elif 'Commodity' in risk_class or 'COMM' in risk_class:
            risk_type = 'Risk_Commodity'
            product_class = 'Commodity'
        else:
            risk_type = 'Risk_Other'
            product_class = 'Other'

        crif_row = {
            'ValuationDate': valuation_date,
            'TradeID': row.get('Trade_ID', ''),
            'PortfolioID': row.get('Book_ID', ''),
            'ProductClass': product_class,
            'RiskType': risk_type,
            'Qualifier': row.get('Qualifier', ''),
            'Bucket': str(row.get('Bucket', '')),
            'Label1': row.get('Label1', ''),
            'Label2': row.get('Label2', ''),
            'Amount': row.get('FS Amount', 0),
            'AmountCurrency': row.get('Amount Currency', 'USD'),
            'AmountUSD': row.get('FS Amount USD', 0),
            'PostRegulations': 'FRTB,SA-CCR',
            'EndDate': ''
        }

        return crif_row

    def from_crif(self, crif_file: str) -> pd.DataFrame:
        """
        Read CRIF format file and convert to FRTB-SA format
        """
        logger.info(f"Reading CRIF file: {crif_file}")

        # Read tab-separated CRIF file
        df = pd.read_csv(crif_file, sep='\t')

        # Validate CRIF structure
        missing_cols = set(self.CRIF_COLUMNS) - set(df.columns)
        if missing_cols:
            logger.warning(f"Missing CRIF columns: {missing_cols}")

        # Convert to FRTB-SA format
        frtbsa_data = self._convert_crif_to_frtbsa(df)

        return frtbsa_data

    def _convert_crif_to_frtbsa(self, crif_df: pd.DataFrame) -> pd.DataFrame:
        """Convert CRIF format to FRTB-SA format"""

        frtbsa_data = []

        for _, row in crif_df.iterrows():
            frtbsa_row = {
                'Risk Date': row.get('ValuationDate', ''),
                'Trade_ID': row.get('TradeID', ''),
                'Book_ID': row.get('PortfolioID', ''),
                'RiskClass': self._map_product_class(row.get('ProductClass', '')),
                'Risk_Type': row.get('RiskType', ''),
                'Qualifier': row.get('Qualifier', ''),
                'Bucket': row.get('Bucket', ''),
                'Label1': row.get('Label1', ''),
                'Label2': row.get('Label2', ''),
                'FS Amount': row.get('Amount', 0),
                'Amount Currency': row.get('AmountCurrency', 'USD'),
                'FS Amount USD': row.get('AmountUSD', 0)
            }
            frtbsa_data.append(frtbsa_row)

        return pd.DataFrame(frtbsa_data)

    def _map_product_class(self, product_class: str) -> str:
        """Map CRIF product class to FRTB-SA risk class"""

        mapping = {
            'RatesFX': 'GIRR',
            'Credit': 'Credit Spread (Non-secur)',
            'Equity': 'Equity',
            'Commodity': 'Commodity'
        }
        return mapping.get(product_class, product_class)

    def validate_crif(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate CRIF format data
        Returns: (is_valid, list_of_errors)
        """
        errors = []

        # Check required columns
        missing_cols = set(self.CRIF_COLUMNS) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")

        # Validate data types
        if 'AmountUSD' in df.columns:
            non_numeric = df[~df['AmountUSD'].apply(lambda x: isinstance(x, (int, float)))]
            if len(non_numeric) > 0:
                errors.append(f"Non-numeric values in AmountUSD: {len(non_numeric)} rows")

        # Validate risk types
        valid_risk_types = ['Risk_IRCurve', 'Risk_FX', 'Risk_CreditQ',
                           'Risk_CreditQ_Sec', 'Risk_Equity', 'Risk_Commodity']
        if 'RiskType' in df.columns:
            invalid_types = df[~df['RiskType'].isin(valid_risk_types)]
            if len(invalid_types) > 0:
                errors.append(f"Invalid risk types: {invalid_types['RiskType'].unique()}")

        is_valid = len(errors) == 0
        return is_valid, errors

class DataValidator:
    """
    Validate FRTB-SA input data according to BCBS-239 standards
    """

    def __init__(self):
        self.validation_rules = self._define_validation_rules()
        self.validation_report = []

    def _define_validation_rules(self) -> Dict:
        """Define validation rules for each field"""

        return {
            'mandatory_fields': [
                'RiskClass', 'Risk_Type', 'Bucket', 'FS Amount USD'
            ],
            'risk_classes': [
                'GIRR', 'General Interest Rate Risk',
                'Credit Spread (Non-secur)', 'CSR_NON_SEC',
                'Credit Spread (Securitisation)', 'CSR_SEC',
                'Equity', 'Commodity', 'FX', 'Foreign Exchange'
            ],
            'numeric_fields': [
                'FS Amount', 'FS Amount USD', 'Trade Notional',
                'PV0 /MTM', 'PnL_Up Delta [ + RW]', 'PnL_Down Delta [- RW]'
            ],
            'date_fields': ['Risk Date'],
            'tenor_buckets': [
                '0.25Y', '0.5Y', '1Y', '2Y', '3Y', '5Y',
                '10Y', '15Y', '20Y', '30Y'
            ],
            'credit_quality': ['IG', 'HY', 'NR'],
            'long_short': ['Long', 'Short']
        }

    def validate(self, df: pd.DataFrame) -> Tuple[bool, pd.DataFrame, List[str]]:
        """
        Validate FRTB-SA data
        Returns: (is_valid, cleaned_data, validation_errors)
        """

        logger.info("Starting data validation")

        errors = []
        warnings = []
        cleaned_df = df.copy()

        # Check mandatory fields
        missing_fields = [f for f in self.validation_rules['mandatory_fields']
                         if f not in df.columns]
        if missing_fields:
            errors.append(f"Missing mandatory fields: {missing_fields}")

        # Validate risk classes
        if 'RiskClass' in df.columns:
            invalid_classes = df[~df['RiskClass'].isin(self.validation_rules['risk_classes'])]
            if len(invalid_classes) > 0:
                warnings.append(f"Unknown risk classes found: {invalid_classes['RiskClass'].unique()}")

        # Validate numeric fields
        for field in self.validation_rules['numeric_fields']:
            if field in df.columns:
                cleaned_df[field] = self._clean_numeric_field(df[field])

        # Validate dates
        for field in self.validation_rules['date_fields']:
            if field in df.columns:
                cleaned_df[field] = pd.to_datetime(df[field], errors='coerce')
                invalid_dates = cleaned_df[cleaned_df[field].isna()]
                if len(invalid_dates) > 0:
                    warnings.append(f"Invalid dates in {field}: {len(invalid_dates)} rows")

        # Validate tenors for GIRR
        if 'Label1' in df.columns:
            girr_data = df[df['RiskClass'].str.contains('GIRR|Interest', case=False, na=False)]
            if len(girr_data) > 0:
                invalid_tenors = girr_data[~girr_data['Label1'].isin(self.validation_rules['tenor_buckets'])]
                if len(invalid_tenors) > 0:
                    warnings.append(f"Invalid GIRR tenors: {invalid_tenors['Label1'].unique()}")

        # Check for duplicates
        if 'Trade_ID' in df.columns:
            duplicates = df[df.duplicated(subset=['Trade_ID'], keep=False)]
            if len(duplicates) > 0:
                warnings.append(f"Duplicate Trade IDs found: {len(duplicates)} rows")

        # Generate validation report
        self.validation_report = {
            'total_records': len(df),
            'valid_records': len(cleaned_df),
            'errors': errors,
            'warnings': warnings,
            'validation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        is_valid = len(errors) == 0

        logger.info(f"Validation complete. Valid: {is_valid}, Errors: {len(errors)}, Warnings: {len(warnings)}")

        return is_valid, cleaned_df, self.validation_report

    def _clean_numeric_field(self, series: pd.Series) -> pd.Series:
        """Clean numeric field by removing formatting"""

        # Convert to string, remove commas, parentheses (negatives), and spaces
        cleaned = series.astype(str).str.replace(',', '').str.replace(' ', '')
        cleaned = cleaned.str.replace('(', '-').str.replace(')', '')

        # Convert to numeric, filling NaN with 0
        return pd.to_numeric(cleaned, errors='coerce').fillna(0)

    def generate_validation_report(self, output_path: str):
        """Generate detailed validation report"""

        with open(output_path, 'w') as f:
            json.dump(self.validation_report, f, indent=2)

        logger.info(f"Validation report saved to {output_path}")

class RiskAggregator:
    """
    Advanced risk aggregation with correlation handling
    """

    def __init__(self):
        self.correlation_matrices = self._define_correlations()

    def _define_correlations(self) -> Dict:
        """Define correlation matrices for each risk class"""

        return {
            'GIRR': {
                'same_currency': {
                    'same_tenor': 1.0,
                    'adjacent_tenor': 0.99,
                    'distant_tenor': 0.60
                },
                'different_currency': 0.50
            },
            'CSR': {
                'same_issuer': 1.0,
                'same_sector': 0.35,
                'different_sector': 0.15
            },
            'EQUITY': {
                'same_bucket': 0.15,
                'different_bucket': 0.0,
                'index_correlation': 0.80
            },
            'FX': 0.60,
            'COMMODITY': {
                'same_type': 0.95,
                'related_type': 0.40,
                'different_type': 0.15
            }
        }

    def aggregate_with_correlation(self, sensitivities: pd.DataFrame,
                                  risk_class: str) -> float:
        """
        Aggregate sensitivities with correlation
        """

        if len(sensitivities) == 0:
            return 0.0

        # Get correlation parameters
        corr_params = self.correlation_matrices.get(risk_class, {})

        # Build correlation matrix
        n = len(sensitivities)
        corr_matrix = np.eye(n)

        if risk_class == 'GIRR':
            # GIRR correlation based on tenor distance
            tenors = sensitivities['Label1'].values
            for i in range(n):
                for j in range(i+1, n):
                    if sensitivities.iloc[i]['Bucket'] == sensitivities.iloc[j]['Bucket']:
                        # Same currency
                        tenor_distance = self._get_tenor_distance(tenors[i], tenors[j])
                        if tenor_distance == 0:
                            corr = corr_params['same_currency']['same_tenor']
                        elif tenor_distance == 1:
                            corr = corr_params['same_currency']['adjacent_tenor']
                        else:
                            corr = corr_params['same_currency']['distant_tenor']
                    else:
                        # Different currency
                        corr = corr_params['different_currency']

                    corr_matrix[i, j] = corr
                    corr_matrix[j, i] = corr

        elif risk_class == 'EQUITY':
            # Equity correlation based on bucket
            buckets = sensitivities['Bucket'].values
            for i in range(n):
                for j in range(i+1, n):
                    if buckets[i] == buckets[j]:
                        corr = corr_params['same_bucket']
                    else:
                        corr = corr_params['different_bucket']
                    corr_matrix[i, j] = corr
                    corr_matrix[j, i] = corr

        # Calculate aggregated capital
        ws = sensitivities['weighted_sensitivity'].values
        capital = np.sqrt(np.dot(ws, np.dot(corr_matrix, ws)))

        return capital

    def _get_tenor_distance(self, tenor1: str, tenor2: str) -> int:
        """Calculate distance between tenors"""

        tenor_order = ['0.25Y', '0.5Y', '1Y', '2Y', '3Y', '5Y', '10Y', '15Y', '20Y', '30Y']

        try:
            idx1 = tenor_order.index(tenor1)
            idx2 = tenor_order.index(tenor2)
            return abs(idx1 - idx2)
        except ValueError:
            return 10  # Large distance for unknown tenors

class StreamingProcessor:
    """
    Process large FRTB-SA files in streaming mode
    """

    def __init__(self, chunk_size: int = 10000):
        self.chunk_size = chunk_size

    def process_large_file(self, input_file: str, output_file: str,
                          processor_func: callable):
        """
        Process large file in chunks
        """

        logger.info(f"Processing large file: {input_file}")

        # Determine file type
        if input_file.endswith('.csv'):
            reader = pd.read_csv(input_file, chunksize=self.chunk_size)
        else:
            # For Excel, read in chunks manually
            reader = self._read_excel_chunks(input_file)

        results = []
        total_rows = 0

        for chunk_num, chunk in enumerate(reader, 1):
            logger.info(f"Processing chunk {chunk_num} ({len(chunk)} rows)")

            # Process chunk
            chunk_result = processor_func(chunk)
            results.append(chunk_result)

            total_rows += len(chunk)

        logger.info(f"Processed {total_rows} total rows")

        # Combine results
        final_result = pd.concat(results, ignore_index=True)

        # Save output
        if output_file.endswith('.csv'):
            final_result.to_csv(output_file, index=False)
        else:
            final_result.to_excel(output_file, index=False)

        logger.info(f"Results saved to {output_file}")

        return final_result

    def _read_excel_chunks(self, file_path: str):
        """Read Excel file in chunks"""

        # First, get total rows
        df_temp = pd.read_excel(file_path, nrows=1)
        total_rows = len(pd.read_excel(file_path))

        # Read in chunks
        for start_row in range(0, total_rows, self.chunk_size):
            chunk = pd.read_excel(file_path,
                                skiprows=range(1, start_row+1),
                                nrows=self.chunk_size)
            yield chunk

# Example usage
def example_usage():
    """
    Example of using the data processor modules
    """

    # Initialize components
    crif_formatter = CRIFFormatter()
    validator = DataValidator()
    aggregator = RiskAggregator()

    # Load sample data
    df = pd.read_excel('FRTBSA data.xlsx')

    # Validate data
    is_valid, cleaned_df, validation_report = validator.validate(df)

    if is_valid:
        print("✓ Data validation passed")

        # Convert to CRIF format
        crif_df = crif_formatter.to_crif(cleaned_df)

        # Save CRIF file
        crif_df.to_csv('FRTBSA_CRIF_Output.txt', sep='\t', index=False)
        print("✓ CRIF file generated")

    else:
        print("✗ Data validation failed:")
        for error in validation_report['errors']:
            print(f"  - {error}")

    # Generate validation report
    validator.generate_validation_report('validation_report.json')

if __name__ == "__main__":
    example_usage()
