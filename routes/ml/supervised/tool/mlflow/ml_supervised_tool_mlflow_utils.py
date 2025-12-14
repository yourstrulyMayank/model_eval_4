import os
import json
import pickle
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import mlflow
import mlflow.sklearn
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, mean_absolute_percentage_error
)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from .create_plots_ml_supervised import (
    generate_model_summary_plots, create_regression_plots, create_classification_plots)
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot

# Set style for better plots
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    plt.style.use('seaborn') 
sns.set_palette("husl")

# Global variables for progress tracking
ml_evaluation_progress = {}
ml_evaluation_results = {}

def update_progress(model_name, current_task, progress_percent):
    ml_evaluation_progress[model_name] = {
        "current_task": current_task,
        "progress_percent": progress_percent,
        "timestamp": datetime.now().isoformat()
    }

def get_ml_progress(model_name):
    return ml_evaluation_progress.get(model_name, {
        "current_task": "Not started",
        "progress_percent": 0,
        "timestamp": datetime.now().isoformat()
    })

def clear_ml_progress(model_name):
    if model_name in ml_evaluation_progress:
        del ml_evaluation_progress[model_name]
    if model_name in ml_evaluation_results:
        del ml_evaluation_results[model_name]

def load_model(model_path):
    try:
        with open(model_path, 'rb') as f:
            model = joblib.load(f)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def detect_problem_type(model):
    model_type = type(model).__name__.lower()
    regression_keywords = ['regression', 'regressor', 'linear', 'ridge', 'lasso', 'elastic']
    classification_keywords = ['classifier', 'logistic', 'svm', 'randomforest', 'decisiontree', 
                               'gradient', 'xgb', 'lgb', 'naive', 'knn', 'ada', 'extra', 'voting']
    for keyword in regression_keywords:
        if keyword in model_type:
            return 'regression'
    for keyword in classification_keywords:
        if keyword in model_type:
            return 'classification'
    return 'regression'


def round_if_needed(preds, model_name):
    if any(keyword in model_name.lower() for keyword in ['rating', 'ordinal', 'score', 'rank']):
        return np.clip(np.round(preds).astype(int), 0, 10)
    return preds

def get_model_info(model):
    info = {
        'model_type': type(model).__name__,
        'model_params': getattr(model, 'get_params', lambda: {})(),
        'feature_count': None,
        'has_feature_importance': hasattr(model, 'feature_importances_'),
        'has_predict_proba': hasattr(model, 'predict_proba'),
        'has_coefficients': hasattr(model, 'coef_'),
        'training_score': None
    }
    if hasattr(model, 'n_features_in_'):
        info['feature_count'] = model.n_features_in_
    elif hasattr(model, 'coef_'):
        info['feature_count'] = model.coef_.shape[-1] if model.coef_.ndim > 1 else len(model.coef_)
    if hasattr(model, 'score_'):
        info['training_score'] = model.score_
    return info

def cleanup_evaluation_resources(model_name):
    try:
        import matplotlib.pyplot as plt
        plt.close('all')
        import gc
        gc.collect()
    except Exception as e:
        print(f"Error during cleanup: {e}")


def calculate_detailed_metrics(y_true, y_pred, problem_type, model=None, X_test=None):
    metrics = {}
    if problem_type == 'regression':
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['r2'] = r2_score(y_true, y_pred)
        try:
            metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)
        except:
            metrics['mape'] = None
        metrics['max_error'] = np.max(np.abs(y_true - y_pred))
        metrics['mean_error'] = np.mean(y_true - y_pred)
    else:
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        unique_classes = np.unique(y_true)
        if len(unique_classes) == 2 and model is not None and hasattr(model, 'predict_proba') and X_test is not None:
            try:
                y_prob = model.predict_proba(X_test)[:, 1]
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            except Exception:
                metrics['roc_auc'] = None
    return metrics

def calculate_data_validation_stats(df):
    """Calculate validation stats for the dataset."""
    stats = {
        'total_records': len(df),
        'missing_values': round(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100, 2),
        'duplicate_records': round(df.duplicated().sum() / len(df) * 100, 2),
        'data_completeness': f"{round((1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100, 1)}%",
        'feature_count': df.shape[1]
    }
    # Simple outlier detection (Z-score > 3) for numerical cols
    try:
        numeric_df = df.select_dtypes(include=[np.number])
        z_scores = np.abs(stats.zscore(numeric_df))
        outliers = (z_scores > 3).sum().sum()
        stats['outliers_detected'] = round(outliers / (df.shape[0] * df.shape[1]) * 100, 2)
    except:
        stats['outliers_detected'] = 0
    
    return stats

def run_ml_evaluation(model_name, model_path, dataset_path, new_version=False):
    """Run comprehensive ML model evaluation using MLflow."""
    try:
        update_progress(model_name, "Initializing...", 5)
        mlflow.set_experiment(f"ML_Evaluation_{model_name}")
        
        with mlflow.start_run(run_name=f"{model_name}_evaluation"):
            update_progress(model_name, "Loading model...", 15)
            model = load_model(model_path)
            if model is None: raise Exception("Failed to load model")
            
            update_progress(model_name, "Loading test data...", 25)
            if not os.path.exists(dataset_path): raise Exception(f"Test dataset not found: {dataset_path}")
            
            df = pd.read_csv(dataset_path)
            # Calculate validation stats
            data_stats = calculate_data_validation_stats(df)
            
            if 'target' not in df.columns:
                target_col = df.columns[-1]
                df = df.rename(columns={target_col: 'target'})
            
            X_test = df.drop(columns=['target'])
            y_test = df['target'].values
            
            update_progress(model_name, "Analyzing model...", 35)
            model_info = get_model_info(model)
            problem_type = detect_problem_type(model)
            
            update_progress(model_name, "Making predictions...", 45)
            y_pred = model.predict(X_test)
            y_pred = round_if_needed(y_pred, model_name)
            
            update_progress(model_name, "Calculating metrics...", 55)
            metrics = calculate_detailed_metrics(y_test, y_pred, problem_type, model, X_test)
            
            update_progress(model_name, "Generating visualizations...", 70)
            plots_dir = f"static/plots/{model_name}"
            os.makedirs(plots_dir, exist_ok=True)
            
            summary_plots = generate_model_summary_plots(model_name, model, model_info)
            if problem_type == 'regression':
                eval_plots = create_regression_plots(model_name, y_test, y_pred, plots_dir)
            else:
                eval_plots = create_classification_plots(model_name, model, X_test, y_test, y_pred, plots_dir)
            
            plt.close('all')
            all_plots = {**summary_plots, **eval_plots}
            
            # Save predictions
            results_df = pd.DataFrame({'actual': y_test, 'predicted': y_pred})
            results_path = f"{plots_dir}/predictions.csv"
            results_df.to_csv(results_path, index=False)
            
            results = {
                'model_name': model_name,
                'timestamp': datetime.now().isoformat(),
                'problem_type': problem_type,
                'model_info': model_info,
                'metrics': metrics,
                'plots': all_plots,
                'mlflow_run_id': mlflow.active_run().info.run_id,
                'model_path': model_path,
                'dataset_path': dataset_path,
                'num_test_samples': len(y_test),
                'num_features': X_test.shape[1],
                'data_validation': data_stats
            }
            
            ml_evaluation_results[model_name] = results
            
            # Update details.json
            update_model_details_json(model_name, results, new_version=new_version)
            
            update_progress(model_name, "Evaluation completed!", 100)
            return results
            
    except Exception as e:
        error_msg = str(e)
        update_progress(model_name, f"Error: {error_msg}", 0)
        raise e
    finally:
        cleanup_evaluation_resources(model_name)


def get_ml_results(model_name):
    return ml_evaluation_results.get(model_name, {})

def list_available_results():
    return list(ml_evaluation_results.keys())

def cleanup_ml_resources(model_name=None):
    """Clean up resources for a specific model or all models."""
    if model_name:
        # Clean up specific model resources
        plots_dir = f"static/plots/{model_name}"
        if os.path.exists(plots_dir):
            import shutil
            shutil.rmtree(plots_dir)
        clear_ml_progress(model_name)
    else:
        # Clean up all resources
        if os.path.exists("static/plots"):
            import shutil
            shutil.rmtree("static/plots")
        ml_evaluation_progress.clear()
        ml_evaluation_results.clear()


def convert_numpy_types(obj):
    import numpy as np
    if isinstance(obj, dict): return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list): return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, np.integer): return int(obj)
    elif isinstance(obj, np.floating): return float(obj)
    elif isinstance(obj, np.ndarray): return obj.tolist()
    return obj


def export_results_to_json(model_name, output_path=None):
    """Export evaluation results to JSON file with better error handling."""
    if model_name not in ml_evaluation_results:
        raise ValueError(f"No results found for model: {model_name}")
    
    try:
        results = ml_evaluation_results[model_name].copy()
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        results = convert_numpy_types(results)
        
        # Always use a predictable filename for download
        if output_path is None:
            output_dir = f"results/{model_name}"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"ml_evaluation_{model_name}.json")
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return output_path
        
    except Exception as e:
        print(f"Error exporting results to JSON: {e}")
        raise e

def compare_models(model_names, metric='accuracy'):
    """Compare multiple models based on a specific metric."""
    comparison_data = []
    
    for model_name in model_names:
        if model_name in ml_evaluation_results:
            results = ml_evaluation_results[model_name]
            if 'metrics' in results and metric in results['metrics']:
                comparison_data.append({
                    'model_name': model_name,
                    'metric_value': results['metrics'][metric],
                    'problem_type': results.get('problem_type', 'unknown'),
                    'timestamp': results.get('timestamp', '')
                })
    
    # Sort by metric value (descending for most metrics)
    comparison_data.sort(key=lambda x: x['metric_value'], reverse=True)
    
    return comparison_data

# # Main execution function for command line usage
# if __name__ == "__main__":
#     import sys
    
#     if len(sys.argv) != 4:
#         print("Usage: python ml_evaluation.py <model_name> <model_path> <dataset_path>")
#         sys.exit(1)
    
#     model_name = sys.argv[1]
#     model_path = sys.argv[2]
#     dataset_path = sys.argv[3]
    
#     try:
#         print(f"Starting evaluation for model: {model_name}")
#         results = run_ml_evaluation(model_name, model_path, dataset_path)
        
#         print(f"\nEvaluation completed successfully!")
#         print(f"Problem type: {results['problem_type']}")
#         print(f"Model type: {results['model_info']['model_type']}")
#         print(f"Number of features: {results['num_features']}")
#         print(f"Test samples: {results['num_test_samples']}")
        
#         if 'metrics' in results:
#             print(f"\nKey Metrics:")
#             for metric, value in list(results['metrics'].items())[:5]:  # Show first 5 metrics
#                 if value is not None:
#                     print(f"  {metric}: {value:.4f}")
        
#         print(f"\nMLflow Run ID: {results['mlflow_run_id']}")
#         print(f"Plots generated: {len(results['plots'])}")
        
#         # Export results to JSON
#         json_path = export_results_to_json(model_name)
#         print(f"Results exported to: {json_path}")
        
#     except Exception as e:
#         print(f"Evaluation failed: {str(e)}")
#         sys.exit(1)

def run_ml_evaluation_wrapper(model_name, model_file_path, test_csv_path, benchmark):
    try:
        import matplotlib
        matplotlib.use('Agg')
        run_ml_evaluation(model_name, model_file_path, test_csv_path, new_version=False)
    except Exception as e:
        print(f"Error in ML evaluation thread: {e}")
    

def extract_test_csv_if_needed(dataset_path):
    """Extract test.csv from dataset.zip if it doesn't exist."""
    test_csv_path = os.path.join(dataset_path, 'test.csv')
    dataset_zip_path = os.path.join(dataset_path, 'dataset.zip')
    
    # If test.csv already exists, return its path
    if os.path.exists(test_csv_path):
        return test_csv_path
    
    # If dataset.zip exists, try to extract test.csv
    if os.path.exists(dataset_zip_path):
        try:
            import zipfile
            with zipfile.ZipFile(dataset_zip_path, 'r') as zip_ref:
                # Look for test.csv in the zip
                if 'test.csv' in zip_ref.namelist():
                    zip_ref.extract('test.csv', dataset_path)
                    return test_csv_path
                else:
                    # Look for any CSV file that might be the test data
                    csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
                    if csv_files:
                        # Extract the first CSV file and rename it to test.csv
                        zip_ref.extract(csv_files[0], dataset_path)
                        extracted_path = os.path.join(dataset_path, csv_files[0])
                        os.rename(extracted_path, test_csv_path)
                        return test_csv_path
        except Exception as e:
            print(f"Error extracting from dataset.zip: {e}")
    
    return None

def generate_ml_report(results):
    """Generate comprehensive ML evaluation report."""
    report = {
        'model_name': results.get('model_name'),
        'evaluation_timestamp': results.get('timestamp'),
        'problem_type': results.get('problem_type'),
        'dataset_info': results.get('dataset_info'),
        'performance_metrics': results.get('metrics'),
        'mlflow_run_id': results.get('mlflow_run_id'),
        'summary': {
            'evaluation_completed': True,
            'total_samples': results.get('dataset_info', {}).get('n_samples', 0),
            'total_features': results.get('dataset_info', {}).get('n_features', 0)
        }
    }
    
    # Add key performance indicators
    metrics = results.get('metrics', {})
    if results.get('problem_type') == 'classification':
        report['summary']['key_metrics'] = {
            'accuracy': metrics.get('accuracy', 0),
            'precision': metrics.get('precision', 0),
            'recall': metrics.get('recall', 0),
            'f1_score': metrics.get('f1_score', 0)
        }
        if 'roc_auc' in metrics:
            report['summary']['key_metrics']['roc_auc'] = metrics['roc_auc']
    else:
        report['summary']['key_metrics'] = {
            'r2_score': metrics.get('r2_score', 0),
            'rmse': metrics.get('rmse', 0),
            'mae': metrics.get('mae', 0),
            'mape': metrics.get('mean_absolute_percentage_error', 0)
        }
    
    # Add cross-validation summary
    if 'cv_scores' in metrics and metrics['cv_scores']:
        report['cross_validation'] = {
            'mean_score': metrics['cv_scores']['mean'],
            'std_score': metrics['cv_scores']['std'],
            'individual_scores': metrics['cv_scores']['scores']
        }
    
    return json.dumps(report, indent=2, default=str)


def update_model_details_json(model_name, evaluation_results, new_version=False):
    """
    Update details.json.
    - If new_version=True, increment version (e.g., v1.0.0 -> v1.1.0).
    - Else, overwrite current version data.
    """
    try:
        details_path = os.path.join('models', model_name, 'details.json')
        
        if os.path.exists(details_path):
            with open(details_path, 'r') as f:
                details = json.load(f)
        else:
            details = {
                "model_name": model_name,
                "version": "v1.0.0",
                "framework": "Scikit-learn",
                "status": "Active"
            }
        
        # Versioning Logic
        if new_version:
            curr_v = details.get("version", "v1.0.0").lstrip('v').split('.')
            if len(curr_v) >= 2:
                # Increment minor version
                new_v = f"v{curr_v[0]}.{int(curr_v[1]) + 1}.0"
                details['version'] = new_v
                # Add to history/comparison if needed, or simply update current
        
        details['last_updated'] = datetime.now().strftime('%Y-%m-%d')
        
        # Model Info
        model_info = evaluation_results.get('model_info', {})
        details['model_info'] = details.get('model_info', {})
        details['model_info'].update({
            'model_type': model_info.get('model_type', 'Unknown'),
            'algorithm': model_info.get('model_type', 'Unknown'),
            'problem_type': evaluation_results.get('problem_type', 'Unknown').capitalize(),
            'feature_count': model_info.get('feature_count') or evaluation_results.get('num_features', 0)
        })
        
        # Benchmarks
        metrics = evaluation_results.get('metrics', {})
        details['benchmarks'] = details.get('benchmarks', {})
        
        problem_type = evaluation_results.get('problem_type', 'regression')
        if problem_type == 'regression':
            r2 = metrics.get('r2', 0)
            details['benchmarks'].update({
                'overall_score': int(r2 * 100) if r2 else 0,
                'primary_metric': round(r2, 4) if r2 else 0,
                'r2_score': round(r2, 4) if r2 else 0,
                'rmse': round(metrics.get('rmse', 0), 2),
                'mae': round(metrics.get('mae', 0), 2),
                'mape': round(metrics.get('mape', 0), 2) if metrics.get('mape') else 0
            })
        else:
            acc = metrics.get('accuracy', 0)
            details['benchmarks'].update({
                'overall_score': int(acc * 100) if acc else 0,
                'accuracy': int(acc * 100) if acc else 0,
                'primary_metric': round(metrics.get('accuracy', 0), 4),
                'precision': round(metrics.get('precision_weighted', 0), 4),
                'recall': round(metrics.get('recall_weighted', 0), 4),
                'f1_score': round(metrics.get('f1_weighted', 0), 4)
            })
            
        # Data Validation
        if 'data_validation' in evaluation_results:
            details['data_validation'] = evaluation_results['data_validation']
            # Hardcoded Scope updates for Capital Risk as requested
            details['scope'] = details.get('scope', {})
            details['scope'].update({
                'target_variable': 'Capital Risk Score',
                'business_use_case': 'Capital Adequacy & Risk Assessment'
            })

        # Model Parameters
        if model_info.get('model_params'):
            details['model_parameters'] = model_info['model_params']

        # Save
        with open(details_path, 'w') as f:
            json.dump(details, f, indent=2)
            
        return True
    except Exception as e:
        print(f"Error updating details.json: {e}")
        return False
