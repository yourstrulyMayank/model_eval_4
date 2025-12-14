from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, send_file, make_response
from .ml_supervised_tool_mlflow_utils import (
    run_ml_evaluation_wrapper, 
    get_ml_progress, 
    convert_numpy_types, 
    get_ml_results,  
    export_results_to_json,
    clear_ml_progress,
    update_progress,    
    run_ml_evaluation,
    list_available_results,
    generate_ml_report,
    load_model,
    detect_problem_type,
    calculate_detailed_metrics,
    round_if_needed
)
import pandas as pd
import numpy as np
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
import tempfile
import joblib

ml_s_t_mlflow_bp = Blueprint('ml_s_t_mlflow', __name__)

@ml_s_t_mlflow_bp.route('/evaluate_ml/<model_name>/<subcategory>', methods=['POST'])
def evaluate_ml(model_name, subcategory):
    """Evaluate ML models with subcategory support."""
    try:        
        model_path = os.path.join('models', model_name, 'model', 'model.pkl')
        dataset_path = os.path.join('models', model_name, 'dataset', 'test.csv')        
        
        if not os.path.exists(model_path):
            flash(f"Model file not found: {model_path}")
            return redirect(url_for('index'))
        if not os.path.exists(dataset_path):
            flash(f"Test CSV file not found: {dataset_path}")
            return redirect(url_for('index'))

        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(
                run_ml_evaluation_wrapper,
                model_name, model_path, dataset_path, 'MLFlow'
            )
        
        flash(f"ML model evaluation started for {model_name}.")
        return render_template('tool_evaluate_ml.html', model_name=model_name, subcategory=subcategory, benchmark='MLFlow')
        
    except Exception as e:
        print(f"Error starting ML evaluation: {e}")
        flash(f"Error starting evaluation: {str(e)}")
        return redirect(url_for('index'))
    


@ml_s_t_mlflow_bp.route('/api/ml_progress/<model_name>')
def get_ml_evaluation_progress(model_name):
    progress = get_ml_progress(model_name)
    status = 'processing'
    if progress.get('progress_percent', 0) >= 100:
        status = 'complete'
    elif 'Error' in progress.get('current_task', ''):
        status = 'error'
    return jsonify({**progress, 'status': status})

@ml_s_t_mlflow_bp.route('/api/ml_results/<model_name>')
def get_ml_evaluation_results(model_name):
    results = get_ml_results(model_name)
    results = convert_numpy_types(results)
    return jsonify(results)

@ml_s_t_mlflow_bp.route('/api/export_results/<model_name>/<format>')
def export_results_api(model_name, format):
    """API endpoint to export results"""
    try:
        results = get_ml_results(model_name)
        if not results:
            return jsonify({"error": "No results found"}), 404
        
        if format == 'json':
            output_path = export_results_to_json(model_name)
            return send_file(output_path, as_attachment=True)
        elif format == 'csv':
            # Create CSV export
            output_dir = f"results/{model_name}"
            os.makedirs(output_dir, exist_ok=True)
            csv_path = os.path.join(output_dir, f"ml_evaluation_{model_name}.csv")
            metrics_df = pd.DataFrame([results['metrics']])
            metrics_df.to_csv(csv_path, index=False)
            return send_file(csv_path, as_attachment=True)
        else:
            return jsonify({"error": "Invalid format"}), 400
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@ml_s_t_mlflow_bp.route('/tool_evaluate_ml/<model_name>')
def tool_evaluate_ml_page(model_name):
    return render_template('tool_evaluate_ml.html', model_name=model_name, subcategory='supervised', benchmark='MLFlow')


@ml_s_t_mlflow_bp.route('/api/upload_new_model_version/<model_name>', methods=['POST'])
def upload_new_model_version(model_name):
    """
    Handles uploading a NEW model file.
    Logic:
    1. Rename existing 'model.pkl' to 'model_void'.
    2. Save new file as 'model.pkl'.
    3. Trigger a flag to increment version in details.json on next eval.
    """
    try:
        base_path = os.path.join('models', model_name)
        model_dir = os.path.join(base_path, 'model')
        dataset_dir = os.path.join(base_path, 'dataset')
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(dataset_dir, exist_ok=True)

        files_processed = []

        # Handle Model Upload
        if 'model_file' in request.files:
            file = request.files['model_file']
            if file.filename != '':
                current_model_path = os.path.join(model_dir, 'model.pkl')
                
                # Voiding logic
                if os.path.exists(current_model_path):
                    void_path = os.path.join(model_dir, 'model_void') # Removes extension as requested
                    if os.path.exists(void_path):
                        os.remove(void_path) # Overwrite existing void if any
                    os.rename(current_model_path, void_path)
                
                # Save new model
                file.save(current_model_path)
                files_processed.append("Model updated (v+1)")

        # Handle Dataset Upload
        if 'test_file' in request.files:
            file = request.files['test_file']
            if file.filename != '':
                current_test_path = os.path.join(dataset_dir, 'test.csv')
                
                # Voiding logic
                if os.path.exists(current_test_path):
                    void_path = os.path.join(dataset_dir, 'test_void')
                    if os.path.exists(void_path):
                        os.remove(void_path)
                    os.rename(current_test_path, void_path)
                    
                file.save(current_test_path)
                files_processed.append("Test data updated")

        # Automatically trigger evaluation with new_version=True
        if files_processed:
            model_path = os.path.join(model_dir, 'model.pkl')
            dataset_path = os.path.join(dataset_dir, 'test.csv')
            
            # Run inline or threaded? Threaded is better for UI response
            def run_new_version_eval():
                try:
                    run_ml_evaluation(model_name, model_path, dataset_path, new_version=True)
                except Exception as e:
                    print(f"New version eval failed: {e}")

            with ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit(run_new_version_eval)

            return jsonify({'status': 'success', 'message': f"Uploaded: {', '.join(files_processed)}. Evaluation started."})
        
        return jsonify({'status': 'error', 'message': 'No files provided'}), 400

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@ml_s_t_mlflow_bp.route('/api/evaluate_memory_model/<model_name>', methods=['POST'])
def evaluate_memory_model(model_name):
    """
    Uploads a model to MEMORY (temp file), evaluates against disk dataset, returns JSON results.
    Does NOT save the model to the model directory.
    """
    temp_model_path = None
    try:
        if 'model_file' not in request.files:
            return jsonify({'error': 'No model file provided'}), 400
        
        file = request.files['model_file']
        
        # Save to temp file to ensure joblib can read it easily
        # (joblib often prefers file paths or seekable file objects)
        fd, temp_model_path = tempfile.mkstemp(suffix='.pkl')
        os.close(fd)
        file.save(temp_model_path)
        
        # Load the temp model
        model = load_model(temp_model_path)
        if model is None:
            return jsonify({'error': 'Failed to load uploaded model'}), 400
            
        # Load existing dataset
        dataset_path = os.path.join('models', model_name, 'dataset', 'test.csv')
        if not os.path.exists(dataset_path):
            return jsonify({'error': 'Test dataset not found on server'}), 400
            
        df = pd.read_csv(dataset_path)
        if 'target' not in df.columns:
            target_col = df.columns[-1]
            df = df.rename(columns={target_col: 'target'})
            
        X_test = df.drop(columns=['target'])
        y_test = df['target'].values
        
        # Eval
        problem_type = detect_problem_type(model)
        y_pred = model.predict(X_test)
        y_pred = round_if_needed(y_pred, model_name)
        
        metrics = calculate_detailed_metrics(y_test, y_pred, problem_type, model, X_test)
        metrics = convert_numpy_types(metrics)
        
        return jsonify({'status': 'success', 'metrics': metrics})

    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500
    finally:
        # Cleanup temp file
        if temp_model_path and os.path.exists(temp_model_path):
            os.remove(temp_model_path)

@ml_s_t_mlflow_bp.route('/api/start_evaluation/<model_name>', methods=['POST'])
def api_start_evaluation(model_name):
    try:
        model_path = os.path.join('models', model_name, 'model', 'model.pkl')
        dataset_path = os.path.join('models', model_name, 'dataset', 'test.csv')
        
        if not os.path.exists(model_path) or not os.path.exists(dataset_path):
            return jsonify({'error': 'Model or Dataset not found'}), 400

        clear_ml_progress(model_name)

        def run_evaluation_task():
            try:
                # Regular evaluation (overwrite current version results)
                run_ml_evaluation(model_name, model_path, dataset_path, new_version=False)
            except Exception as e:
                print(f"Evaluation task failed: {e}")
                update_progress(model_name, f"Error: {str(e)}", 0)
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(run_evaluation_task)

        return jsonify({'status': 'started', 'message': 'Evaluation started'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    
@ml_s_t_mlflow_bp.route('/api/clear_ml_evaluation/<model_name>', methods=['POST'])
def clear_ml_evaluation_data(model_name):
    """Clear evaluation progress and results for a specific model."""
    try:
        clear_ml_progress(model_name)
        return jsonify({'status': 'success', 'message': f'Cleared evaluation data for {model_name}'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
    
@ml_s_t_mlflow_bp.route('/api/available_models')
def get_available_models():
    """API endpoint to get list of available models"""
    models = list_available_results()
    return jsonify(models)

@ml_s_t_mlflow_bp.route('/api/download_report/<model_name>')
def download_ml_report(model_name):
    """Download comprehensive evaluation report."""
    try:
        results = get_ml_results(model_name)
        if not results or 'error' in results:
            flash("No results available for download")
            return redirect(url_for('index'))
        
        # Generate report content
        report_content = generate_ml_report(results)
        
        # Create response
        response = make_response(report_content)
        response.headers['Content-Type'] = 'application/json'
        response.headers['Content-Disposition'] = f'attachment; filename=ml_evaluation_report_{model_name}.json'
        
        return response
        
    except Exception as e:
        flash(f"Error generating report: {str(e)}")
        return redirect(url_for('index'))

@ml_s_t_mlflow_bp.route('/api/upload_challenger/<model_name>', methods=['POST'])
def upload_challenger(model_name):
    """Upload new model/dataset files to replace the current ones for evaluation."""
    try:
        # Define paths
        base_path = os.path.join('models', model_name)
        model_dir = os.path.join(base_path, 'model')
        dataset_dir = os.path.join(base_path, 'dataset')
        
        # Ensure directories exist
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(dataset_dir, exist_ok=True)

        files_processed = []

        # Handle Model File
        if 'model_file' in request.files:
            file = request.files['model_file']
            if file.filename != '':
                # Force rename to standard 'model.pkl' for consistency in the tool
                # or keep original name but ensure the tool finds it
                file.save(os.path.join(model_dir, 'model.pkl'))
                files_processed.append("Model (model.pkl)")

        # Handle Test Data
        if 'test_file' in request.files:
            file = request.files['test_file']
            if file.filename != '':
                file.save(os.path.join(dataset_dir, 'test.csv'))
                files_processed.append("Test Data (test.csv)")

        if not files_processed:
            return jsonify({'status': 'error', 'message': 'No valid files provided'}), 400

        return jsonify({
            'status': 'success', 
            'message': f'Successfully uploaded: {", ".join(files_processed)}. Ready for evaluation.'
        })

    except Exception as e:
        print(f"Upload error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500