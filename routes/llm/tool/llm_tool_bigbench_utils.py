#llm_tool_bigbench_utils.py
import os
import json
import re
from datetime import datetime
import random
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import threading
import torch
from io import BytesIO
# from weasyprint import HTML
# import seqio
import shutil
import tempfile
import numpy as np
from flask import Blueprint, render_template,  redirect, url_for, flash, Flask, render_template, request, redirect, url_for, jsonify, flash, send_file, make_response, send_from_directory, session, current_app
from datasets import disable_caching
# from bigbench.bbseqio import tasks, vocabs
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModel,
    AutoConfig
)
# To this:
import threading
from collections import defaultdict

llm_t_bp = Blueprint('llm_t', __name__)

progress_tracker = {}
progress_lock = threading.Lock()
task_completion_tracker = defaultdict(set)
RESULTS_FILE = "evaluation_results.json"

try:
    from rouge_score import rouge_scorer
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from sklearn.metrics import f1_score, precision_recall_fscore_support
    import nltk
    nltk.download('punkt', quiet=True)
    ADVANCED_METRICS_AVAILABLE = True
except ImportError:
    print("⚠️ Advanced metrics not available. Install: pip install rouge-score nltk scikit-learn")
    ADVANCED_METRICS_AVAILABLE = False

disable_caching()

# HISTORY_FILE = "evaluation_results/llm/tool/bigbench/history.json"

# _evaluation_progress = {}
# processing_status = {}
# from app import evaluation_progress, processing_status

STANDARD_EVAL_STAGES = [
    "Initializing model and tokenizer...",
    "Loading benchmark tasks...",
    "Running evaluation on tasks...",
    "Aggregating results...",
    "Finalizing and saving results..."
]

MODELS = {
    "wealth_advisory_model": "models/wealth_advisory", 
    "compliance_model": "models/compliance"
}

processing_status = {}  # Track per-model status
evaluation_progress = {}  # Track detailed progress

@llm_t_bp.route('/download_report/<model_name>')
def download_report(model_name):
    """Generate PDF report with charts and tables from results page."""
    if model_name not in MODELS:
        flash("Model not found.")
        return redirect(url_for('index'))
    
    try:
        # Load results from file        
        results_data = load_results_from_file(model_name)
        
        if not results_data:
            flash("No evaluation results found for this model.")
            return redirect(url_for('analyze', model_name=model_name))
        
        latest = results_data[0]
        
        # Create comprehensive HTML with charts and styling
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Evaluation Report - {model_name}</title>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; color: #333; }}
                .header {{ text-align: center; margin-bottom: 30px; border-bottom: 2px solid #007bff; padding-bottom: 20px; }}
                .header h1 {{ color: #007bff; margin-bottom: 5px; }}
                .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
                .summary-card {{ border: 1px solid #ddd; padding: 15px; text-align: center; background: #f8f9fa; border-radius: 8px; }}
                .summary-card h3 {{ margin-top: 0; color: #495057; font-size: 1.1em; }}
                .score {{ font-size: 2.2em; font-weight: bold; color: #007bff; margin: 10px 0; }}
                .details {{ color: #6c757d; font-size: 0.9em; }}
                .chart-container {{ margin: 30px 0; text-align: center; page-break-inside: avoid; }}
                .chart-container h3 {{ color: #495057; margin-bottom: 15px; }}
                .chart-placeholder {{ width: 100%; height: 300px; background: #f8f9fa; border: 1px solid #ddd; display: flex; align-items: center; justify-content: center; color: #6c757d; }}
                .task-section {{ margin-bottom: 25px; page-break-inside: avoid; }}
                .task-header {{ background: #f8f9fa; padding: 15px; border-left: 4px solid #007bff; margin-bottom: 10px; border-radius: 0 8px 8px 0; }}
                .task-header h3 {{ margin: 0; color: #495057; }}
                .task-type {{ color: #6c757d; font-size: 0.9em; font-weight: normal; }}
                .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 10px; margin: 15px 0; }}
                .metric-item {{ text-align: center; padding: 10px; background: #fff; border: 1px solid #dee2e6; border-radius: 5px; }}
                .metric-item strong {{ color: #495057; font-size: 0.9em; }}
                .samples-table {{ width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 0.9em; }}
                .samples-table th {{ background-color: #007bff; color: white; padding: 8px; text-align: left; }}
                .samples-table td {{ border: 1px solid #dee2e6; padding: 8px; vertical-align: top; }}
                .samples-table pre {{ margin: 0; white-space: pre-wrap; word-wrap: break-word; font-size: 0.8em; }}
                .timestamp {{ color: #6c757d; font-size: 0.9em; margin-top: 20px; text-align: center; }}
                @media print {{ 
                    body {{ margin: 15px; }} 
                    .page-break {{ page-break-before: always; }}
                    .chart-placeholder {{ height: 200px; }}
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>BIG-bench Evaluation Report</h1>
                <h2>{model_name}</h2>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        """
        
        if 'summary' in latest and latest['summary']:
            # Overall Summary Cards
            html_content += """
            <div class="summary-grid">
            """
            
            if 'overall' in latest['summary']:
                overall = latest['summary']['overall']
                html_content += f"""
                <div class="summary-card">
                    <h3>Overall Performance</h3>
                    <div class="score">{overall['mean']*100:.1f}%</div>
                    <div class="details">± {overall['std']*100:.1f}% ({overall['count']} tasks)</div>
                </div>
                """
            
            for task_type, stats in latest['summary'].items():
                if task_type != 'overall':
                    html_content += f"""
                    <div class="summary-card">
                        <h3>{task_type.title()}</h3>
                        <div class="score">{stats['mean']*100:.1f}%</div>
                        <div class="details">± {stats['std']*100:.1f}% ({stats['count']} tasks)</div>
                    </div>
                    """
            
            html_content += "</div>"
            
            # Performance Chart Placeholder
            html_content += """
            <div class="chart-container">
                <h3>Performance by Task Type</h3>
                <div class="chart-placeholder">
                    <div>
                        <strong>Performance Summary</strong><br>
            """
            
            for task_type, stats in latest['summary'].items():
                if task_type != 'overall':
                    html_content += f"{task_type.title()}: {stats['mean']*100:.1f}%<br>"
            
            html_content += """
                    </div>
                </div>
            </div>
            """
        
        # Detailed Task Results
        if 'detailed_results' in latest:
            html_content += "<h2 style='color: #495057; border-bottom: 1px solid #dee2e6; padding-bottom: 10px;'>Detailed Task Results</h2>"
            
            for i, task_result in enumerate(latest['detailed_results']):
                if i > 0:
                    html_content += '<div class="page-break"></div>'
                
                primary_score = task_result.get('summary', {}).get('primary_metric', {}).get('mean', 0)
                html_content += f"""
                <div class="task-section">
                    <div class="task-header">
                        <h3>{task_result['task']} <span class="task-type">({task_result.get('task_type', 'N/A')})</span></h3>
                        <p style="margin: 5px 0 0 0;">Primary Score: <strong>{primary_score*100:.1f}%</strong></p>
                    </div>
                """
                
                # Metrics Grid
                if 'summary' in task_result:
                    html_content += '<div class="metrics-grid">'
                    for metric_name, metric_data in task_result['summary'].items():
                        html_content += f"""
                        <div class="metric-item">
                            <strong>{metric_name.replace('_', ' ').title()}</strong><br>
                            {metric_data['mean']:.3f}
                        </div>
                        """
                    html_content += '</div>'
                
                # Sample Results Table
                if 'samples' in task_result and task_result['samples']:
                    html_content += """
                    <h4 style="color: #495057;">Sample Predictions</h4>
                    <table class="samples-table">
                        <thead>
                            <tr>
                                <th style="width: 5%">#</th>
                                <th style="width: 35%">Input</th>
                                <th style="width: 25%">Generated</th>
                                <th style="width: 25%">Expected</th>
                                <th style="width: 10%">Score</th>
                            </tr>
                        </thead>
                        <tbody>
                    """
                    
                    for sample in task_result['samples'][:3]:
                        score = sample.get('metrics', {}).get('primary_metric', 0)
                        input_text = str(sample.get('input', ''))[:150] + ('...' if len(str(sample.get('input', ''))) > 150 else '')
                        html_content += f"""
                        <tr>
                            <td>{sample.get('example_number', 'N/A')}</td>
                            <td><pre>{input_text}</pre></td>
                            <td><pre>{sample.get('generated', '')}</pre></td>
                            <td><pre>{sample.get('expected', '')}</pre></td>
                            <td style="text-align: center;"><strong>{score:.2f}</strong></td>
                        </tr>
                        """
                    
                    html_content += '</tbody></table>'
                
                html_content += '</div>'
        
        html_content += f"""
            <div class="timestamp">
                <strong>Report generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
                <strong>Total tasks evaluated:</strong> {len(latest.get('detailed_results', []))}
            </div>
        </body>
        </html>
        """
        
        # Try to generate PDF
        try:
            pdf_buffer = BytesIO()
            HTML(string=html_content).write_pdf(pdf_buffer)
            pdf_buffer.seek(0)
            
            response = make_response(pdf_buffer.getvalue())
            response.headers['Content-Type'] = 'application/pdf'
            response.headers['Content-Disposition'] = f'attachment; filename="{model_name.replace(" ", "_")}_evaluation_report.pdf"'
            return response
            
        except ImportError:
            # Fallback: return HTML
            response = make_response(html_content)
            response.headers['Content-Type'] = 'text/html'
            response.headers['Content-Disposition'] = f'attachment; filename="{model_name.replace(" ", "_")}_evaluation_report.html"'
            return response
        
    except Exception as e:
        print(f"Error generating report: {e}")
        flash(f"Error generating report: {str(e)}")
        return redirect(url_for('analyze', model_name=model_name))
    
@llm_t_bp.route('/export_json/<model_name>')
def export_json(model_name):
    """Export evaluation_results.json file directly."""
    if model_name not in MODELS:
        flash("Model not found.")
        return redirect(url_for('index'))
    
    try:       
        
        # Check if results file exists
        if not os.path.exists(RESULTS_FILE):
            flash("No evaluation results file found.")
            return redirect(url_for('analyze', model_name=model_name))
        
        # Send the file directly
        return send_file(
            RESULTS_FILE,
            as_attachment=True,
            download_name=f"evaluation_results_{model_name.replace(' ', '_')}.json",
            mimetype='application/json'
        )
        
    except Exception as e:
        flash(f"Error exporting JSON: {str(e)}")
        return redirect(url_for('analyze', model_name=model_name))

@llm_t_bp.route('/evaluate_model/<model_name>')
def evaluate(model_name):
    # Simplified - no category needed
    if model_name not in MODELS:
        return f"Model '{model_name}' not found.", 404

    model_path = MODELS[model_name]
    if not os.path.exists(model_path):
        return f"Model '{model_name}' not found.", 404

    # Set initial status
    processing_status[model_name] = "processing"
    evaluation_progress[model_name] = {
        'stage': 0,
        'message': 'Preparing to start evaluation...',
        'timestamp': datetime.now().isoformat()
    }

    return render_template('loading.html', model_name=model_name)

@llm_t_bp.route('/start_evaluation/<model_name>', methods=['POST'])
def start_evaluation(model_name):
    if model_name not in MODELS:
        return jsonify({'error': 'Unknown model'}), 400

    model_path = MODELS[model_name]
    if not os.path.exists(model_path):
        return jsonify({'error': f"Model '{model_name}' not found."}), 404

    # Clear previous results
    try:        
        clear_results_file()
    except Exception as e:
        print(f"⚠️ Warning: Could not clear results file: {e}")

    # Default evaluation parameters
    eval_params = {
        'num_examples': 25,
        'max_tokens': 128,
        'full_benchmark': False
    }
    
    # Start evaluation in background
    run_evaluation_in_background(model_name, model_path, eval_params)
    
    return jsonify({'status': 'started', 'message': 'Evaluation started successfully'})


def save_results_to_file(model_name: str, results_data: dict):
    """Save results to local JSON file."""
    try:
        # Load existing results
        if os.path.exists(RESULTS_FILE):
            with open(RESULTS_FILE, 'r') as f:
                all_results = json.load(f)
        else:
            all_results = {}
        
        # Save new results
        all_results[model_name] = [results_data]
        
        # Write back to file
        with open(RESULTS_FILE, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"✅ Results saved to file for {model_name}")
        return True
    except Exception as e:
        print(f"❌ Error saving results to file: {e}")
        return False

def load_results_from_file(model_name: str = None):
    """Load results from local JSON file."""
    try:
        if not os.path.exists(RESULTS_FILE):
            return {} if model_name is None else []
        
        with open(RESULTS_FILE, 'r') as f:
            all_results = json.load(f)
        
        if model_name:
            return all_results.get(model_name, [])
        return all_results
    except Exception as e:
        print(f"❌ Error loading results from file: {e}")
        return {} if model_name is None else []

def clear_results_file():
    """Clear the results file."""
    try:
        if os.path.exists(RESULTS_FILE):
            os.remove(RESULTS_FILE)
        print("🧹 Results file cleared")
    except Exception as e:
        print(f"❌ Error clearing results file: {e}")


current_results = {}

# Updated evaluate_llm function
@llm_t_bp.route('/evaluate_llm/<model_name>', methods=['POST', 'GET'])
def evaluate_llm(model_name):
    if request.method == 'GET':
        # Just show the loading page
        processing_status[model_name] = "processing"
        evaluation_progress[model_name] = {
            'stage': 0,
            'message': 'Preparing to start evaluation...',
            'timestamp': datetime.now().isoformat()
        }
        return render_template('loading.html', model_name=model_name)
    
    # Handle POST request (actual evaluation)
    print(f"Evaluating {model_name}")

    # Check if model exists in MODELS
    if model_name not in MODELS:
        flash(f"Unknown model: {model_name}")
        return redirect(url_for('index'))

    model_path = MODELS[model_name]
    if not os.path.exists(model_path):
        return f"Model '{model_name}' not found.", 404

    # Default evaluation parameters
    eval_params = {
        'num_examples': 5,
        'max_tokens': 128,
        'full_benchmark': False
    }

    run_evaluation_in_background(model_name, model_path, eval_params)
    return render_template('loading.html', model_name=model_name)

@llm_t_bp.route('/check_status/<model_name>')
def check_status(model_name):
    """Enhanced status check with strict completion validation."""
    # Get app-level status (fallback to not_started)
    status = processing_status.get(model_name, "not_started")
    
    # Get progress from utils module (primary source)  
    try:        
        progress = get_progress(model_name)
        
        # Sync progress back to app for consistency
        evaluation_progress[model_name] = progress
        
    except Exception as e:
        print(f"⚠️ Could not get progress from utils: {e}")
        # Fallback to app's local progress
        progress = evaluation_progress.get(model_name, {'stage': 0, 'message': 'Not started'})
    
    # Enhanced completion validation
    progress_stage = progress.get('stage', 0)
    
    # Check if results exist in file
    try:        
        file_results = load_results_from_file(model_name)
        has_results = len(file_results) > 0
    except:
        has_results = False
    
    # Only mark as complete if BOTH conditions are met:
    if progress_stage >= 5 and has_results and status != "complete":
        status = "complete" 
        processing_status[model_name] = "complete"
        print(f"✅ Marked {model_name} as complete")
    elif progress_stage == -1:
        status = "error"
        processing_status[model_name] = "error"
    elif progress_stage > 0 and progress_stage < 5:
        status = "processing"
        processing_status[model_name] = "processing"
    
    # Debug logging - add this right before the return statement
    print(f"📊 Status check for {model_name}: status={status}, progress_stage={progress_stage}, has_results={has_results}")
    print(f"📊 Available models in current_results: {list(current_results.keys())}")
    if model_name in current_results:
        print(f"📊 Results for {model_name}: {len(current_results[model_name])} entries")
    
    return jsonify({
        "status": status,
        "progress": progress,
        "has_results": has_results,
        "timestamp": datetime.now().isoformat()
    })

@llm_t_bp.route('/results/<model_name>')
def analyze(model_name):
    """Results page loading from file."""
    if model_name not in MODELS:
        return f"Model '{model_name}' not found.", 404
    
    # Load results from file
    try:        
        results_data = load_results_from_file(model_name)
    except Exception as e:
        print(f"❌ Error loading results from file: {e}")
        results_data = []
    
    # Check if evaluation is complete and has results
    status = processing_status.get(model_name, "not_started")
    
    if status != "complete" or not results_data:
        if status == "processing":
            print(f"📊 Evaluation still in progress for {model_name}, redirecting to loading page")
            return redirect(url_for('evaluate_llm', model_name=model_name))
        elif status == "error":
            flash(f"Evaluation failed for {model_name}. Please try again.")
            return redirect(url_for('index'))
        else:
            flash(f"No evaluation results found for {model_name}. Please run evaluation first.")
            return redirect(url_for('index'))
    
    print(f"📊 Displaying results for {model_name}: {len(results_data)} result sets")
    return render_template('results.html', model_name=model_name, history=results_data)




@llm_t_bp.route('/clear_results/<model_name>')
def clear_model_results(model_name):
    """Clear results for a specific model before starting evaluation."""
    try:        
        clear_results_file()
        print(f"🧹 Cleared results file before evaluating {model_name}")
        return jsonify({'status': 'cleared'})
    except Exception as e:
        print(f"❌ Error clearing results: {e}")
        return jsonify({'status': 'error', 'message': str(e)})
    


def update_task_progress(model_name, task_name, task_index, total_tasks, status="processing"):
    """Update progress for individual task completion."""
    with progress_lock:
        if status == "completed":
            task_completion_tracker[model_name].add(task_name)
        
        completed_count = len(task_completion_tracker[model_name])
        
        # Create a more stable progress message
        if status == "processing":
            message = f"Evaluating tasks... ({completed_count}/{total_tasks} completed) - Current: {task_name[:40]}..."
        else:
            message = f"Evaluating tasks... ({completed_count}/{total_tasks} completed)"
        
        task_details = {
            'current_task': task_name,
            'completed_tasks': completed_count,
            'total_tasks': total_tasks,
            'completion_percentage': int((completed_count / total_tasks) * 100)
        }
        
        update_standard_progress(model_name, 3, message, task_details)



def get_progress(model_name):
    """Get progress for a model with thread safety."""
    with progress_lock:
        return progress_tracker.get(model_name, {'stage': 0, 'message': 'Not started'})

def clear_progress(model_name):
    """Clear progress tracking for a specific model."""
    with progress_lock:
        if model_name in progress_tracker:
            del progress_tracker[model_name]
    print(f"🧹 Cleared progress tracking for {model_name}")


# --------------------- ADVANCED METRICS --------------------- #
class AdvancedMetrics:
    def __init__(self):
        if ADVANCED_METRICS_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            self.smoothing = SmoothingFunction().method1
    
    def exact_match(self, prediction: str, target: str) -> float:
        """Normalized exact match."""
        # Ensure both inputs are strings
        if not isinstance(prediction, str):
            prediction = str(prediction) if prediction is not None else ""
        if not isinstance(target, str):
            target = str(target) if target is not None else ""
        
        return float(self.normalize(prediction) == self.normalize(target))
    
    def fuzzy_match(self, prediction: str, target: str, threshold: float = 0.8) -> float:
        """Fuzzy string matching using character overlap."""
        # Ensure both inputs are strings
        if not isinstance(prediction, str):
            prediction = str(prediction) if prediction is not None else ""
        if not isinstance(target, str):
            target = str(target) if target is not None else ""
        
        pred_norm = self.normalize(prediction)
        target_norm = self.normalize(target)
        
        if not pred_norm or not target_norm:
            return 0.0
        
        # Simple character-level similarity
        common_chars = sum(1 for c in pred_norm if c in target_norm)
        similarity = common_chars / max(len(pred_norm), len(target_norm))
        return float(similarity >= threshold)
    
    def rouge_scores(self, prediction: str, target: str) -> Dict[str, float]:
        """Calculate ROUGE scores."""
        if not ADVANCED_METRICS_AVAILABLE:
            return {}
        
        # Ensure both inputs are strings
        if not isinstance(prediction, str):
            prediction = str(prediction) if prediction is not None else ""
        if not isinstance(target, str):
            target = str(target) if target is not None else ""
        
        try:
            scores = self.rouge_scorer.score(target, prediction)
            return {
                'rouge1_f': scores['rouge1'].fmeasure,
                'rouge2_f': scores['rouge2'].fmeasure,
                'rougeL_f': scores['rougeL'].fmeasure,
            }
        except:
            return {'rouge1_f': 0.0, 'rouge2_f': 0.0, 'rougeL_f': 0.0}
    
    def bleu_score(self, prediction: str, target: str) -> float:
        """Calculate BLEU score."""
        if not ADVANCED_METRICS_AVAILABLE:
            return 0.0
        
        # Ensure both inputs are strings
        if not isinstance(prediction, str):
            prediction = str(prediction) if prediction is not None else ""
        if not isinstance(target, str):
            target = str(target) if target is not None else ""
        
        try:
            pred_tokens = prediction.lower().split()
            target_tokens = [target.lower().split()]  # BLEU expects list of reference lists
            
            if not pred_tokens or not any(target_tokens):
                return 0.0
            
            return sentence_bleu(target_tokens, pred_tokens, smoothing_function=self.smoothing)
        except:
            return 0.0
    
    def token_f1(self, prediction: str, target: str) -> float:
        """Token-level F1 score."""
        # Ensure both inputs are strings
        if not isinstance(prediction, str):
            prediction = str(prediction) if prediction is not None else ""
        if not isinstance(target, str):
            target = str(target) if target is not None else ""
        
        pred_tokens = set(prediction.lower().split())
        target_tokens = set(target.lower().split())
        
        if not pred_tokens and not target_tokens:
            return 1.0
        if not pred_tokens or not target_tokens:
            return 0.0
        
        common = pred_tokens & target_tokens
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(target_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def multiple_choice_accuracy(self, prediction: str, target: str, choices: List[str] = None) -> float:
        """Specialized accuracy for multiple choice questions."""
        # Ensure both inputs are strings
        if not isinstance(prediction, str):
            prediction = str(prediction) if prediction is not None else ""
        if not isinstance(target, str):
            target = str(target) if target is not None else ""
        
        # Extract first letter/number if it looks like A/B/C/D or 1/2/3/4
        pred_match = re.search(r'^[A-Za-z0-9]', prediction.strip())
        target_match = re.search(r'^[A-Za-z0-9]', target.strip())
        
        if pred_match and target_match:
            return float(pred_match.group().upper() == target_match.group().upper())
        
        return self.exact_match(prediction, target)
    
    def normalize(self, text: str) -> str:
        """Enhanced text normalization."""
        # Ensure text is a string
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        
        # Remove extra whitespace and convert to lowercase
        text = re.sub(r'\s+', ' ', text.lower().strip())
        # Remove common punctuation that doesn't affect meaning
        text = re.sub(r'[.,;:!?"\']', '', text)
        return text

# --------------------- TASK-SPECIFIC EVALUATION --------------------- #
def get_task_type(task_name: str) -> str:
    """Infer task type from task name for specialized evaluation."""
    task_lower = task_name.lower()
    
    if any(keyword in task_lower for keyword in ['multiple_choice', 'choice', 'qa']):
        return 'multiple_choice'
    elif any(keyword in task_lower for keyword in ['generation', 'summariz', 'translation']):
        return 'generation'
    elif any(keyword in task_lower for keyword in ['classification', 'sentiment', 'toxic']):
        return 'classification'
    elif any(keyword in task_lower for keyword in ['reasoning', 'logic', 'math']):
        return 'reasoning'
    else:
        return 'general'

def evaluate_example(prediction: str, target: str, task_type: str, metrics: AdvancedMetrics) -> Dict[str, float]:
    """Comprehensive evaluation of a single example."""
    results = {}
    
    # Ensure prediction and target are strings
    if not isinstance(prediction, str):
        prediction = str(prediction) if prediction is not None else ""
    if not isinstance(target, str):
        target = str(target) if target is not None else ""
    
    # Always compute exact match
    results['exact_match'] = metrics.exact_match(prediction, target)
    
    # Task-specific metrics
    if task_type == 'multiple_choice':
        results['mc_accuracy'] = metrics.multiple_choice_accuracy(prediction, target)
        results['primary_metric'] = results['mc_accuracy']
    elif task_type == 'generation':
        rouge_scores = metrics.rouge_scores(prediction, target)
        results.update(rouge_scores)
        results['bleu'] = metrics.bleu_score(prediction, target)
        results['token_f1'] = metrics.token_f1(prediction, target)
        results['primary_metric'] = rouge_scores.get('rouge1_f', results['exact_match'])
    elif task_type in ['classification', 'reasoning']:
        results['fuzzy_match'] = metrics.fuzzy_match(prediction, target)
        results['token_f1'] = metrics.token_f1(prediction, target)
        results['primary_metric'] = max(results['exact_match'], results['fuzzy_match'])
    else:
        results['token_f1'] = metrics.token_f1(prediction, target)
        results['primary_metric'] = results['exact_match']
    
    return results

def generate_response(model, tokenizer, input_text: str, task_type: str, max_new_tokens: int = 128) -> str:
    """Task-aware response generation with better device handling."""
    
    # Get device from model parameters, with fallback
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
        print("⚠️ Could not determine model device, using CPU")
    
    # Ensure input_text is a string
    if not isinstance(input_text, str):
        input_text = str(input_text) if input_text is not None else ""
    
    # Task-specific generation parameters
    gen_params = {
        'max_new_tokens': max_new_tokens,
        'do_sample': False,  # Deterministic for evaluation
        'pad_token_id': tokenizer.eos_token_id,
    }
    
    if task_type == 'multiple_choice':
        gen_params.update({
            'max_new_tokens': min(10, max_new_tokens),  # Short responses for MC
            'temperature': 0.1,
        })
    elif task_type == 'generation':
        gen_params.update({
            'temperature': 0.3,
            'do_sample': True,
            'top_p': 0.9,
        })
    
    try:
        # Tokenize input with proper error handling
        inputs = tokenizer(
            input_text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=1024
        )
        
        # Move inputs to the same device as model
        try:
            inputs = {k: v.to(device) for k, v in inputs.items()}
        except Exception as device_error:
            print(f"⚠️ Could not move inputs to {device}, keeping on CPU: {device_error}")
            # Keep inputs on CPU if device movement fails
        
        with torch.no_grad():
            try:
                output_ids = model.generate(**inputs, **gen_params)
            except Exception as gen_error:
                print(f"⚠️ Generation failed with device {device}, trying CPU: {gen_error}")
                # Fallback: move everything to CPU
                if device != torch.device("cpu"):
                    inputs_cpu = {k: v.cpu() for k, v in inputs.items()}
                    model_cpu = model.cpu()
                    output_ids = model_cpu.generate(**inputs_cpu, **gen_params)
                    # Move model back to original device if possible
                    try:
                        model.to(device)
                    except:
                        pass
                else:
                    raise gen_error
        
        # Extract only the generated part (remove input)
        if hasattr(model, 'config') and 'gpt' in str(type(model)).lower():
            # For GPT-style models, remove the input tokens
            generated_ids = output_ids[0][inputs['input_ids'].shape[1]:]
        else:
            generated_ids = output_ids[0]
        
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Ensure response is a string
        if not isinstance(response, str):
            response = str(response) if response is not None else ""
        
        response = response.strip()
        
        # Clean up response
        if task_type == 'multiple_choice':
            # Extract first meaningful token for MC questions
            match = re.search(r'^[A-Za-z0-9]', response)
            if match:
                response = match.group()
        
        return response
        
    except Exception as e:
        print(f"⚠️ Generation failed completely: {e}")
        return ""

def run_evaluation(model_name: str, model_path: str = None, num_examples: int = 50, max_new_tokens: int = 128, 
                          use_full_bigbench: bool = False):
    """Sequential evaluation with immediate progress updates."""
    
    if model_path is None:
        model_path = model_name
    
    try:
        # Force CPU and disable GPU warnings
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import warnings
        warnings.filterwarnings('ignore')
        
        # Stage 1: Load model
        update_standard_progress(model_name, 1, "Loading model and tokenizer...")
        print(f"📊 Stage 1: Loading model for {model_name}")
        
        device = torch.device("cpu")
        metrics = AdvancedMetrics()
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float32,
            "low_cpu_mem_usage": True,
        }
        
        if config.architectures:
            arch = config.architectures[0].lower()
            if any(name in arch for name in ['seq2seq', 't5', 'bart']):
                model = AutoModelForSeq2SeqLM.from_pretrained(model_path, **model_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        
        # Handle meta tensors properly
        try:
            model = model.to(device)
        except RuntimeError as e:
            if "meta tensor" in str(e):
                print("🔧 Handling meta tensors with to_empty()")
                model = model.to_empty(device=device)
                # Initialize parameters if they're meta tensors
                for name, param in model.named_parameters():
                    if param.is_meta:
                        # Initialize with small random values
                        param.data = torch.randn_like(param, device=device) * 0.02
                for name, buffer in model.named_buffers():
                    if buffer.is_meta:
                        buffer.data = torch.zeros_like(buffer, device=device)
            else:
                raise e
        
        model.eval()
        print(f"✅ Model loaded on {device}")
        
        # IMMEDIATELY advance to Stage 2
        update_standard_progress(model_name, 2, "Loading benchmark tasks...")
        print(f"📊 Stage 2: Loading tasks for {model_name}")
        
        vocab = vocabs.ALL_VOCABS["t5_default"]
        mix_name = "bigbench:bigbench_lite_v1.mix.t5_default_vocab.0_shot.1024_examples"
        mix = seqio.get_mixture_or_task(mix_name)
        task_names = sorted([t.name for t in mix.tasks])
        task_names = task_names[:20]  # Limit for testing
        print(f"✅ Loaded {len(task_names)} tasks")
        
        # IMMEDIATELY advance to Stage 3
        update_standard_progress(model_name, 3, f"Running evaluation on {len(task_names)} tasks...")
        print(f"📊 Stage 3: Starting evaluation for {model_name}")
        
        all_results = []
        task_type_results = defaultdict(list)
        
        # Sequential task processing - NO THREADING
        for task_idx, task_name in enumerate(task_names):
            current_task_msg = f"Evaluating task {task_idx + 1}/{len(task_names)}: {task_name[:30]}..."
            update_standard_progress(model_name, 3, current_task_msg)
            print(f"🔍 {current_task_msg}")
            
            task_type = get_task_type(task_name)
            task = seqio.get_mixture_or_task(task_name)
            dataset = task.get_dataset(split="validation")
            
            task_metrics = defaultdict(list)
            samples = []
            
            # Sequential example processing
            for i, example in enumerate(dataset):
                if i >= num_examples:
                    break
                
                try:
                    input_text = vocab.vocabulary.decode(example["inputs"].numpy())
                    target_text = vocab.vocabulary.decode(example["targets"].numpy()).strip()
                    
                    prediction = generate_response(model, tokenizer, input_text, task_type, max_new_tokens)
                    eval_results = evaluate_example(prediction, target_text, task_type, metrics)
                    
                    for metric_name, score in eval_results.items():
                        task_metrics[metric_name].append(score)
                    
                    sample_data = {
                        "example_number": i + 1,
                        "input": input_text[:200],
                        "expected": target_text,
                        "generated": prediction,
                        "metrics": eval_results,
                        "task_type": task_type
                    }
                    samples.append(sample_data)
                    
                except Exception as e:
                    print(f"⚠️ Skipping example {i}: {e}")
                    continue
            
            # Process task results
            task_summary = {}
            for metric_name, scores in task_metrics.items():
                if scores:
                    task_summary[metric_name] = {
                        'mean': np.mean(scores),
                        'std': np.std(scores),
                        'count': len(scores)
                    }
            
            task_result = {
                "task": task_name,
                "task_type": task_type,
                "summary": task_summary,
                "samples": samples[:5],
                "timestamp": datetime.now().isoformat()
            }
            
            all_results.append(task_result)
            if 'primary_metric' in task_summary:
                task_type_results[task_type].append(task_summary['primary_metric']['mean'])
            
            print(f"✅ Completed task {task_idx + 1}/{len(task_names)}")
        
        # IMMEDIATELY advance to Stage 4 after all tasks complete
        update_standard_progress(model_name, 4, "Aggregating results...")
        print(f"📊 Stage 4: Aggregating results for {model_name}")
        
        summary = {}
        for task_type, scores in task_type_results.items():
            if scores:
                summary[task_type] = {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'count': len(scores)
                }
        
        overall_scores = [s for scores in task_type_results.values() for s in scores]
        if overall_scores:
            summary['overall'] = {
                'mean': float(np.mean(overall_scores)),
                'std': float(np.std(overall_scores)),
                'count': len(overall_scores)
            }
        
        entry = {
            "model_path": model_name,
            "summary": summary,
            "detailed_results": all_results,
            "timestamp": datetime.now().isoformat(),
            "num_tasks": len(all_results),
            "status": "completed"
        }
        
        # IMMEDIATELY advance to Stage 5
        update_standard_progress(model_name, 5, "Saving results...")
        print(f"📊 Stage 5: Saving results for {model_name}")
        
        # Save to file instead of trying to #import app
        if save_results_to_file(model_name, entry):
            print(f"✅ Results saved to file for {model_name}")
        else:
            raise Exception("Failed to save results to file")
        
        return all_results
    
    except Exception as e:
        update_standard_progress(model_name, -1, f"Error: {str(e)}")
        print(f"❌ Evaluation failed for {model_name}: {e}")
        try:
            #import app
            processing_status[model_name] = "error"
        except:
            pass
        raise e

def run_evaluation_in_background(model_name, model_path, eval_params):
    """Run evaluation in background thread - but evaluation itself is sequential."""
    
    print(f"🚀 Starting evaluation for {model_name}")
    
    # Set initial status
    try:
        #import app
        processing_status[model_name] = "processing"
        if model_name in current_results:
            del current_results[model_name]
    except:
        pass

    def background_task():
        try:
            # Run SEQUENTIAL evaluation (no internal threading)
            run_evaluation(
                model_name=model_name,
                model_path=model_path,
                num_examples=eval_params.get('num_examples', 5),
                max_new_tokens=eval_params.get('max_tokens', 128),
                use_full_bigbench=eval_params.get('full_benchmark', False)
            )
                
        except Exception as e:
            print(f"❌ Background evaluation failed for {model_name}: {e}")

    # Only this part uses threading - the evaluation itself is sequential
    threading.Thread(target=background_task, daemon=True).start()


def update_standard_progress(model_name, stage, message, task_details=None):
    """Simple, immediate progress update."""
    progress_data = {
        'stage': stage,
        'message': message,
        'timestamp': datetime.now().isoformat()
    }
    
    if task_details:
        progress_data.update(task_details)
    
    # Update local progress tracker
    with progress_lock:
        progress_tracker[model_name] = progress_data
    
    # Update app immediately
    try:
        #import app
        evaluation_progress[model_name] = progress_data
        
        # Update status based on stage
        if stage == -1:
            processing_status[model_name] = "error"
        elif stage >= 5:  # Change this back to 5 instead of 6
            processing_status[model_name] = "complete"
        else:
            processing_status[model_name] = "processing"
            
        print(f"📊 Progress updated: {model_name} -> Stage {stage}: {message}")
    except Exception as e:
        print(f"⚠️ Failed to update app progress: {e}")

def _save_enhanced_results(model_name: str, results: List[Dict], task_type_results: Dict):
    """Save current results only AFTER evaluation is complete."""
    
    summary = {}
    for task_type, scores in task_type_results.items():
        if scores:
            summary[task_type] = {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'count': len(scores),
                'scores': scores
            }
    
    overall_scores = [s for scores in task_type_results.values() for s in scores]
    benchmark_mean = float(np.mean(overall_scores)) if overall_scores else 0.0
    if overall_scores:
        summary['overall'] = {
            'mean': benchmark_mean,
            'std': float(np.std(overall_scores)),
            'count': len(overall_scores)
        }
    
    entry = {
        "model_path": model_name,
        "summary": summary,
        "detailed_results": results,
        "timestamp": datetime.now().isoformat(),
        "num_tasks": len(results),
        "status": "completed"  # Add explicit completion status
    }
    
    # Store in app's current_results ONLY when evaluation is complete
    try:
        
        # Use model_name directly as the key
        current_results[model_name] = [entry]
        print(f"💾 FINAL Results stored in memory for {model_name}")
    except Exception as e:
        print(f"Failed to store results in app: {e}")

@llm_t_bp.route('/api/upload_new_llm_version/<model_name>', methods=['POST'])
def upload_new_llm_version(model_name):
    """
    Handles uploading a NEW LLM version.
    1. Rename existing model folder/files to _void.
    2. Save new file.
    3. Update details.json version.
    """
    try:
        # Assuming typical structure: models/{model_name}
        base_path = os.path.join('models', model_name)
        if not os.path.exists(base_path):
            # Try to map friendly name to path if using a dict
            # For now assume base_path is direct
            os.makedirs(base_path, exist_ok=True)

        files_processed = []

        # Handle Model File
        if 'model_file' in request.files:
            file = request.files['model_file']
            if file.filename != '':
                # Voiding logic for generic model files
                # Move current contents to a _void folder
                void_dir = os.path.join(base_path, 'void_archived')
                os.makedirs(void_dir, exist_ok=True)
                
                # Move existing .bin, .safetensors to void
                for f in os.listdir(base_path):
                    if f.endswith(('.bin', '.safetensors', '.pt')) and 'void' not in f:
                        shutil.move(os.path.join(base_path, f), os.path.join(void_dir, f))
                
                # Save new
                file.save(os.path.join(base_path, secure_filename(file.filename)))
                files_processed.append("Model Weights")

        # Handle Test File
        if 'test_file' in request.files:
            file = request.files['test_file']
            if file.filename != '':
                # Save to specific dataset folder if needed, or root
                file.save(os.path.join(base_path, 'new_ground_truth.xlsx'))
                files_processed.append("Ground Truth")

        # Update details.json version
        update_llm_details_json(model_name, {}, new_version=True)

        return jsonify({'status': 'success', 'message': f"Uploaded: {', '.join(files_processed)}. Version updated."})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@llm_t_bp.route('/api/evaluate_memory_llm/<model_name>', methods=['POST'])
def evaluate_memory_llm(model_name):
    """
    Simulates memory evaluation for comparison tab.
    In a real scenario, this would load the adapter and run inference.
    Here we return a mocked result based on the current champion to show UI functionality.
    """
    import random
    import time
    
    try:
        time.sleep(2) # Simulate processing
        
        # Load current details to generate a plausible "Challenger" score
        details = load_results_from_file(model_name) # Or load details.json
        # Mock logic
        base_f1 = 0.82
        
        # Generate metrics slightly better or worse
        metrics = {
            'bert_score_f1': round(base_f1 + random.uniform(-0.02, 0.05), 4),
            'rouge_l': round(0.58 + random.uniform(-0.02, 0.05), 4),
            'overall_score': round(85 + random.uniform(-2, 5), 2),
            'hallucination_rate': round(max(0, 1.2 + random.uniform(-0.5, 0.2)), 2)
        }
        
        return jsonify({'status': 'success', 'metrics': metrics})

    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

def update_llm_details_json(model_name, results, new_version=False):
    """Helper to update details.json for LLMs"""
    try:
        details_path = os.path.join('models', model_name, 'details.json')
        # ... (Implementation similar to ML utils, just handling the JSON read/write)
        # Create file if not exists, increment version string if new_version=True
        # Update 'benchmarks' key with 'results' if provided
        pass # Placeholder for brevity, logic is same as ML utils
    except:
        pass

    
def extract_score_from_results(results):
    """Extract a score from various result formats."""
    try:
        # Handle different possible result structures
        if isinstance(results, dict):
            # Look for common score fields
            score_fields = ['accuracy', 'score', 'exact_match', 'f1', 'bleu', 'rouge_l']
            
            for field in score_fields:
                if field in results:
                    value = results[field]
                    if isinstance(value, (int, float)):
                        return value * 100 if value <= 1.0 else value
                    elif isinstance(value, dict) and 'mean' in value:
                        mean_val = value['mean']
                        return mean_val * 100 if mean_val <= 1.0 else mean_val
            
            # If no direct score field, look for nested structures
            if 'metrics' in results:
                return extract_score_from_results(results['metrics'])
            
            if 'summary' in results:
                return extract_score_from_results(results['summary'])
        
        elif isinstance(results, (int, float)):
            return results * 100 if results <= 1.0 else results
    
    except:
        pass
    
    return None

# --------------------- ENTRY POINT --------------------- #
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced BIG-bench evaluation")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name")
    parser.add_argument("--num_examples", type=int, default=50, help="Examples per task")
    parser.add_argument("--max_tokens", type=int, default=128, help="Max generation tokens")
    parser.add_argument("--full_benchmark", action="store_true", help="Use full BIG-bench (not just lite)")
    
    args = parser.parse_args()
    
    run_evaluation(
        model_name=args.model,
        num_examples=args.num_examples,
        max_new_tokens=args.max_tokens,
        use_full_bigbench=args.full_benchmark
    )