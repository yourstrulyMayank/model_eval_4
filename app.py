# app.py - Enhanced Flask App
from flask import Flask, render_template, request, redirect, url_for, jsonify, flash, send_file, make_response, send_from_directory, session
import os
import threading
import json
from datetime import datetime
from io import BytesIO
import base64
import requests
import pandas as pd
import matplotlib.pyplot as plt
import os
import traceback
import logging
import openpyxl
from collections import defaultdict
import contextlib
import threading
import uuid
from openai import OpenAI
from werkzeug.utils import secure_filename
import PyPDF2
# from weasyprint import HTML, CSS
from urllib.parse import unquote
import numpy as np
app = Flask(__name__)


## ML Imports ##
from routes.ml.supervised.tool.mlflow.ml_supervised_tool_mlflow import ml_s_t_mlflow_bp
from routes.ml.supervised.custom.ml_supervised_custom import ml_s_c_bp
## ML Blueprints ##
app.register_blueprint(ml_s_t_mlflow_bp)
app.register_blueprint(ml_s_c_bp)


## LLM Imports ## 
from routes.llm.tool.llm_tool_bigbench_utils import llm_t_bp
from routes.llm.custom.custom_evaluate_llm import llm_c_bp
## LLM Blueprints ##
app.register_blueprint(llm_t_bp)
app.register_blueprint(llm_c_bp)


## Test Case Generation Imports ##
from routes.test_case_generation.test_case_generation import tcg_bp 
## Test Case Generation Blueprints ##
app.register_blueprint(tcg_bp)


## Agentic Imports ##
from routes.agentic.agentic import agentic_bp  
## Agentic Blueprints ##
app.register_blueprint(agentic_bp)


## Risk Imports #
from routes.risk import risk_blueprints

## Risk Blueprints ##
for bp in risk_blueprints:
    app.register_blueprint(bp)

# Add logging configuration
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# PORT = 8056


app.secret_key = 'your-secret-key-here'
model_base_path = "models"

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf', 'xlsx', 'xls', 'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
# Create upload directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# Add this global variable with other globals


js_function = '''
<script>
function setBenchmark(type, index) {
    const dropdown = document.getElementById('benchmark-' + type + '-' + index);
    const hiddenInput = document.getElementById('benchmark-input-' + type + '-' + index);
    if (dropdown && hiddenInput) {
        hiddenInput.value = dropdown.value;
    }
}
</script>
'''
@app.route('/')
def index():
    return render_template("index.html")      

@app.route('/model-evaluation')
def model_evaluation():
    """Model evaluation page with tabs for standard and custom evaluation"""
    model_name = request.args.get('model')
    model_type = request.args.get('type')
    
    if not model_name or not model_type:
        flash("Invalid model selection")
        return redirect(url_for('index'))
    
    # Store model data in session for access across tabs
    session['current_model'] = {
        'name': model_name,
        'type': model_type
    }
    
    return render_template('model_evaluation.html', 
                         model_name=model_name, 
                         model_type=model_type)


@app.route('/api/model-metadata/<model_name>')
def get_model_metadata(model_name):
    """Load model metadata from a text file or database"""
    try:
        # Example: Load from a JSON/text file
        metadata_file = f"model_metadata/{model_name.lower().replace(' ', '_')}.json"
        
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            # Default metadata
            metadata = {
                'version': 'v1.0.0',
                'last_updated': datetime.now().strftime('%Y-%m-%d'),
                'framework': 'PyTorch' if 'llm' in model_name.lower() else 'Scikit-learn',
                'status': 'Active'
            }
        
        return jsonify(metadata)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/check_ml_status/<model_name>')
def check_ml_status(model_name):
    """Check ML evaluation status"""
    try:
        # Import your ML status checking function
        from routes.ml.supervised.tool.mlflow.ml_supervised_tool_mlflow import get_evaluation_status
        
        status = get_evaluation_status(model_name)
        return jsonify(status)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

    
@app.route('/risk/<risk_model>')
def risk_model_page(risk_model):
    # Render template risk_<modelname>.html, e.g. risk_frtbsa.html
    template_name = f"risk_{risk_model}.html"
    try:
        return render_template(template_name, model_name=risk_model.upper())
    except Exception:
        return f"Risk model page for '{risk_model}' not found.", 404

if __name__ == '__main__':
    app.run(debug=True)