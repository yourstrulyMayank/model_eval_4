import os
from flask import Blueprint, render_template,  redirect, url_for, flash, Flask, render_template, request, redirect, url_for, jsonify, flash, send_file, make_response, send_from_directory, session, current_app
import PyPDF2
import pandas as pd
import requests
import json
from werkzeug.utils import secure_filename
from io import BytesIO


tcg_bp = Blueprint('tcg', __name__)
# Configuration
TCG_UPLOAD_FOLDER = 'uploads/test_case_generation/models'
ALLOWED_MODEL_CARD = {'pdf'}
ALLOWED_TEST_FORMAT = {'csv', 'xlsx', 'xls'}
os.makedirs(TCG_UPLOAD_FOLDER, exist_ok=True)
OLLAMA_API_URL = "http://localhost:11434/api/generate" 
EXISTING_MODELS = ['Compliance Assist', 'Wealth Assist', 'Capital Risk']
model_data = {
    'Compliance Assist': {
        'model_card': 'Compliance Assist helps with regulatory compliance tasks',
        'test_format': r'uploads/test_case_generation/models/compliance_assist/compliance_assist_sample.csv'
    },
    'Wealth Assist': {
        'model_card': 'Wealth Assist provides wealth management guidance',
        'test_format': r'uploads/test_case_generation/models/wealth_assist/wealth_assist_sample.csv'
    },
    'Capital Risk': {
        'model_card': 'Capital Risk assesses financial risk metrics',
        'test_format': r'uploads/test_case_generation/models/capital_risk/capital_risk_sample.csv'
    }
}


def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def extract_text_from_pdf(file_stream):
    """Extract text from a PDF file-like object."""
    text = ""
    pdf_reader = PyPDF2.PdfReader(file_stream)  # Use the file-like object directly
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_pdf_path(pdf_path):
    """Extract text from PDF file path"""
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def read_test_format_from_file(file_obj, filename):
    """Read test format CSV/XLSX from file object (in-memory)"""
    if filename.endswith('.csv'):
        df = pd.read_csv(file_obj)
    else:
        df = pd.read_excel(file_obj)
    
    columns = df.columns.tolist()
    sample_rows = df.head(3).to_dict('records')
    
    return {
        'columns': columns,
        'sample_data': sample_rows,
        'total_rows': len(df)
    }

def read_test_format_from_path(file_path):
    """Read test format CSV/XLSX from file path"""
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)
    
    columns = df.columns.tolist()
    sample_rows = df.head(3).to_dict('records')
    
    return {
        'columns': columns,
        'sample_data': sample_rows,
        'total_rows': len(df)
    }

def get_model_data_from_uploads(model_name):
    """Get model card and test format from uploads folder for existing models"""
    # Normalize model name to filename (e.g., "Compliance Assist" -> "compliance_assist")
    # filename = model_name.lower().replace(' ', '_')
    
    # model_card_path = os.path.join(TCG_UPLOAD_FOLDER, f"{filename}_model_card.pdf")
    
    # Try different extensions for test format
    # test_format_path = None
    # for ext in ['csv', 'xlsx', 'xls']:
    #     path = os.path.join(TCG_UPLOAD_FOLDER, f"{filename}_test_format.{ext}")
    #     if os.path.exists(path):
    #         test_format_path = path
    #         break
    
    # if not os.path.exists(model_card_path) or not test_format_path:
    #     return None
    
    model_card_text = model_data[model_name]['model_card']#extract_text_from_pdf_path(model_card_path)
    test_format_info = read_test_format_from_path(model_data[model_name]['test_format'])#read_test_format_from_path(test_format_path)
    
    return {
        'model_card': model_card_text,
        'test_format': test_format_info
    }

def call_ollama_llm(prompt, model="llama3.2"):
    """Call Ollama API with the given prompt"""
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            },
            timeout=10000
        )
        response.raise_for_status()
        return response.json()['response']
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error calling Ollama API: {str(e)}")
    
@tcg_bp.route('/test-case-generation')
def test_case_generation():
    # Clear any temporary model data from session when page loads
    if 'temp_models' in session:
        session.pop('temp_models')
    return render_template('test_case_generation.html')

@tcg_bp.route('/upload-new-model', methods=['POST'])
def upload_new_model():
    try:
        if 'model_card' not in request.files or 'test_format' not in request.files:
            return jsonify({'error': 'Both model card and test format files are required'}), 400
        
        model_card_file = request.files['model_card']
        test_format_file = request.files['test_format']
        
        if model_card_file.filename == '' or test_format_file.filename == '':
            return jsonify({'error': 'No files selected'}), 400
        
        if not allowed_file(model_card_file.filename, ALLOWED_MODEL_CARD):
            return jsonify({'error': 'Model card must be a PDF file'}), 400
        
        if not allowed_file(test_format_file.filename, ALLOWED_TEST_FORMAT):
            return jsonify({'error': 'Test format must be CSV or XLSX file'}), 400
        
        # Generate model name from model card filename
        model_name = secure_filename(model_card_file.filename.rsplit('.', 1)[0])
        
        # Extract data directly from the FileStorage object
        model_card_text = extract_text_from_pdf(model_card_file.stream)  # Use .stream for in-memory file
        test_format_info = read_test_format_from_file(test_format_file.stream, test_format_file.filename)
        
        # Store in session (in-memory, will be cleared on refresh/session end)
        if 'temp_models' not in session:
            session['temp_models'] = {}
        
        session['temp_models'][model_name] = {
            'model_card': model_card_text,
            'test_format': test_format_info
        }
        session.modified = True
        
        return jsonify({
            'success': True,
            'model_name': model_name
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

from collections import OrderedDict

@tcg_bp.route('/generate-test-cases', methods=['POST'])
def generate_test_cases():
    try:
        data = request.json
        user_message = data.get('message')
        selected_model = data.get('model')
        
        if not selected_model:
            return jsonify({'error': 'No model selected'}), 400
        
        # Check if it's a temporary uploaded model
        model_info = None
        if 'temp_models' in session and selected_model in session['temp_models']:
            temp_model = session['temp_models'][selected_model]
            model_info = {
                'model_card': temp_model['model_card'],
                'test_format': temp_model['test_format']
            }
        # Check if it's an existing model
        elif selected_model in EXISTING_MODELS:
            model_info = get_model_data_from_uploads(selected_model)
            if not model_info:
                return jsonify({'error': f'Model files not found in uploads folder for {selected_model}'}), 400
        else:
            return jsonify({'error': 'Invalid model selected'}), 400
        
        model_card_text = model_info['model_card']
        test_format_info = model_info['test_format']
        
        # Create prompt for LLM
        prompt = f"""You are a test case generation assistant for the model: {selected_model}

        Model Information:
        {model_card_text[:1500]}

        Test Case Format:
        Columns: {', '.join(test_format_info['columns'])}

        Sample Test Cases:
        {json.dumps(test_format_info['sample_data'], indent=2)}

        User Request: {user_message}

        Generate 5 relevant test cases in the EXACT same format as the samples above. MAINTAIN COLUMN ORDER.
        IMPORTANT: Return ONLY a valid JSON array of objects, where each object represents one test case with the exact column names.
        Do not include any explanation or markdown formatting, just the raw JSON array.

        Example format:
        [
        {{"column1": "value1", "column2": "value2"}},
        {{"column1": "value3", "column2": "value4"}}
        ]"""

        # Call Ollama LLM
        llm_response = call_ollama_llm(prompt, model="llama3.2")
        
        # Try to parse JSON from response
        import re
        try:
            # Extract JSON array from response
            json_match = re.search(r'\[.*\]', llm_response, re.DOTALL)
            if json_match:
                table_data = json.loads(json_match.group())
                
                # Enforce original column order
                ordered_table_data = [
                    OrderedDict((col, row.get(col, "")) for col in test_format_info['columns'])
                    for row in table_data
                ]
                
                return jsonify({
                    'response': f'Generated {len(ordered_table_data)} test cases:',
                    'table_data': ordered_table_data,
                    'columns': test_format_info['columns']
                })
        except:
            pass
        
        # If JSON parsing fails, return as text
        return jsonify({
            'response': llm_response
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@tcg_bp.route('/download-test-cases', methods=['POST'])
def download_test_cases():
    try:
        data = request.json
        test_cases = data.get('test_cases', [])
        model_name = data.get('model_name', 'test_cases')
        column_order = data.get('column_order', [])
        
        if not test_cases:
            return jsonify({'error': 'No test cases to download'}), 400
        
        # Create DataFrame and enforce column order
        df = pd.DataFrame(test_cases)
        
        # Reorder columns if column_order is provided
        if column_order:
            # Only include columns that exist in both column_order and df
            ordered_columns = [col for col in column_order if col in df.columns]
            df = df[ordered_columns]
        
        # Generate CSV
        output = BytesIO()
        df.to_csv(output, index=False, encoding='utf-8')
        output.seek(0)
        
        response = make_response(output.getvalue())
        response.headers['Content-Type'] = 'text/csv'
        response.headers['Content-Disposition'] = f'attachment; filename="{model_name}_test_cases.csv"'
        
        return response
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500