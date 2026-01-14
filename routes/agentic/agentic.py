from flask import Blueprint, render_template,  redirect, url_for, flash, Flask, render_template, request, redirect, url_for, jsonify, flash, send_file, make_response, send_from_directory, session, current_app
import json
from routes.llm.tool.llm_tool_bigbench_utils import  load_results_from_file
from routes.test_case_generation.test_case_generation import call_ollama_llm, get_model_data_from_uploads
import os
import pandas as pd

UPLOAD_FOLDER = 'uploads'
agentic_bp = Blueprint('agentic', __name__)

current_results = {}

@agentic_bp.route('/agentic-evaluation')
def agentic_evaluation():
    return render_template('agentic_evaluation.html')


@agentic_bp.route('/process-agentic-request', methods=['POST'])
def process_agentic_request():
    """Process natural language requests using LLM to route to appropriate evaluation"""
    try:
        data = request.json
        user_message = data.get('message', '').lower()

        # Check if this is a response to test case generation question
        if session.get('awaiting_testcase_response'):
            model_info = session.pop('awaiting_testcase_response')

            # Check if user wants test case generation
            if 'yes' in user_message or 'generate' in user_message or 'new' in user_message:
                return jsonify({
                    'response': f'Generating test cases for {model_info["model_name"]}...',
                    'log': 'Initiating test case generation',
                    'log_type': 'processing',
                    'trigger_testcase_generation': True,
                    'model_name': model_info['model_name']
                })
            else:
                # User said no, proceed directly to custom evaluation
                redirect_url = url_for('ml_s_c.custom_ml',
                                      model_name='capital_risk')
                return jsonify({
                    'response': f'Starting custom evaluation for {model_info["model_name"]} with existing test data...',
                    'log': 'Proceeding to custom evaluation without test generation',
                    'log_type': 'success',
                    'redirect_url': redirect_url,
                    'model_name': model_info['model_name']
                })

        # Improved classification prompt
        classification_prompt = f"""Analyze this user request and extract information.

        Available models and their types:
        - "compliance_model" - LLM model
        - "wealth_advisory_model" - LLM model
        - "capital_risk" - ML model

        User request: "{user_message}"

        Determine:
        1. Is this "standard" or "custom" evaluation?
        2. Which model name from the list above?
        3. Is it "ml" or "llm" type?

        Respond ONLY with valid JSON, no other text:
        {{"evaluation_type": "standard", "model_name": "capital_risk", "model_type": "ml"}}"""

        llm_response = call_ollama_llm(classification_prompt, model="llama3.3")

        import re
        json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
        if json_match:
            intent = json.loads(json_match.group())

            eval_type = intent.get('evaluation_type')
            model_name = intent.get('model_name')
            model_type = intent.get('model_type')

            # Validate model name
            valid_models = {
                'ml': ['capital_risk'],
                'llm': ['compliance_model', 'wealth_advisory_model']
            }

            if model_name not in valid_models.get(model_type, []):
                return jsonify({
                    'error': f'Invalid model "{model_name}" for type {model_type}',
                    'log': 'Model name validation failed',
                    'log_type': 'error'
                })

            # Store in session for auto-trigger
            session['agentic_request'] = {
                'model_name': model_name,
                'eval_type': eval_type,
                'model_type': model_type,
                'auto_trigger': True
            }

            # For ML custom, ASK first about test case generation
            if model_type == 'ml' and eval_type == 'custom':
                session['awaiting_testcase_response'] = {
                    'model_name': model_name,
                    'model_type': model_type
                }
                return jsonify({
                    'response': f'Would you like me to generate new test cases for {model_name}, or use existing test data?',
                    'log': 'Asking user about test case generation',
                    'log_type': 'question',
                    'awaiting_response': True
                })

            # Generate redirect URL for other cases
            if model_type == 'ml':
                if eval_type == 'standard':
                    return jsonify({
                        'response': f'Starting {eval_type} evaluation for {model_name}...',
                        'log': f'Agent routing to {eval_type} evaluation workflow',
                        'log_type': 'success',
                        'redirect_url': None,
                        'trigger_ml_standard': True,
                        'model_name': model_name
                    })
            else:  # llm
                if eval_type == 'standard':
                    redirect_url = url_for('llm_t.evaluate_llm', model_name=model_name)
                else:
                    redirect_url = url_for('llm_c.custom_llm', model_name=model_name)

            return jsonify({
                'response': f'Starting {eval_type} evaluation for {model_name}...',
                'log': f'Agent routing to {eval_type} evaluation workflow for {model_name}',
                'log_type': 'success',
                'redirect_url': redirect_url
            })

        return jsonify({
            'error': 'Could not understand the request. Please specify evaluation type and model name.',
            'log': 'Failed to parse user intent',
            'log_type': 'error'
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'log': f'Error in agentic processing: {str(e)}',
            'log_type': 'error'
        }), 500


@agentic_bp.route('/agentic_generate_testcases/<model_name>', methods=['POST'])
def agentic_generate_testcases(model_name):
    """Generate test cases across ALL categories and combine them into one table for ML evaluation"""
    try:
        print(f"\n{'='*60}")
        print(f"Starting test case generation for: {model_name}")
        print(f"{'='*60}\n")
        
        # Determine if it's an existing model and if it's an LLM
        is_existing_llm = model_name in ['Compliance Assist', 'Wealth Assist']
        
        print(f"Model type: {'LLM' if is_existing_llm else 'ML'}")
        
        model_info = get_model_data_from_uploads('Capital Risk', is_llm=is_existing_llm)
        if not model_info:
            print(f"❌ Error: Model data not found for {model_name}\n")
            return jsonify({'error': f'Model data not found for {model_name}'}), 400
            
        # ML models must have test_format
        if not is_existing_llm and 'test_format' not in model_info:
            print(f"❌ Error: Test format data not found for {model_name}\n")
            return jsonify({'error': f'Test format data not found for {model_name}'}), 400

        print(f"✓ Model data loaded successfully")
        
        model_card_text = model_info['model_card']
        test_format_info = model_info['test_format']
        all_combined_test_cases = []
        
        total_categories = len(ML_TEST_CATEGORIES)
        print(f"✓ Processing {total_categories} categories\n")

        # Iterate through all categories defined in the test_case_generation script
        for idx, (category_key, category_info) in enumerate(ML_TEST_CATEGORIES.items(), 1):
            print(f"[{idx}/{total_categories}] Generating test cases for: {category_info['name']}")
            print(f"    Focus areas: {', '.join(category_info['focus_areas'])}")
            
            prompt = f"""You are a test case generation assistant for the ML model: {model_name}

Model Information:
{model_card_text[:1500]}

Test Case Format:
Columns: {', '.join(test_format_info['columns'])}

Sample Test Cases:
{json.dumps(test_format_info['sample_data'], indent=2)}

Category: {category_info['name']}
Description: {category_info['description']}
Focus Areas: {', '.join(category_info['focus_areas'])}

Generate 5 test cases for this category in the EXACT same format as the samples above. MAINTAIN COLUMN ORDER.
IMPORTANT: Return ONLY a valid JSON array of objects with the exact column names. Do not include any explanation or markdown."""

            print(f"    Calling LLM...")
            llm_response = call_ollama_llm(prompt, model="llama3.2")
            
            import re
            json_match = re.search(r'\[.*\]', llm_response, re.DOTALL)
            if json_match:
                try:
                    category_cases = json.loads(json_match.group())
                    
                    # Ensure each test case follows the correct column order
                    ordered_cases = [
                        OrderedDict((col, case.get(col, "")) for col in test_format_info['columns'])
                        for case in category_cases
                    ]
                    
                    all_combined_test_cases.extend(ordered_cases)
                    print(f"    ✓ Generated {len(ordered_cases)} test cases")
                    print(f"    Total test cases so far: {len(all_combined_test_cases)}\n")
                except Exception as e:
                    print(f"    ❌ Error parsing category {category_key}: {e}\n")
                    continue
            else:
                print(f"    ❌ Failed to extract JSON from LLM response\n")

        if not all_combined_test_cases:
            print(f"❌ Failed to generate any valid test cases across categories\n")
            return jsonify({'error': 'Failed to generate any valid test cases across categories'}), 500

        print(f"\n{'='*60}")
        print(f"Test case generation complete!")
        print(f"Total test cases generated: {len(all_combined_test_cases)}")
        print(f"{'='*60}\n")

        # Create combined DataFrame with proper column ordering
        print("Creating DataFrame and ensuring column order...")
        df = pd.DataFrame(all_combined_test_cases)
        
        # Ensure column order matches the model's required format
        ordered_columns = [col for col in test_format_info['columns'] if col in df.columns]
        df = df[ordered_columns]
        print(f"✓ DataFrame created with {len(df)} rows and {len(ordered_columns)} columns")

        # Save to the specific model folder for the custom evaluation route
        model_folder_name = model_name.lower().replace(' ', '_')
        upload_dir = os.path.join(UPLOAD_FOLDER, model_folder_name)
        os.makedirs(upload_dir, exist_ok=True)

        csv_path = os.path.join(upload_dir, 'test.csv')
        df.to_csv(csv_path, index=False)
        
        print(f"✓ Test cases saved to: {csv_path}")
        print(f"{'='*60}\n")

        return jsonify({
            'status': 'success',
            'csv_path': csv_path,
            'test_cases_count': len(all_combined_test_cases),
            'categories_processed': list(ML_TEST_CATEGORIES.keys()),
            'columns': ordered_columns
        })

    except Exception as e:
        print(f"\n❌ FATAL ERROR: {str(e)}\n")
        return jsonify({'error': str(e)}), 500

@agentic_bp.route('/agentic_summarize_results/<model_type>/<model_name>', methods=['POST'])
def agentic_summarize_results(model_type, model_name):
    """Generate AI summary of evaluation results"""
    try:
        results = None

        # Get results based on model type
        if model_type == 'ml_custom':
            from routes.ml.supervised.custom.ml_supervised_custom_utils import custom_evaluation_results
            results = custom_evaluation_results.get(f"{model_name}_ml", {})
        elif model_type == 'ml_standard':
            from routes.ml.supervised.tool.mlflow.ml_supervised_tool_mlflow_utils import get_ml_results
            results = get_ml_results(model_name)
        elif model_type == 'llm_custom':
            results = current_results.get(f"{model_name}_custom", {})
        elif model_type == 'llm_standard':

            results_data = load_results_from_file(model_name)
            results = results_data[0] if results_data else {}

        if not results or results.get('error'):
            return jsonify({'error': 'No results available to summarize'}), 400

        # Create comprehensive summary prompt
        summary_prompt = f"""You are an AI model evaluation expert. Analyze these evaluation results and provide a clear, insightful summary.

        Model: {model_name}
        Evaluation Type: {model_type.replace('_', ' ').title()}

        Results Data:
        {json.dumps(results, indent=2, default=str)[:3000]}

        Provide a professional summary covering:
        1. Overall Performance (highlight key metrics)
        2. Strengths (what the model does well)
        3. Areas for Improvement (weaknesses or failures)
        4. Recommendations (actionable insights)

        Keep it concise (4-6 sentences) and use bullet points where appropriate."""

        # Call LLM for summary
        summary = call_ollama_llm(summary_prompt, model="llama3.2")

        return jsonify({
            'status': 'success',
            'summary': summary
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500