from flask import Blueprint, render_template, redirect, url_for, json, request, jsonify, session, current_app
import os
import pandas as pd
import re
from routes.test_case_generation.test_case_generation import call_ollama_llm, get_model_data_from_uploads

agentic_bp = Blueprint('agentic', __name__)
UPLOAD_FOLDER = 'uploads'

@agentic_bp.route('/agentic-evaluation')
def agentic_evaluation():
    return render_template('agentic_evaluation.html')

@agentic_bp.route('/process-agentic-request', methods=['POST'])
def process_agentic_request():
    try:
        data = request.json
        user_message = data.get('message', '').lower()
        
        # 1. Handle Follow-up questions (e.g., Test Case Generation confirmation)
        if session.get('awaiting_testcase_response'):
            model_info = session.pop('awaiting_testcase_response')
            if any(word in user_message for word in ['yes', 'generate', 'new', 'create']):
                return jsonify({
                    'response': f'Generating new test cases for {model_info["model_name"]}...',
                    'trigger_testcase_generation': True,
                    'model_name': model_info['model_name']
                })
            else:
                # Default to custom ML evaluation with existing data
                return jsonify({
                    'response': 'Proceeding with existing test data...',
                    'redirect_url': url_for('ml_s_c.custom_ml', model_name='capital_risk', subcategory='supervised')
                })

        # 2. Powerful System Prompt for full app access
        classification_prompt = f"""
        Analyze the user's request for a Model Evaluation System.
        
        APP CAPABILITIES:
        - ML Supervised: Standard (MLFlow) and Custom (External Test Data).
        - LLM: Standard (BigBench) and Custom (User Prompts).
        - RISK: FRTB-SA (Standardized Approach).
        - TOOLS: Test Case Generation, Result Summarization.

        AVAILABLE MODELS:
        - "capital_risk" (ML)
        - "compliance_model" (LLM)
        - "wealth_advisory_model" (LLM)
        - "FRTB-SA" (Risk)

        User request: "{user_message}"

        Respond ONLY in JSON:
        {{
            "action": "evaluate" | "generate_test_cases" | "summarize" | "navigate",
            "model_type": "ml" | "llm" | "risk",
            "eval_mode": "standard" | "custom",
            "model_name": "Full Model Name",
            "target_page": "url_slug"
        }}
        """
        
        llm_response = call_ollama_llm(classification_prompt, model="llama3.2")
        json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
        
        if not json_match:
            return jsonify({'error': 'Could not parse intent.'})

        intent = json.loads(json_match.group())
        m_name = intent.get('model_name', '')
        name_map = {
            "compliance assist": "compliance_model",
            "wealth assist": "wealth_advisory_model"
        }
        m_name = name_map.get(m_name.lower(), m_name)
        
        m_type = intent.get('model_type')
        mode = intent.get('eval_mode', 'standard')

        # --- ROUTING ENGINE ---

        # Case A: Risk Models (FRTB-SA)
        if m_type == 'risk' or 'frtb' in user_message:
            return jsonify({
                'response': 'Navigating to Risk Evaluation for FRTB-SA...',
                'redirect_url': url_for('risk_frtbsa_bp.risk_frtbsa') # Matches your risk route
            })

        # Case B: ML Models
        if m_type == 'ml':
            if mode == 'custom':
                session['awaiting_testcase_response'] = {'model_name': m_name}
                return jsonify({
                    'response': f'I see you want a custom evaluation for {m_name}. Should I generate new test cases first?',
                    'awaiting_response': True
                })
            return jsonify({
                'response': f'Initiating standard ML evaluation for {m_name}...',
                'trigger_ml_standard': True,
                'model_name': m_name
            })

        # Case C: LLM Models
        if m_type == 'llm':
            endpoint = 'llm_t.evaluate_llm' if mode == 'standard' else 'llm_c.custom_llm'
            return jsonify({
                'response': f'Opening {mode} LLM evaluation for {m_name}...',
                'redirect_url': url_for(endpoint, model_name=m_name)
            })

        # Case D: Summary/Analytics
        if intent.get('action') == 'summarize':
            return jsonify({
                'response': f'Generating summary for {m_name}...',
                'trigger_summary': True,
                'model_type': f"{m_type}_{mode}",
                'model_name': m_name
            })

        return jsonify({'response': "I understand what you want, but I need more specifics about the model or evaluation type."})

    except Exception as e:
        return jsonify({'error': str(e)}), 500