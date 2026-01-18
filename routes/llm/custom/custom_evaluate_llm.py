# custom_evaluate_llm.py
import os
import pandas as pd
from flask import Blueprint, render_template,  redirect, url_for, flash, Flask, render_template, request, redirect, url_for, jsonify, flash, send_file, make_response, send_from_directory, session, current_app
import threading
from datetime import datetime
import traceback
import glob
import requests
from tqdm import tqdm
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import re
import nltk
import torch
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import util
from bert_score import score as bert_score
from rouge_score import rouge_scorer
import spacy
import PyPDF2
import pymupdf
# import fitz  # PyMuPDF
import faiss
import numpy as np
import requests

from ollama import Client as ollama_client
llm_c_bp = Blueprint('llm_c', __name__)
current_results = {}
MODELS = {
    "wealth_advisory_model": "models/wealth_advisory",
    "compliance_model": "models/compliance"
}

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("Warning: spaCy model not found. Entity overlap will be disabled.")
    nlp = None

# Global configuration
EMBED_MODEL = SentenceTransformer("models/custom_embedding")
# OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "mistral:7b"
# CHROMA_DB_PATH = "./chroma_db"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
# Global progress tracker
progress_tracker = {}
progress_lock = threading.Lock()

# Initialize Ollama client
cli = ollama_client(host='http://10.177.213.115:11434')

# Global configuration
EMBED_MODEL = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda:0")
#EMBED_MODEL = SentenceTransformer("models/custom_embedding")
# OLLAMA_MODEL = "mistral:7b"
CHROMA_DB_PATH = "./chroma_db"
processing_status = {}  # Track per-model status
evaluation_progress = {}  # Track detailed progress

# --- Configuration ---
PDF_PATH_WEALTH = "/home/poc_boa/model_evaluation_4/uploads/compliance/Fedility_portfolio_statement.pdf"
PDF_PATH_COMPLIANCE = "/home/poc_boa/model_evaluation_4/uploads/wealth_advisory/BaselIII_3000_Transactions_Full.pdf"
MODEL_NAME = "mistral:7b-instruct"
OLLAMA_URL = "http://localhost:11434/api/chat"
CHUNK_SIZE = 300  # characters
CHUNK_OVERLAP = 50
print('All import done')
UPLOAD_FOLDER = 'uploads'
@llm_c_bp.route('/custom_llm/<model_name>')
def custom_llm(model_name):
    if model_name not in MODELS:
        flash(f"Unknown model: {model_name}")
        return redirect(url_for('index'))

    # Get current evaluation results only
    results = current_results.get(f"{model_name}_custom", {})

    return render_template('custom_llm.html',
                         model_name=model_name,
                         evaluation_results=results)

@llm_c_bp.route('/run_custom_evaluation/<model_name>', methods=['POST'])
def run_custom_evaluation_route(model_name):
    if model_name not in MODELS:
        return jsonify({'error': 'Unknown model'}), 400

    try:
        # Import custom evaluator
        upload_dir = os.path.join(UPLOAD_FOLDER)
        # For custom_llm, get model path from custom_models folder
        # Determine model path based on evaluation type
        wealth_advisory_dir = os.path.join(upload_dir, 'wealth_advisory')
        compliance_dir = os.path.join(upload_dir, 'compliance')

        if os.path.exists(wealth_advisory_dir):
            model_path = f"custom_models/wealth_advisory/{model_name}"
        elif os.path.exists(compliance_dir):
            model_path = f"custom_models/compliance/{model_name}"
        else:
            model_path = MODELS[model_name]  # Fallback to original MODELS dict
        upload_dir = os.path.join(UPLOAD_FOLDER)

        if not os.path.exists(upload_dir):
            return jsonify({'error': 'Upload directory not found'}), 400

        # Check if wealth_advisory or compliance folder exists
        wealth_advisory_dir = os.path.join(upload_dir, 'wealth_advisory')
        compliance_dir = os.path.join(upload_dir, 'compliance')

        if not os.path.exists(wealth_advisory_dir) and not os.path.exists(compliance_dir):
            return jsonify({'error': 'Neither wealth_advisory nor compliance folder found in uploads'}), 400

        # Set processing status
        processing_status[f"{model_name}_custom"] = "processing"

        def background_evaluation():
            try:
                print(f"Starting custom RAG evaluation with NLP metrics for {model_name}")
                # Run custom evaluation with NLP pipeline
                results = run_custom_evaluation(model_name, model_path, upload_dir)
                print(f"Custom evaluation with NLP metrics completed for {model_name}")

                # Store in current results only
                current_results[f"{model_name}_custom"] = results
                processing_status[f"{model_name}_custom"] = "complete"

            except Exception as e:
                print(f"Custom evaluation error for {model_name}: {e}")
                import traceback
                traceback.print_exc()

                processing_status[f"{model_name}_custom"] = "error"
                current_results[f"{model_name}_custom"] = {"error": str(e)}

        # Run in background
        threading.Thread(target=background_evaluation, daemon=True).start()

        return jsonify({'status': 'started', 'message': 'Evaluation with NLP metrics started successfully'})

    except Exception as e:
        print(f"Error starting evaluation: {e}")
        processing_status[f"{model_name}_custom"] = "error"
        return jsonify({'error': f'Error starting evaluation: {str(e)}'}), 500

@llm_c_bp.route('/clear_custom_results/<model_name>', methods=['POST'])
def clear_custom_results(model_name):
    """Clear custom evaluation results for a model."""
    try:
        # Clear from current results only
        custom_key = f"{model_name}_custom"
        if custom_key in current_results:
            del current_results[custom_key]

        # Clear processing status
        if custom_key in processing_status:
            del processing_status[custom_key]

        # Clear progress tracking

        clear_progress(model_name)

        return jsonify({'status': 'success', 'message': 'Results cleared successfully'})

    except Exception as e:
        print(f"Error clearing results for {model_name}: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@llm_c_bp.route('/check_custom_status/<model_name>')
def check_custom_status(model_name):
    status_key = f"{model_name}_custom"
    status = processing_status.get(status_key, "not_started")
    results = current_results.get(status_key, {})

    # Get progress information
    try:

        progress_info = get_progress(model_name)
    except:
        progress_info = {'stage': 0, 'message': 'Not started'}

    response_data = {
        "status": status,
        "results": results,
        "progress": progress_info,
        "timestamp": datetime.now().isoformat()
    }

    return jsonify(response_data)

@llm_c_bp.route('/download_custom_excel/<model_name>')
def download_custom_excel(model_name):
    """Download custom evaluation results as Excel file with NLP metrics."""
    if model_name not in MODELS:
        flash("Model not found.")
        return redirect(url_for('index'))

    try:
        # Get current evaluation results only
        results = current_results.get(f"{model_name}_custom", {})

        if not results or results.get('error'):
            flash("No evaluation results found for this model.")
            return redirect(url_for('custom_llm', model_name=model_name))

        import pandas as pd
        from io import BytesIO

        output = BytesIO()

        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Main results sheet (UI compatible format)
            if 'ground_truth_comparison' in results:
                comparison_data = results['ground_truth_comparison']

                df_results = pd.DataFrame([
                    {
                        'Prompt': item['prompt'],
                        'Ground_Truth_Actual': item['actual'],
                        'Model_Extracted': item['extracted'],
                        'Confidence_Score': item['score'],
                        'Test_Grade': item['grade'],
                        'Status': 'Pass' if item['grade'] == '✅ Pass' else 'Fail'
                    }
                    for item in comparison_data
                ])

                df_results.to_excel(writer, sheet_name='Evaluation_Results', index=False)

            # Detailed NLP metrics sheet
            if 'detailed_nlp_results' in results:
                nlp_df = pd.DataFrame(results['detailed_nlp_results'])
                # Remove columns that might not be needed for Excel export
                columns_to_exclude = ['GT_Entities', 'Pred_Entities']
                nlp_df = nlp_df.drop(columns=[col for col in columns_to_exclude if col in nlp_df.columns])
                nlp_df.to_excel(writer, sheet_name='Detailed_NLP_Metrics', index=False)

            # Summary sheet
            summary_data = {
                'Metric': [
                    'Model Name', 'Evaluation Type', 'Evaluation Date', 'Total Tests',
                    'Tests Passed', 'Tests Failed', 'Overall Score (%)',
                    'Success Rate (%)', 'Average Score', 'Highest Score', 'Lowest Score',
                    'Median Score', 'Standard Deviation'
                ],
                'Value': [
                    results.get('model_name', model_name),
                    results.get('evaluation_type', 'N/A'),
                    results.get('timestamp', 'N/A'),
                    results.get('total_tests', 0),
                    results.get('pass_count', 0),
                    results.get('fail_count', 0),
                    round(results.get('overall_score', 0), 2),
                    round(results.get('success_rate', 0), 2),
                    round(results.get('average_score', 0), 2),
                    round(results.get('summary_statistics', {}).get('highest_score', 0), 2),
                    round(results.get('summary_statistics', {}).get('lowest_score', 0), 2),
                    round(results.get('summary_statistics', {}).get('median_score', 0), 2),
                    round(results.get('summary_statistics', {}).get('std_deviation', 0), 2)
                ]
            }

            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(writer, sheet_name='Summary', index=False)

        output.seek(0)

        response = make_response(output.getvalue())
        response.headers['Content-Type'] = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        response.headers['Content-Disposition'] = f'attachment; filename="{model_name}_custom_nlp_evaluation_results.xlsx"'

        return response

    except Exception as e:
        print(f"Error generating Excel file: {e}")
        flash(f"Error generating Excel file: {str(e)}")
        return redirect(url_for('custom_llm', model_name=model_name))


def update_progress(model_name, stage, message):
    """Update progress for a specific model evaluation."""
    with progress_lock:
        if model_name not in progress_tracker:
            progress_tracker[model_name] = {}
        progress_tracker[model_name].update({
            'stage': stage,
            'message': message,
            'timestamp': datetime.now().isoformat()
        })
    print(f"📊 Progress Update - {model_name}: Stage {stage} - {message}")


def get_progress(model_name):
    """Get current progress for a model."""
    with progress_lock:
        return progress_tracker.get(model_name, {'stage': 0, 'message': 'Not started'})


def clear_progress(model_name):
    """Clear progress tracking for a specific model."""
    with progress_lock:
        if model_name in progress_tracker:
            del progress_tracker[model_name]
    print(f"🧹 Cleared progress tracking for {model_name}")


# def get_embedding(text):
    # """Generate embedding for text using custom embedding model"""
    # return EMBED_MODEL.encode(text).tolist()


# def ingest_documents(folder_path, collection_name="docs"):
    # """Ingest documents from a folder into the vector database"""
    # try:
        # client = chromadb.Client(Settings(persist_directory=CHROMA_DB_PATH))

        # # # Delete existing collection if it exists
        # # try:
        # # client.delete_collection(collection_name)
        # # print(f"🗑️ Deleted existing collection: {collection_name}")
        # # except:
        # # pass

        # collection = client.get_or_create_collection(collection_name)
        # print("MASTER CHECKKKKKKKK--------", collection)

        # # Get all PDF files in the folder
        # files = glob.glob(os.path.join(folder_path, "*.pdf"))
        # if not files:
            # print(f"No PDF files found in {folder_path}")
            # return False

        # for file in tqdm(files, desc="Ingesting documents"):
            # try:
                # # For now, we'll assume PDF content is extracted elsewhere
                # # In a real implementation, you'd use PyPDF2 or similar
                # with open(file, "r", encoding="utf-8", errors='ignore') as f:
                    # text = f.read()

                # # Split text into chunks (500 characters each)
                # chunks = [text[i:i + 500] for i in range(0, len(text), 500)]

                # for idx, chunk in enumerate(chunks):
                    # if chunk.strip():  # Only add non-empty chunks
                        # emb = get_embedding("passage: " + chunk)
                        # collection.add(
                            # documents=[chunk],
                            # embeddings=[emb],
                            # ids=[f"{os.path.basename(file)}_{idx}"]
                        # )
            # except Exception as e:
                # print(f"Error processing file {file}: {str(e)}")
                # continue

        # # client.persist()
        # print("Document ingestion completed successfully.")
        # return True

    # except Exception as e:
        # print(f"Error during ingestion: {str(e)}")
        # return False


# def query_documents(question, collection_name="docs", num_results=4):
    # """Query the document collection with a question"""
    # try:
        # print("In query Comuments---------> ", collection_name)

        # # def load_document(file_path):
        # #     with open(file_path, 'r', encoding='latin-1') as file:
        # #         return file.read()

        # if 'wealth' in collection_name.lower():
            # PDF_PATH = PDF_PATH_WEALTH

        # else 'compliance' in collection_name.lower():
            # PDF_PATH = PDF_PATH_COMPLIANCE

        # def load_document(file_path):
            # with open(file_path, 'rb') as f:
                # reader = PyPDF2.PdfReader(f)
                # return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

        # # Step 2: Send a prompt to Mistral via Ollama API
        # def query_ollama(prompt, model='mistral:7b-instruct', host='http://localhost:11434' ):
            # url = f"{host}/api/generate"
            # payload = {
                # "model": model,
                # "prompt": prompt,
                # "stream": False  # Set to True for streaming output
            # }
            # response = requests.post(url, json=payload)
            # response.raise_for_status()
            # return response.json()['response']

        # doc_content = load_document(PDF_PATH)
        # prompt = f"Answer the question based on the context below.\n\nContext:\n{doc_content}\n\nQuestion: {question}\nAnswer:"
        # result = query_ollama(prompt)
        # print("RESPONSE------------", result)
        # return result
    #         client = chromadb.Client(Settings(persist_directory=CHROMA_DB_PATH))
    #         collection = client.get_collection(collection_name)
    #
    #         # Generate embedding for the question
    #         q_emb = get_embedding("query: " + question)
    #
    #         # Query the collection
    #         results = collection.query(
    #             query_embeddings=[q_emb],
    #             n_results=num_results
    #         )
    #
    #         # Combine context from relevant chunks
    #         context = "\n\n".join([doc for doc in results["documents"][0]])
    #
    #         # Create prompt for the model
    #         prompt = f"Answer the question based on the context below.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    #         print(prompt)
    #
    #         # Get response from Ollama
    #         # response = requests.post(
    #             # f"{OLLAMA_URL}/api/generate",
    #             # json={"model": OLLAMA_MODEL, "prompt": prompt}
    #         # )
    #         print("CHECKERRRRRRRRRRRR----------", OLLAMA_MODEL)
    #         response = cli.chat(model=OLLAMA_MODEL, messages=[
    # 	{"role": "system", "content": "You are an wealth advisory agent and a ground truth generator. Generate the ground truth in a generalised mannner  "},
    # 	{"role": "user", "content": prompt}
    # ])
    #         answer = response['message']['content']
    #         #return None if "UNANSWERABLE" in answer or len(answer.strip()) < 20 else answer
    #         print("OLLAMA-------ANSWER", answer)
    #         print("OLLAMA-------RESPONSE", response)
    #
    #
    #         return answer

    # except Exception as e:
        # return f"Error: {str(e)}"


# === NLP Evaluation Functions ===
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[^a-z0-9\s\.,;:?!\'"\-\n]', '', text)
    return text.strip()


def evaluate_texts(gt_text, pred_text):
    gt_clean = preprocess_text(gt_text)
    pred_clean = preprocess_text(pred_text)

    gt_tokens = word_tokenize(gt_clean)
    pred_tokens = word_tokenize(pred_clean)
    gt_word_count = len(gt_tokens)
    pred_word_count = len(pred_tokens)

    # Embedding similarity
    embed_model = SentenceTransformer("all-mpnet-base-v2", device="cuda:0")
    gt_emb = embed_model.encode(gt_clean, convert_to_tensor=True)
    pred_emb = embed_model.encode(pred_clean, convert_to_tensor=True)
    cosine_sim = util.pytorch_cos_sim(gt_emb, pred_emb).item()

    # BERTScore
    P, R, F1 = bert_score([pred_clean], [gt_clean], lang='en', model_type='microsoft/deberta-xlarge-mnli')
    bert_precision = P[0].item()
    bert_recall = R[0].item()
    bert_f1 = F1[0].item()

    # ROUGE
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge_scores = rouge.score(gt_clean, pred_clean)
    rouge1 = rouge_scores['rouge1'].fmeasure
    rougeL = rouge_scores['rougeL'].fmeasure

    # Entity Overlap
    def extract_entities(text):
        if nlp is None:
            return set()
        return set(ent.text.lower() for ent in nlp(text).ents)

    ents_gt = extract_entities(gt_clean)
    ents_pred = extract_entities(pred_clean)
    entity_overlap = len(ents_gt & ents_pred) / max(len(ents_gt | ents_pred), 1) if ents_gt or ents_pred else 1.0

    # Length ratio and coverage
    length_ratio = pred_word_count / max(gt_word_count, 1)
    coverage_score = bert_recall * length_ratio
    composite_score = 0.5 * bert_f1 + 0.5 * coverage_score

    return {
        "GT_WordCount": gt_word_count,
        "Pred_WordCount": pred_word_count,
        "LengthRatio": round(length_ratio, 4),
        "CosineSimilarity": round(cosine_sim, 4),
        "BERTScore_Precision": round(bert_precision, 4),
        "BERTScore_Recall": round(bert_recall, 4),
        "BERTScore_F1": round(bert_f1, 4),
        "ROUGE-1_F": round(rouge1, 4),
        "ROUGE-L_F": round(rougeL, 4),
        "EntityOverlap": round(entity_overlap, 4),
        "CoverageScore": round(coverage_score, 4),
        "CompositeScore": round(composite_score, 4),
        "GT_Entities": list(ents_gt),
        "Pred_Entities": list(ents_pred)
    }


def decide_pass_fail(row):
    thresholds = {
        "CosineSimilarity": 0.75,
        "BERTScore_Precision": 0.70,
        "BERTScore_Recall": 0.30,
        "BERTScore_F1": 0.50,
        "ROUGE-1_F": 0.40,
        "ROUGE-L_F": 0.40,
        "EntityOverlap": 0.50,
        "CoverageScore": 0.30,
        "CompositeScore": 0.50,
        "LengthRatio": 0.40
    }
    for metric, threshold in thresholds.items():
        if row[metric] < threshold:
            return "❌ Fail"
    return "✅ Pass"


def failed_metrics(row):
    thresholds = {
        "CosineSimilarity": 0.75,
        "BERTScore_Precision": 0.70,
        "BERTScore_Recall": 0.30,
        "BERTScore_F1": 0.50,
        "ROUGE-1_F": 0.40,
        "ROUGE-L_F": 0.40,
        "EntityOverlap": 0.50,
        "CoverageScore": 0.30,
        "CompositeScore": 0.50,
        "LengthRatio": 0.40
    }
    return ", ".join([f"{m}<{thresholds[m]}" for m in thresholds if row[m] < thresholds[m]])


def run_evaluation_on_dataframe(input_df):
    records = []
    for _, row in input_df.iterrows():
        result = evaluate_texts(row['GroundTruth'], row['Predicted'])
        for key in result:
            row[key] = result[key]
        records.append(row)

    df = pd.DataFrame(records)
    df['PassFail'] = df.apply(decide_pass_fail, axis=1)
    df['FailedMetrics'] = df.apply(failed_metrics, axis=1)
    return df


def run_custom_evaluation(model_name, model_path, upload_dir):
    """Main function to run custom RAG-based evaluation with NLP metrics"""
    print(f"🚀 Starting custom RAG evaluation for model: {model_name}")

    try:
        # Stage 1: Loading models and initializing
        update_progress(model_name, 1, "Loading models and initializing...")

        # Determine evaluation type based on folder structure
        wealth_advisory_dir = os.path.join(upload_dir, 'wealth_advisory')
        compliance_dir = os.path.join(upload_dir, 'compliance')

        eval_type = None
        eval_dir = None

        if 'wealth' in model_name.lower():
            eval_type = "wealth_advisory"
            eval_dir = wealth_advisory_dir
        elif 'compliance' in model_name.lower():
            eval_type = "compliance"
            eval_dir = compliance_dir
        else:
            raise Exception("Neither 'wealth_advisory' nor 'compliance' folder found in uploads directory")


        # if os.path.exists(wealth_advisory_dir):
            # eval_type = "wealth_advisory"
            # eval_dir = wealth_advisory_dir
        # elif os.path.exists(compliance_dir):
            # eval_type = "compliance"
            # eval_dir = compliance_dir
        # else:
            # raise Exception("Neither 'wealth_advisory' nor 'compliance' folder found in uploads directory")

        if 'wealth' in eval_type:
            PDF_PATH = PDF_PATH_WEALTH
        else:
            PDF_PATH = PDF_PATH_COMPLIANCE

        print(f"📁 Using evaluation type: {eval_type}")
        print(f"📂 Evaluation directory: {eval_dir}")

        # Find PDF files and Excel ground truth
        pdf_files = glob.glob(os.path.join(eval_dir, "*.pdf"))
        excel_files = glob.glob(os.path.join(eval_dir, "*.xlsx")) + glob.glob(os.path.join(eval_dir, "*.xls"))

        if not pdf_files:
            raise Exception(f"No PDF files found in {eval_dir}")
        if not excel_files:
            raise Exception(f"No Excel ground truth files found in {eval_dir}")

        pdf_file = pdf_files[0]
        excel_file = excel_files[0]

        print(f"📄 Using PDF: {os.path.basename(pdf_file)}")
        print(f"📊 Using Excel: {os.path.basename(excel_file)}")

        # Stage 2: Processing uploaded files and ingesting to ChromaDB
        update_progress(model_name, 2, "Processing files and creating vector database...")

        collection_name = f"{model_name}_{eval_type}".replace(" ", "_")
        print("📄 Extracting and chunking document...")
        chunks = extract_text_chunks(PDF_PATH)

        print("🔗 Embedding chunks...")
        embed_model = EMBED_MODEL
        embeddings = embed_chunks(chunks, embed_model)

        print("📚 Creating FAISS index...")
        index = create_faiss_index(embeddings)

        # Stage 3: Loading ground truth and running queries
        update_progress(model_name, 3, "Loading ground truth and running queries...")

        # Load ground truth Excel file with appropriate sheet
        sheet_name = 'WA_GT' if eval_type == 'wealth_advisory' else 'CA_GT'
        try:
            df_ground_truth = pd.read_excel(excel_file, sheet_name=sheet_name)
        except:
            # Fallback to default sheet if named sheet doesn't exist
            df_ground_truth = pd.read_excel(excel_file)

        if 'Prompt' not in df_ground_truth.columns or 'GroundTruth' not in df_ground_truth.columns:
            raise Exception("Excel file must contain 'Prompt' and 'GroundTruth' columns")

        print(f"📋 Loaded {len(df_ground_truth)} prompts from ground truth")

        # Stage 4: Analyzing content and generating responses
        update_progress(model_name, 4, "Analyzing content and generating responses...")

        # ----------------------------Model Extraction starts----------------------------------
        predicted_responses = []
        for idx, row in tqdm(df_ground_truth.iterrows(), total=len(df_ground_truth), desc="Processing prompts"):
            prompt = row['Prompt']
            ground_truth = row['GroundTruth']

            if pd.isna(prompt) or prompt.strip() == '':
                predicted_responses.append({
                    'Prompt': prompt,
                    'Predicted': ""
                })
                continue

            try:
                # Query the RAG system
                # extracted_answer = query_documents(prompt, collection_name)
                top_chunks = search_chunks(index, prompt, embed_model, chunks)
                print("🤖 Querying Mistral via Ollama...")
                extracted_answer = query_ollama(prompt, top_chunks)
                # Calculate similarity score
                score = calculate_similarity_score(ground_truth, extracted_answer)
                grade = grade_confidence(score)
                predicted_responses.append({
                    'Prompt': prompt,
                    'Predicted': str(extracted_answer),
                    'extracted': str(extracted_answer),
                    'score': float(score),
                    'grade': grade
                })

            except Exception as e:
                print(f"Error processing prompt: {prompt} - {str(e)}")
                predicted_responses.append({
                    'Prompt': prompt,
                    'Predicted': f"Error: {str(e)}",
                    'extracted': f"Error: {str(e)}",
                    'score': 0.0,
                    'grade': '❌ Fail'

                })

        # Create predicted dataframe
        predicted_df = pd.DataFrame(predicted_responses)

        #predicted_df.to_csv('/home/poc_boa/Model_evaluation_single/uploads/predicted_df.csv')

        # Merge ground truth with predictions
        df_merged = pd.merge(df_ground_truth, predicted_df, on='Prompt', how='inner')

        #df_merged.to_csv('/home/poc_boa/Model_evaluation_single/uploads/df_merged.csv')
        # ----------------------------Model Extracted_df completed----------------------------------

        # Stage 5: Running NLP evaluation
        update_progress(model_name, 5, "Running comprehensive NLP evaluation...")

        final_report_df = run_evaluation_on_dataframe(df_merged)
        #final_report_df.to_csv('/home/poc_boa/Model_evaluation_single/uploads/final_report_df.csv')

        if final_report_df.empty:
            raise Exception("No results generated from evaluation")

        # Convert NLP results to the expected format for UI
        results = []
        for _, row in final_report_df.iterrows():
            results.append({
                'prompt': row['Prompt'],
                'actual': str(row['GroundTruth']),
                'extracted': str(row['Predicted']),
                'score': row['score'],  # Convert to percentage
                'grade': row['grade']
                # 'score': float(row['CompositeScore'] * 100),  # Convert to percentage
                # 'grade': row['PassFail']
            })

#        final_report_df.to_csv('/home/poc_boa/Model_evaluation_single/uploads/final_report_df1.csv')
        final_report_df.drop(['Prompt_Type', 'PassFail', 'extracted'], axis=1, inplace=True)


        # Calculate summary statistics
        total_tests = len(results)
        pass_count = len([r for r in results if r['grade'] == '✅ Pass'])
        # intermittent_count = 0  # Not used in new evaluation
        intermittent_count = len([r for r in results if r['grade'] == '⚠ Intermittent'])
        fail_count = len([r for r in results if r['grade'] == '❌ Fail'])

        scores = [r['score'] for r in results]
        avg_score = sum(scores) / len(scores) if scores else 0
        overall_score = avg_score
        success_rate = (pass_count / total_tests * 100) if total_tests > 0 else 0

        print(f"📈 Evaluation Summary:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {pass_count} ({pass_count / total_tests * 100:.1f}%)")
        print(f"   Failed: {fail_count} ({fail_count / total_tests * 100:.1f}%)")
        print(f"   Average Score: {avg_score:.1f}%")

        # Stage 6: Finalizing results
        update_progress(model_name, 6, "Finalizing evaluation results...")

        # Prepare comprehensive results
        final_results = {
            "model_name": model_name,
            "evaluation_type": eval_type,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "files_processed": len(pdf_files) + len(excel_files),
            "overall_score": float(overall_score),
            "total_tests": total_tests,
            "pass_count": pass_count,
            "intermittent_count": intermittent_count,
            "fail_count": fail_count,
            "average_score": float(avg_score),
            "success_rate": float(success_rate),
            "ground_truth_comparison": results,
            "file_info": {
                "pdf_file": os.path.basename(pdf_file),
                "ground_truth_file": os.path.basename(excel_file),
                "evaluation_directory": eval_type
            },
            "summary_statistics": {
                "highest_score": float(max(scores)) if scores else 0,
                "lowest_score": float(min(scores)) if scores else 0,
                "median_score": float(sorted(scores)[len(scores) // 2]) if scores else 0,
                "std_deviation": float(pd.Series(scores).std()) if scores else 0
            },
            "detailed_nlp_results": final_report_df.to_dict('records')  # Store detailed NLP metrics
        }

        # Mark as completed
        update_progress(model_name, 7, "Evaluation completed successfully!")

        print("🎉 Custom RAG evaluation with NLP metrics completed successfully!")
        return final_results

    except Exception as e:
        error_msg = f"Evaluation failed: {str(e)}"
        print(f"❌ {error_msg}")
        print("🔍 Full traceback:")
        traceback.print_exc()

        update_progress(model_name, -1, f"Error: {error_msg}")

        return {
            "error": error_msg,
            "files_processed": 0,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "traceback": traceback.format_exc()
        }


def calculate_similarity_score(actual, extracted):
    """Calculate similarity score between actual and extracted values"""
    if pd.isna(actual) or pd.isna(extracted):
        return 0.0

    actual_str = str(actual).strip().lower()
    extracted_str = str(extracted).strip().lower()

    # Simple similarity calculation (you can enhance this)
    if actual_str == extracted_str:
        return 100.0
    elif actual_str in extracted_str or extracted_str in actual_str:
        return 80.0
    else:
        # Use sentence transformer for semantic similarity
        actual_emb = EMBED_MODEL.encode(actual_str)
        extracted_emb = EMBED_MODEL.encode(extracted_str)
        from sklearn.metrics.pairwise import cosine_similarity
        score = cosine_similarity([actual_emb], [extracted_emb])[0][0]
        return round(score * 100, 2)

def grade_confidence(score):
    """Grade the confidence score"""
    if score >= 80:
        return '✅ Pass'
    elif score >= 70:
        return '⚠ Intermittent'
    else:
        return '❌ Fail'


def extract_text_chunks(pdf_path):
    doc = pymupdf.open(pdf_path)
    chunks = []

    for page in doc:
        text = page.get_text().strip()
        if not text:
            continue

        for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk = text[i:i + CHUNK_SIZE]
            chunks.append(chunk)

    return chunks


# --- Step 2: Embed Chunks ---
def embed_chunks(chunks, model):
    embeddings = model.encode(chunks, convert_to_tensor=False, show_progress_bar=True)
    return np.array(embeddings)

# --- Step 3: Create FAISS Index ---
def create_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


# --- Step 4: Search and Retrieve Top-k Chunks ---
def search_chunks(index, query, embed_model, chunks, top_k=5):
    query_embedding = embed_model.encode([query])[0]
    D, I = index.search(np.array([query_embedding]), top_k)
    return [chunks[i] for i in I[0]]


# --- Step 5: Query Ollama with Context ---
def query_ollama(question, context_chunks, model=MODEL_NAME, host=OLLAMA_URL):
    context = "\n\n".join(context_chunks)
    full_prompt = f"""You are a financial assistant. Use the context below to answer the user's question.

Context:
{context}

Question:
{question}

Answer:"""

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": full_prompt}
        ],
        "stream": False
    }

    response = requests.post(host, json=payload)
    response.raise_for_status()
    return response.json()["message"]["content"]
