from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for, session
from werkzeug.utils import secure_filename
import os
import json
from PyPDF2 import PdfReader
import tiktoken
from together import Together
import google.generativeai as genai
from groq import Groq
import openai
from dotenv import load_dotenv
import shutil
from datetime import datetime
import uuid

load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}
app.config['SECRET_KEY'] = os.urandom(24)
app.config['SESSION_FILE_DIR'] = 'session_files'

PROMPTS_FILE = 'saved_prompts.json'

# Ensure session file directory exists
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

# Initialize AI clients
together_client = Together(api_key=os.getenv('TOGETHER_API_KEY'))
genai.configure(api_key=os.getenv('GOOGLE_AI_API_KEY'))
groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def load_saved_prompts():
    if os.path.exists(PROMPTS_FILE):
        with open(PROMPTS_FILE, 'r') as f:
            return json.load(f)
    return []

def save_prompt(title, prompt):
    prompts = load_saved_prompts()
    prompts.append({'title': title, 'prompt': prompt})
    with open(PROMPTS_FILE, 'w') as f:
        json.dump(prompts, f)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def count_tokens(text):
    tokens = tokenizer.encode(text)
    return len(tokens)

def generate_response(provider, system_message, user_input, temperature, max_tokens):
    if provider == 'together':
        response = together_client.chat.completions.create(
            model="meta-llama/Meta-Llama-3-70B-Instruct-Turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_input}
            ],
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            top_p=0.7,
            top_k=50,
            repetition_penalty=1,
            stop=["<|eot_id|>"],
            stream=False
        )
        return response.choices[0].message.content
    elif provider == 'google':
        generation_config = {
            "temperature": float(temperature),
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": int(max_tokens),
            "response_mime_type": "text/plain",
        }
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
            system_instruction=system_message,
        )
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(user_input)    
        return response.text


    elif provider =='googleproexp':
        generation_config = {
            "temperature": float(temperature),
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": int(max_tokens),
            "response_mime_type": "text/plain",
        }
        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro",
            generation_config=generation_config,
            system_instruction=system_message,
        )
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(user_input)    
        return response.text

    elif provider =='googlepro':
        generation_config = {
            "temperature": float(temperature),
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": int(max_tokens),
            "response_mime_type": "text/plain",
        }
        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro",
            generation_config=generation_config,
            system_instruction=system_message,
        )
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(user_input)    
        return response.text
    
    elif provider == 'groq':
        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_input}
            ],
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            top_p=1,
            stream=False,
            stop=None,
        )
        return response.choices[0].message.content
    
    elif provider == 'openai':
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_input}
            ],
            temperature=float(temperature),
            max_tokens=int(max_tokens)
        )
        return response.choices[0].message.content
    
    elif provider == 'openaimini':
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_input}
            ],
            temperature=float(temperature),
            max_tokens=int(max_tokens)
        )
        return response.choices[0].message.content

def save_session_data(data):
    session_id = str(uuid.uuid4())
    file_path = os.path.join(app.config['SESSION_FILE_DIR'], f"{session_id}.json")
    with open(file_path, 'w') as f:
        json.dump(data, f)
    return session_id

def load_session_data(session_id):
    file_path = os.path.join(app.config['SESSION_FILE_DIR'], f"{session_id}.json")
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return None

@app.route('/')
def index():
    return redirect(url_for('process_pdf'))

@app.route('/save_prompt', methods=['POST'])
def save_prompt_route():
    title = request.form['prompt_title']
    prompt = request.form['user_prompt']
    save_prompt(title, prompt)
    return redirect(url_for('process_pdf'))

@app.route('/process', methods=['GET', 'POST'])
def process_pdf():
    if request.method == 'POST':
        # File handling
        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                extracted_text = extract_text_from_pdf(file_path)
                token_count = count_tokens(extracted_text)
            else:
                filename = ''
                token_count = 0
        else:
            filename = request.form.get('filename', '')
            token_count = int(request.form.get('token_count', 0))
        
        # Process form data
        title = request.form['title']
        provider = request.form['provider']
        system_message = "Atue como um excelente assistente jurídico de um juiz de direito ou um juiz federal"
        user_prompt = request.form['user_prompt']
        temperature = request.form['temperature']
        max_tokens = request.form['max_tokens']
        additional_context = request.form['additional_context']
        
        # Prepare full prompt
        if filename:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.exists(file_path):
                extracted_text = extract_text_from_pdf(file_path)
                full_prompt = f"{user_prompt}\n\nContext from PDF:\n{extracted_text}\n\nAdditional Context:\n{additional_context}"
            else:
                full_prompt = f"{user_prompt}\n\nAdditional Context:\n{additional_context}"
        else:
            full_prompt = f"{user_prompt}\n\nAdditional Context:\n{additional_context}"
        
        # Generate response
        response = generate_response(provider, system_message, full_prompt, temperature, max_tokens)
        
        # Save results
        session_id = session.get('session_id')
        data = load_session_data(session_id) if session_id else {}
        
        pdf_results = data.get('pdf_results', []) if isinstance(data, dict) else data if isinstance(data, list) else []
        
        new_result = {
            'title': title,
            'response': response,
            'token_count': token_count,
            'filename': filename
        }
        
        pdf_results.append(new_result)
        
        new_data = {
            'pdf_results': pdf_results,
            'form_data': request.form
        }
        new_session_id = save_session_data(new_data)
        session['session_id'] = new_session_id
        
        # Render results page
        return render_template('results.html', results=pdf_results, latest_result=new_result)
    else:
        session_id = session.get('session_id')
        data = load_session_data(session_id) if session_id else {}
        form_data = data.get('form_data', {}) if isinstance(data, dict) else {}
        saved_prompts = load_saved_prompts()
        return render_template('process.html', filename='', token_count=0, form_data=form_data, saved_prompts=saved_prompts)
    
@app.route('/reset', methods=['POST'])
def reset():
    session_id = session.get('session_id')
    if session_id:
        file_path = os.path.join(app.config['SESSION_FILE_DIR'], f"{session_id}.json")
        if os.path.exists(file_path):
            os.remove(file_path)
    session.clear()
    
    # Delete all files in the upload folder
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
    
    return redirect(url_for('index'))

@app.route('/final_process', methods=['GET', 'POST'])
def final_process():
    if request.method == 'POST':
        provider = request.form['provider']
        system_message = "Atue como um excelente assistente jurídico de um juiz de direito ou um juiz federal"
        final_prompt = request.form['final_prompt']
        temperature = request.form['temperature']
        max_tokens = request.form['max_tokens']
        additional_context = request.form['additional_context']
        
        session_id = session.get('session_id')
        data = load_session_data(session_id) if session_id else None
        
        # Handle cases where data might be a list or a dictionary
        if isinstance(data, list):
            pdf_results = data
        elif isinstance(data, dict):
            pdf_results = data.get('pdf_results', [])
        else:
            pdf_results = []
        
        if not pdf_results:
            return "No PDF results found. Please process PDFs before final processing.", 400
        
        previous_responses = "\n\n".join([f"Title: {result['title']}\nResponse: {result['response']}" for result in pdf_results])
        full_prompt = f"{final_prompt}\n\nPrevious Responses:\n{previous_responses}\n\nAdditional Context:\n{additional_context}"
        
        response = generate_response(provider, system_message, full_prompt, temperature, max_tokens)
        
        final_data = {
            'pdf_results': pdf_results,
            'final_result': response,
            'form_data': request.form
        }
        new_session_id = save_session_data(final_data)
        session['session_id'] = new_session_id
        
        return render_template('final_result.html', response=response)

    # For GET requests, we'll display the form and show available PDF results
    session_id = session.get('session_id')
    data = load_session_data(session_id) if session_id else None
    
    # Handle cases where data might be a list or a dictionary
    if isinstance(data, list):
        pdf_results = data
        form_data = {}
    elif isinstance(data, dict):
        pdf_results = data.get('pdf_results', [])
        form_data = data.get('form_data', {})
    else:
        pdf_results = []
        form_data = {}
    
    return render_template('final_process.html', pdf_results=pdf_results, form_data=form_data)
    
@app.route('/save_results', methods=['POST'])
def save_results():
    session_id = session.get('session_id')
    data = load_session_data(session_id)
    if not data:
        return "No data to save", 400

    pdf_results = data.get('pdf_results', [])
    final_result = data.get('final_result', '')
    
    timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    filename = f"Resultado-{timestamp}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        for result in pdf_results:
            f.write(f"Title: {result['title']}\n")
            f.write(f"Response: {result['response']}\n\n")
        
        f.write("Final Result:\n")
        f.write(final_result)
    
    return send_from_directory('.', filename, as_attachment=True)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=8080)