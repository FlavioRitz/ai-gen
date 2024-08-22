from flask import Flask, send_file, render_template, request, jsonify, send_from_directory, redirect, url_for, session
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
from io import BytesIO
import logging

# Import the user authentication module
from user_auth import login_required, register_user, authenticate_user

load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}
app.config['SECRET_KEY'] = os.urandom(24)
app.config['SESSION_FILE_DIR'] = 'session_files'
app.config['CHAT_HISTORY_DIR'] = 'chat_histories'

# Ensure chat history directory exists
os.makedirs(app.config['CHAT_HISTORY_DIR'], exist_ok=True)

logging.basicConfig(level=logging.INFO)

PROMPTS_FILE = 'saved_prompts.json'
SYSTEM_INSTRUCTION = '''Você é um assistente jurídico altamente qualificado, trabalhando para um juiz federal ou para um juiz de direito em uma interface de chat. Seu objetivo é auxiliar na análise de processos judiciais e na elaboração de relatórios, votos e acórdãos, principalmente em casos previdenciários. Siga estas instruções cuidadosamente:

1. Coleta de Informações:
   a) Solicite o texto integral da sentença.
   b) Peça o texto do recurso interposto (pela parte autora, INSS, União, CEF ou outras instituições).
   c) Pergunte sobre documentos adicionais relevantes, como:
      - Perfil Profissiográfico Previdenciário (PPP)
      - Laudo pericial (para casos de benefícios por incapacidade ou aposentadoria por invalidez)
   d) Esteja preparado para receber informações sobre outros tipos de documentos.

2. Análise Imediata:
   Ao receber qualquer documento (sentença, recurso, etc.), faça imediatamente um breve resumo dos pontos principais antes de solicitar mais informações. SOMENTE PASSE À REDAÇÃO DO VOTO OU ACÓRDÃO QUANDO SOLICITADO PELO USUÁRIO. Responda com o resumo dos principais pontos do documento e solicite mais instruções se necessário. 

3. Elaboração do Voto/Acórdão:
   a) Relatório:
      - Liste os principais argumentos da sentença sem usar marcadores ou formatação especial.
      - Resuma os argumentos do recurso, focando na parte recorrente:
        * Use "O autor alega..." ou "A autora alega" para pessoas físicas, a depender do caso. Poderá utilizar também "O recorrente" ou "a recorrente".
        * Use "O INSS", "A União", "A CEF" sustenta, alega, argumenta, etc para instituições a depender de quem for o recorrente.
      - Indique o pedido final do recorrente.
      - Conclua com "É o que cumpria relatar".
   b) Voto:
      - Inicie com uma síntese dos fundamentos da sentença.
      - Analise os argumentos da parte recorrente.
      - Determine o resultado da análise do recurso, seguindo as instruções do usuário.
   c) Dispositivo:
      - Use fórmulas como "Ante o exposto, nego provimento ao recurso de [parte recorrente]" ou "dou parcial provimento ao recurso de [parte recorrente]".

4. Diretrizes Gerais:
   - Mantenha o foco em questões jurídicas.
   - Adapte-se às instruções específicas do usuário.
   - Evite dizer que o "Juiz de primeiro grau errou". Use "o Juízo de origem se equivocou" se necessário.
   - Não use a expressão "o recurso alega". Foque na pessoa ou instituição recorrente. Diga o recorrente alega..., aduz, afirma, assinala, destaca, argumenta, etc
   - Seja flexível quanto à ordem de recebimento das informações e esteja preparado para realizar tarefas parciais.
   - Lembre-se que a maioria dos casos são previdenciários, envolvendo um autor/autora que propõe uma ação em face do INSS.

5. Interação:
   - Solicite esclarecimentos quando necessário.
   - Esteja preparado para elaborar relatórios parciais ou completos conforme solicitado.
   - Adapte suas respostas com base no contexto e nas necessidades específicas do caso.

Lembre-se: Sua função é auxiliar na análise e na tomada de decisões, fornecendo informações precisas e relevantes para o processo judicial em questão.'''


# Ensure session file directory exists
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

# Initialize AI clients
together_client = Together(api_key=os.getenv('TOGETHER_API_KEY'))
genai.configure(api_key=os.getenv('GOOGLE_AI_API_KEY'))
groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def get_chat_history_path(user_id):
    return os.path.join(app.config['CHAT_HISTORY_DIR'], f"{user_id}_chat_history.json")

def load_chat_history(user_id):
    file_path = get_chat_history_path(user_id)
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return []

def save_chat_history(user_id, chat_history):
    file_path = get_chat_history_path(user_id)
    with open(file_path, 'w') as f:
        json.dump(chat_history, f)
        
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
            model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_input}
            ],
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            top_p=0.7,
            top_k=50,
            repetition_penalty=1,
            stop=["<|eot_id|>","<|eom_id|>"],
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
            model_name="gemini-1.5-pro-exp-0801", # Experimental model
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
            model="gpt-4o-2024-08-06", # Added the new openai model
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

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if authenticate_user(username, password):
            session['user_id'] = username
            return redirect(url_for('chatbot'))
        else:
            return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')

@app.route('/chatbot', methods=['GET'])
@login_required
def chatbot():
    user_id = session.get('user_id')
    chat_history = load_chat_history(user_id)
    
    # If chat history is empty, initialize it with the system message
    """ if not chat_history:
        chat_history.append({"role": "system", "content": SYSTEM_INSTRUCTION})
        save_chat_history(user_id, chat_history)
 """    
    return render_template('chatbot.html', chat_history=chat_history)

@app.route('/chatbot_send', methods=['POST'])
@login_required
def chatbot_send():
    user_id = session.get('user_id')
    message = request.form.get('message', '')
    chat_history = load_chat_history(user_id)
    
    #logging.info(f"Received message: {message}")
    #logging.info(f"Current chat history: {chat_history}")
    
    # Handle PDF upload (if needed)
    pdf_text = ""
    if 'pdf' in request.files:
        pdf_file = request.files['pdf']
        if pdf_file and allowed_file(pdf_file.filename):
            filename = secure_filename(pdf_file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            pdf_file.save(file_path)
            
            # Extract text from PDF
            pdf_text = extract_text_from_pdf(file_path)
            
            # Remove the temporary PDF file
            os.remove(file_path)
    
    # Combine user message with PDF text
    full_message = f"{message}\n\nAqui está o texto do PDF que estou enviando:\n{pdf_text}" if pdf_text else message
    
    try:
        genai.configure(api_key=os.environ["GOOGLE_AI_API_KEY"])
        
        generation_config = {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }
        
        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro-exp-0801",
            generation_config=generation_config,
            system_instruction=SYSTEM_INSTRUCTION,
        )
        
        # Convert chat history to the format expected by Gemini
        gemini_history = []
        for msg in chat_history:
            if msg["role"] == "user":
                gemini_history.append({"role": "user", "parts": [msg["content"]]})
            elif msg["role"] == "assistant":
                gemini_history.append({"role": "model", "parts": [msg["content"]]})
        
        chat_session = model.start_chat(history=gemini_history)
        response = chat_session.send_message(full_message)
        
        assistant_message = response.text

        # Update chat history
        chat_history.append({"role": "user", "content": full_message})
        chat_history.append({"role": "assistant", "content": assistant_message})
    
         # Save the updated chat history
        save_chat_history(user_id, chat_history)
    
        return jsonify({
            "status": "success",
            "message": assistant_message,
            "chat_history": chat_history
        })        

    except Exception as e:
        logging.error(f"Error in chatbot API call: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"An error occurred while processing your request: {str(e)}"
        }), 500
                        
@app.route('/clear_chat', methods=['POST'])
@login_required
def clear_chat():
    user_id = session.get('user_id')
    file_path = get_chat_history_path(user_id)
    if os.path.exists(file_path):
        os.remove(file_path)
    
    # Initialize an empty chat history
    save_chat_history(user_id, [])
    
    return jsonify({
        "status": "success",
        "message": "Chat history cleared successfully"
    })

@app.route('/download_chat_history')
@login_required
def download_chat_history():
    user_id = session.get('user_id')
    chat_history = load_chat_history(user_id)
    
    # Filter out system messages and check if chat history is empty
    user_assistant_messages = [msg for msg in chat_history if msg['role'] != 'system']
    if not user_assistant_messages:
        logging.warning("Chat history is empty or contains only system messages")
        return jsonify({"error": "Chat history is empty"}), 400
    
    # Create a string containing the filtered chat history
    chat_content = "Chat History:\n\n"
    for message in user_assistant_messages:
        content = message['content']
        if message['role'] == 'user':
            # Remove PDF content if present
            pdf_start = content.find("Aqui está o texto do PDF que estou enviando:")
            if pdf_start != -1:
                content = content[:pdf_start].strip()
        
        chat_content += f"{message['role'].capitalize()}: {content}\n\n"
    
    # Generate a filename with the current timestamp
    timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    filename = f"chat_history_{timestamp}.txt"
    
    return send_file(
        BytesIO(chat_content.encode('utf-8')),
        as_attachment=True,
        download_name=filename,
        mimetype='text/plain'
    )
    
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if register_user(username, password):
            session['user_id'] = username
            return redirect(url_for('index'))
        else:
            return render_template('register.html', error='Username already exists')
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('index'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/save_prompt', methods=['POST'])
@login_required
def save_prompt_route():
    title = request.form['prompt_title']
    prompt = request.form['user_prompt']
    save_prompt(title, prompt)
    saved_prompts = load_saved_prompts()
    return jsonify({'success': True, 'saved_prompts': saved_prompts})


@app.route('/process', methods=['GET', 'POST'])
@login_required
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
    
@app.route('/reset', methods=['GET', 'POST'])
@login_required
def reset():
    user_id = session.get('user_id')
    
    session_id = session.get('session_id')
    if session_id:
        file_path = os.path.join(app.config['SESSION_FILE_DIR'], f"{session_id}.json")
        if os.path.exists(file_path):
            os.remove(file_path)
    
    session.clear()
    session['user_id'] = user_id
    
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
    
    save_chat_history(user_id, [])
    
    redirect_to = request.args.get('redirect', 'chatbot')
    if redirect_to == 'process':
        return redirect(url_for('process_pdf'))
    elif redirect_to == 'final_process':
        return redirect(url_for('final_process'))
    else:
        return redirect(url_for('chatbot'))

@app.route('/final_process', methods=['GET', 'POST'])
@login_required
def final_process():
    if request.method == 'POST':
        provider = request.form['provider']
        system_message = "Atue como um excelente assistente jurídico de um juiz de direito ou um juiz federal"
        final_prompt = request.form['final_prompt']
        temperature = request.form['temperature']
        max_tokens = request.form['max_tokens']
        additional_context = request.form['additional_context']
        
        # Handle optional PDF upload
        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                extracted_text = extract_text_from_pdf(file_path)
                additional_context += f"\n\nExtracted text from uploaded PDF:\n{extracted_text}"
        
        session_id = session.get('session_id')
        data = load_session_data(session_id) if session_id else None
        
        if isinstance(data, list):
            pdf_results = data
        elif isinstance(data, dict):
            pdf_results = data.get('pdf_results', [])
        else:
            pdf_results = []
        
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

    # For GET requests
    session_id = session.get('session_id')
    data = load_session_data(session_id) if session_id else None
    
    if isinstance(data, list):
        pdf_results = data
        form_data = {}
    elif isinstance(data, dict):
        pdf_results = data.get('pdf_results', [])
        form_data = data.get('form_data', {})
    else:
        pdf_results = []
        form_data = {}
    
    saved_prompts = load_saved_prompts()
    return render_template('final_process.html', pdf_results=pdf_results, form_data=form_data, saved_prompts=saved_prompts)
    
@app.route('/save_results', methods=['POST'])
@login_required
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
@login_required
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=8080)