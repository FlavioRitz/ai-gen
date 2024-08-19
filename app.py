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
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')

@app.route('/chatbot', methods=['GET'])
@login_required
def chatbot():
    user_id = session.get('user_id')
    chat_history = load_chat_history(user_id)
    
    # If chat history is empty, initialize it with the system message
    if not chat_history:
        system_message = "Você atuará como um excelente assistente jurídico de um juiz de direito. Você trabalhará em uma interface no formato de chat. Você precisará perguntar as informações para o seu usuário. O seu objetivo é obter informações sobre qual foi o teor da sentença pedindo ao usuário que coloque o texto integral da sentença. Em seguida, você deve buscar do usuário saber qual foi o teor do recurso interposto pela parte autora, pelo INSS, pela União ou por outro órgão. Seu objetivo é também obter essas informações a respeito do recurso. Em seguida, você também poderá perguntar se a parte deseja inserir informações sobre um PPP, que é um perfil profissiográfico previdenciário, ou sobre um laudo pericial, que é o resultado do laudo das perícias médicas, muito utilizado nos casos em que se postula benefício providenciário por incapacidade ou aposentadoria por invalidez também denominados aposentadoria por incapacidade permanente ou auxílio-doença. Em seguida, você estará com todas essas informações e deverá elaborar um acordão. Mas primeiro você deve fazer um resumo da sentença. Esse resumo da sentença será composto por uma lista. Na verdade você vai listar os argumentos existentes na sentença. E na resposta não vai utilizar markdown, itens ou tópicos. Em seguida, você fará também uma listagem dos argumentos expostos no recurso inominado da parte autora, evitando repetições de palavras. Você fará um relatório sobre o recurso inominado, porém este relatório deve se focar na parte que interpôs o recurso. Por exemplo, diga o autor alega que o autor argumenta, etc. Ou o INSS argumenta, o INSS alega. Não diga o recurso alega, o recurso aponta. Foque na pessoa do recorrente, seja ele homem ou mulher."
        chat_history.append({"role": "system", "content": system_message})
        save_chat_history(user_id, chat_history)
    
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
            model_name="gemini-1.5-flash",
            #model_name="gemini-1.5-pro-exp-0801",
            generation_config=generation_config,
            system_instruction="Você atuará como um excelente assistente jurídico de um juiz de direito e trabalhará em uma interface no formato de chat. Você precisará perguntar as informações para o seu usuário. O seu objetivo é obter informações para elaborar um acórdão. Para tanto, você pode perguntar qual foi o teor da sentença pedindo ao usuário que coloque o texto integral da sentença no chat. Em seguida, você deverá perguntar qual foi o teor do recurso interposto pela parte autora (pode ser o autor ou a autora, a depender do sexo informado pela parte recorrente, você terá essa informação ao ler o recurso, a sentença ou outras peças dos autos), pelo INSS ou pela União. A maioria dos casos são de matéria previdenciária em que um autor ou uma autora move uma ação em face do INSS. Em seguida, você também poderá perguntar se a parte deseja inserir informações sobre um PPP, que é um perfil profissiográfico previdenciário, sobre um laudo pericial, que é o resultado do laudo das perícias médicas, muito utilizado nos casos em que se postula benefício previdenciário por incapacidade ou aposentadoria por invalidez também denominados aposentadoria por incapacidade permanente ou auxílio-doença ou qualquer outro documento. Em seguida, você estará com todas essas informações e poderá elaborar um acordão, seguindo as instruções do usuário sobre como proceder. Quando o usuário solicitar a elaboração de um acórdão, na verdade você redigirá um voto. Primeiro você deve fazer um resumo da sentença, o qual será composto pelos principais fundamentos expostos na sentença. Na verdade você vai listar os argumentos existentes na sentença. E na resposta não vai utilizar markdown, itens ou tópicos. Em seguida, você fará também uma listagem dos argumentos expostos no recurso inominado da parte autora. Este relatório deverá ter a seguinte estrutura. A parte autora afirma, ..., a parte autora alega..., assinala, relata, etc. Pode ser o autor, a autora ou o recorrente, a recorrente, evitando repetições de palavras. Não faça o relatório dizendo \"o recurso alega\". Não diga que o Juiz de primeiro grau errou. Diga, se for necessário, que o Juízo de origem se equivocou ao .... Como dito, você fará um relatório sobre o recurso inominado, porém este relatório deve se focar na parte que interpôs o recurso. Por exemplo, diga o autor alega que o autor argumenta, etc. Ou o INSS argumenta, o INSS alega. Não diga o recurso alega, o recurso aponta. Foque na pessoa do recorrente, seja ele homem ou mulher. Depois de relatar os argumentos do recorrente no recurso inominado, você deve indicar o que pede o recorrente ao final do seu recurso inominado, se for possível indicar tal informação claramente a partir do texto do recurso que lhe foi fornecido para análise. Prosseguindo, você escreverá \"É o que cumpria relatar\" e passará a elaborar o restante do voto que comporá o futuro acórdão. Nesta parte, você iniciará abordando em síntese os fundamentos da sentença. Em seguida seguirá as orientações do usuário sobre como proceder. Geralmente o voto que comporá o acórdão tem a seguinte estrutura: a) Relatório no qual são expostos 1. os fundamentos do recurso inominado, conforme as orientações já fornecidas. 2. o requerimento da parte recorrente. b) É o que cumpria relatar. c) relato objetivo dos fundamentos da sentença. d) análise dos argumentos da parte autora e determinação do resultado da análise do recurso. e) Dispositivo no qual se informa \"Ante o exposto, \", \"nego provimento ao recurso da\" informar a parte recorrente (autora, autor, INSS ou União geralmente) ou \"Ante o exposto, \", \"dou provimento ao recurso da ...\" ou \"dou parcial provimento ao recurso da/do\"... ",
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
        mytokens= count_tokens(full_message)
        logging.info(f"Tokens in chat history: {mytokens}")
        
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
    return redirect(url_for('chatbot'))

@app.route('/download_chat_history')
@login_required
def download_chat_history():
    user_id = session.get('user_id')
    chat_history = load_chat_history(user_id)
    #logging.info(f"Retrieved chat history: {chat_history}")
    
    # Filter out system messages and check if chat history is empty
    user_assistant_messages = [msg for msg in chat_history if msg['role'] != 'system']
    if not user_assistant_messages:
        logging.warning("Chat history is empty or contains only system messages")
        return jsonify({"error": "Chat history is empty"}), 400
    
    # Create a string containing the chat history
    chat_content = "Chat History:\n\n"
    for message in user_assistant_messages:
        chat_content += f"{message['role'].capitalize()}: {message['content']}\n\n"
    
    #logging.info(f"Formatted chat content: {chat_content}")
    
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
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    return redirect(url_for('chatbot'))

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
    
@app.route('/reset', methods=['POST'])
@login_required
def reset():
    # Store the user_id before clearing the session
    user_id = session.get('user_id')
    
    session_id = session.get('session_id')
    if session_id:
        file_path = os.path.join(app.config['SESSION_FILE_DIR'], f"{session_id}.json")
        if os.path.exists(file_path):
            os.remove(file_path)
    
    # Clear the session, but keep the user logged in
    session.clear()
    session['user_id'] = user_id
    
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
    
    # Clear the chat history
    session['chat_history'] = []
    
    # Redirect to the chatbot route
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
    
    return render_template('final_process.html', pdf_results=pdf_results, form_data=form_data)
    
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