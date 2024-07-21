from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
from PyPDF2 import PdfReader
import tiktoken
from together import Together
import google.generativeai as genai
from groq import Groq
import openai
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

# Initialize AI clients
together_client = Together(api_key=os.getenv('TOGETHER_API_KEY'))
genai.configure(api_key=os.getenv('GOOGLE_AI_API_KEY'))
groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

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
        messages=[],
        max_tokens=int(max_tokens),
        temperature=float(user_input),
        top_p=0.7,
        top_k=50,
        repetition_penalty=1,
        stop=["<|eot_id|>"],
        stream=False
        )
        return response.choices[0].message.content
    
    
    elif provider == 'google':
        generation_config = {"temperature": float(temperature), "top_p": 0.95,"top_k": 64, "max_output_tokens": int(max_tokens), "response_mime_type": "text/plain",}
        
        model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
        system_instruction=system_message,
        )

        chat_session = model.start_chat(
        history=[
        ]
        )

        response = chat_session.send_message(user_input)    
        
        return response.text

    elif provider == 'groq':
        response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
        {
            "role": "system",
            "content": system_message
        },
        {
            "role": "user",
            "content": user_input
        }
        ],
        temperature=float(temperature),
        max_tokens=int(max_tokens),
        top_p=1,
        stream=False,
        stop=None,
        )

        return response.choices[0].message

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

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_files = request.files.getlist('file')
        results = []
        for file in uploaded_files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                extracted_text = extract_text_from_pdf(file_path)
                token_count = count_tokens(extracted_text)
                results.append({
                    'filename': filename,
                    'text': extracted_text,
                    'token_count': token_count
                })
        return jsonify({'results': results})
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    data = request.json
    provider = data['provider']
    system_message = data['system_message']
    user_prompt = data['user_prompt']
    temperature = data['temperature']
    max_tokens = data['max_tokens']
    context1 = data['context1']
    context2 = data['context2']
    
    full_prompt = f"{user_prompt}\n\nContext from PDF:\n{context1}\n\nAdditional Context:\n{context2}"
    
    response = generate_response(provider, system_message, full_prompt, temperature, max_tokens)
    return jsonify({'response': response})

@app.route('/final_process', methods=['POST'])
def final_process():
    data = request.json
    provider = data['provider']
    system_message = data['system_message']
    final_prompt = data['final_prompt']
    temperature = data['temperature']
    max_tokens = data['max_tokens']
    previous_responses = data['previous_responses']
    additional_context = data['additional_context']
    
    full_prompt = f"{final_prompt}\n\nPrevious Responses:\n{previous_responses}\n\nAdditional Context:\n{additional_context}"
    
    response = generate_response(provider, system_message, full_prompt, temperature, max_tokens)
    return jsonify({'response': response})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=8080)