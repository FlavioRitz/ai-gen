# PDF Processing and AI Analysis Web Application

This Flask-based web application allows users to upload PDF files, process them using various AI providers, and generate a final analysis based on the processed results.

## Features

- PDF upload and text extraction
- Processing of extracted text using multiple AI providers (Together AI, Google AI, Groq, OpenAI)
- Final analysis generation based on processed results
- Results saving functionality
- Session management using file-based storage to handle large datasets

## Prerequisites

- Python 3.7+
- Flask
- PyPDF2
- tiktoken
- together
- google-generativeai
- groq
- openai
- python-dotenv

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root and add your API keys:
   ```
   TOGETHER_API_KEY=your_together_api_key
   GOOGLE_AI_API_KEY=your_google_ai_api_key
   GROQ_API_KEY=your_groq_api_key
   OPENAI_API_KEY=your_openai_api_key
   ```

## Usage

1. Run the Flask application:
   ```
   python app.py
   ```

2. Open a web browser and navigate to `http://localhost:8080`.

3. Upload a PDF file and process it using the available AI providers.

4. After processing one or more PDFs, proceed to the final processing step to generate an overall analysis.

5. Save the results as a text file.

## Project Structure

- `app.py`: Main Flask application file
- `templates/`: HTML templates for the web interface
- `uploads/`: Directory for storing uploaded PDF files
- `session_files/`: Directory for storing session data

## Contributing

Please feel free to submit issues and pull requests for any improvements or bug fixes.

## License

[MIT License](https://opensource.org/licenses/MIT)