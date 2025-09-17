import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import uuid
import json
from flask import Flask, request, render_template, jsonify, Response
from threading import Thread
from queue import Queue
from markdown_it import MarkdownIt

from processor import process_file
from stt_service import STTService
from summarizer import TextSummarizer

app = Flask(__name__)

# Ensure the output directory exists
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# In-memory storage for job status and results
# In a production app, you'd use a database or Redis
jobs = {}

md = MarkdownIt()

@app.route('/')
def index():
    """Renders the main page with dynamic method options."""
    try:
        stt_methods = STTService().get_available_methods()
    except Exception:
        stt_methods = ['whisper_local'] # Fallback
    
    try:
        summarize_methods = TextSummarizer().get_available_methods()
    except Exception:
        summarize_methods = ['local_model'] # Fallback

    return render_template('index.html', 
                           stt_methods=stt_methods, 
                           summarize_methods=summarize_methods)

def run_background_processing(job_id, audio_path, options):
    """The function that runs in a background thread."""
    def status_callback(message):
        # Put status message into the job's queue
        jobs[job_id]['queue'].put(message)

    try:
        result = process_file(
            audio_file=audio_path,
            output_dir=OUTPUT_FOLDER,
            stt_method=options.get('stt_method'),
            summarize_method=options.get('summarize_method'),
            summary_type=options.get('summary_type'),
            context_file=None, # Context file upload not implemented yet
            no_summary=options.get('no_summary', False),
            bullet_points=options.get('bullet_points', False),
            status_callback=status_callback
        )
        jobs[job_id]['result'] = result
        jobs[job_id]['status'] = 'complete'
    except Exception as e:
        jobs[job_id]['status'] = 'error'
        jobs[job_id]['result'] = {"error": str(e)}
        status_callback(f"❌ 오류 발생: {e}")
    finally:
        # Signal completion to the SSE stream
        status_callback("__STREAM_END__")

@app.route('/process', methods=['POST'])
def process():
    """Handles file upload and starts the background processing."""
    if 'audio_file' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    file = request.files['audio_file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded file
    filename = f"{uuid.uuid4()}_{file.filename}"
    audio_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(audio_path)

    # Collect options from form
    options = {
        "stt_method": request.form.get('stt_method'),
        "summarize_method": request.form.get('summarize_method'),
        "summary_type": request.form.get('summary_type'),
        "no_summary": request.form.get('no_summary') == 'true',
        "bullet_points": request.form.get('bullet_points') == 'true',
    }

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        'status': 'processing',
        'queue': Queue(),
        'result': None
    }

    # Start the background thread
    thread = Thread(target=run_background_processing, args=(job_id, audio_path, options))
    thread.start()

    return jsonify({"status": "processing", "job_id": job_id})

@app.route('/status/<job_id>')
def status(job_id):
    """Server-Sent Events stream for status updates."""
    if job_id not in jobs:
        return jsonify({"error": "Invalid job ID"}), 404

    def generate():
        q = jobs[job_id]['queue']
        while True:
            message = q.get() # This will block until a message is available
            if message == "__STREAM_END__":
                break
            # SSE format: data: {json_string}\n\n
            data = json.dumps({"message": message})
            yield f"data: {data}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/result/<job_id>')
def result(job_id):
    """Provides the final result of the processing."""
    if job_id not in jobs or jobs[job_id]['status'] != 'complete':
        return jsonify({"error": "Job not complete or invalid ID"}), 404

    result_data = jobs[job_id]['result']
    response = {}

    if result_data.get('summary'):
        # Re-create the markdown summary to be sure
        summary_content = result_data['summary']
        audio_filename = os.path.basename(result_data['transcript_file']).split('_')[0]
        
        # This part is a bit of a hack, ideally the processor returns all this info
        # For now, we just extract what we can.
        md_summary_text = f"## 📜 요약 내용\n{summary_content}"
        response['summary_html'] = md.render(md_summary_text)
        response['file_info'] = f"- **파일:** `{audio_filename}`"

    elif result_data.get('transcript'):
        response['transcript'] = result_data['transcript']

    # Clean up the job from memory
    del jobs[job_id]

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=5001, threaded=True)