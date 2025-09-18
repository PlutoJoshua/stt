import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import uuid
import json
import time
from datetime import datetime
from flask import Flask, request, render_template, jsonify, Response, make_response
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
        # Record end time and duration
        end_time = time.time()
        start_time = jobs[job_id].get('start_time', end_time)
        duration = end_time - start_time
        jobs[job_id]['end_time'] = datetime.fromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S")
        jobs[job_id]['duration'] = f"{duration:.2f} 초"

        # Signal completion to the SSE stream
        if jobs[job_id]['status'] == 'complete':
            status_callback('{"stage": "complete"}')
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
    start_time = time.time()
    jobs[job_id] = {
        'status': 'processing',
        'queue': Queue(),
        'result': None,
        'start_time': start_time,
        'start_time_str': datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S")
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
            try:
                # Check if the message is a JSON string
                json_data = json.loads(message)
                data = message
            except json.JSONDecodeError:
                # If not, wrap it in the standard structure
                data = json.dumps({"message": message})

            yield f"data: {data}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/result/<job_id>')
def result(job_id):
    """Provides the final result of the processing or a downloadable markdown file."""
    if job_id not in jobs:
        return jsonify({"error": "Invalid or expired job ID"}), 404

    job_info = jobs[job_id]
    if job_info['status'] != 'complete':
        return jsonify({"error": "Job not complete"}), 404

    result_data = job_info['result']
    download_format = request.args.get('format')

    # --- Helper function to generate markdown content ---
    def generate_markdown():
        audio_filename = os.path.basename(result_data.get('transcript_file', '')).split('_')[0]
        
        # 1. File Info
        md_content = "# 📝 음성 기록 요약\n\n"
        md_content += "## 📁 파일 정보\n"
        md_content += f"- **파일:** `{audio_filename}`\n"
        if result_data.get('audio_info'):
            audio_info = result_data['audio_info']
            md_content += f"- **길이:** `{audio_info.get('duration_formatted')}`\n"
            md_content += f"- **크기:** `{audio_info.get('file_size_mb', 0):.1f}MB`\n\n"

        # 2. Timing Info
        md_content += "## ⏱️ 처리 시간 정보\n"
        md_content += f"- **시작 시간:** `{job_info.get('start_time_str', 'N/A')}`\n"
        md_content += f"- **종료 시간:** `{job_info.get('end_time', 'N/A')}`\n"
        md_content += f"- **총 소요 시간:** `{job_info.get('duration', 'N/A')}`\n\n"

        # 3. Summary
        if result_data.get('summary'):
            md_content += "## 📜 요약 내용\n"
            md_content += f"{result_data['summary']}\n\n"

        # 4. Transcript
        if result_data.get('transcript'):
            md_content += "## ✍️ 전체 텍스트\n"
            md_content += f"```\n{result_data['transcript']}\n```\n"
        
        return md_content, audio_filename

    # --- Handle Download Request ---
    if download_format == 'md':
        markdown_content, audio_filename = generate_markdown()
        download_filename = f"{audio_filename}_summary_{job_id[:8]}.md"
        
        response = make_response(markdown_content)
        response.headers['Content-Disposition'] = f"attachment; filename*=UTF-8''{download_filename}"
        response.headers['Content-Type'] = 'text/markdown; charset=utf-8'
        return response

    # --- Handle JSON API Request ---
    response_json = {}
    if result_data.get('summary'):
        summary_content = result_data['summary']
        md_summary_text = f"## 📜 요약 내용\n{summary_content}"
        response_json['summary_html'] = md.render(md_summary_text)
        
        audio_filename = os.path.basename(result_data['transcript_file']).split('_')[0]
        file_info_text = f"- **파일:** `{audio_filename}`"
        if result_data.get('audio_info'):
            audio_info = result_data['audio_info']
            file_info_text += f"\n- **길이:** `{audio_info.get('duration_formatted')}`"
            file_info_text += f"\n- **크기:** `{audio_info.get('file_size_mb', 0):.1f}MB`"
        response_json['file_info'] = md.render(file_info_text)

    elif result_data.get('transcript'):
        response_json['transcript'] = result_data['transcript']

    response_json['timing_info'] = md.render(f"### ⏱️ 처리 시간 정보\n- **시작 시간:** `{job_info.get('start_time_str', 'N/A')}`\n- **종료 시간:** `{job_info.get('end_time', 'N/A')}`\n- **총 소요 시간:** `{job_info.get('duration', 'N/A')}`\n")
    response_json['download_url'] = f"/result/{job_id}?format=md"

    # In a production app, you'd use a more robust job cleanup strategy (e.g., TTL, background worker)
    # For this simple app, we keep the job result in memory for a while.
    # del jobs[job_id] 

    return jsonify(response_json)



if __name__ == '__main__':
    app.run(debug=True, port=5001, threaded=True)