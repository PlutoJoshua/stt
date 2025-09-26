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
        stt_methods = STTService.get_available_methods()
    except Exception:
        stt_methods = ['whisper_local'] # Fallback
    
    try:
        summarize_methods = TextSummarizer.get_available_methods()
    except Exception:
        summarize_methods = ['local_model'] # Fallback

    return render_template('index.html', 
                           stt_methods=stt_methods, 
                           summarize_methods=summarize_methods)

def run_background_processing(job_id, audio_paths, options):
    """The function that runs in a background thread."""
    def status_callback(message):
        # Put status message into the job's queue
        jobs[job_id]['queue'].put(message)

    try:
        result = process_file(
            audio_files=audio_paths,
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
        minutes, seconds = divmod(duration, 60)
        
        jobs[job_id]['end_time'] = datetime.fromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S")
        if minutes >= 1:
            jobs[job_id]['duration'] = f"{int(minutes)}분 {int(seconds)}초"
        else:
            jobs[job_id]['duration'] = f"{int(seconds)}초"

        # Signal completion to the SSE stream
        if jobs[job_id]['status'] == 'complete':
            status_callback('{"stage": "complete"}')
        status_callback("__STREAM_END__")

@app.route('/process', methods=['POST'])
def process():
    """Handles file upload and starts the background processing."""
    if 'audio_files' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    files = request.files.getlist('audio_files')
    if not files or files[0].filename == '':
        return jsonify({"error": "No selected file"}), 400

    audio_paths = []
    for file in files:
        filename = f"{uuid.uuid4()}_{file.filename}"
        audio_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(audio_path)
        audio_paths.append(audio_path)

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
    thread = Thread(target=run_background_processing, args=(job_id, audio_paths, options))
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
    """Provides the final result of the processing or a downloadable file."""
    if job_id not in jobs:
        return jsonify({"error": "Invalid or expired job ID"}), 404

    job_info = jobs[job_id]
    if job_info['status'] != 'complete':
        return jsonify({"error": "Job not complete"}), 404

    result_data = job_info.get('result', {})
    download_type = request.args.get('type')
    audio_filename_base = os.path.basename(result_data.get('transcript_file', '')).split('_')[0]

    # --- Handle Download Requests ---
    if download_type == 'summary':
        if not result_data.get('summary'):
            return "요약이 생성되지 않았습니다.", 404
        md_content = f"# 📝 '{audio_filename_base}' 음성 기록 요약\n\n## 📜 요약 내용\n{result_data['summary']}"
        download_filename = f"{audio_filename_base}_summary_{job_id[:8]}.md"
        response = make_response(md_content)
        response.headers['Content-Disposition'] = f"attachment; filename*=UTF-8''{download_filename}"
        response.headers['Content-Type'] = 'text/markdown; charset=utf-8'
        return response

    if download_type == 'transcript':
        if not result_data.get('transcript'):
            return "변환된 텍스트가 없습니다.", 404
        transcript_content = result_data['transcript']
        download_filename = f"{audio_filename_base}_transcript_{job_id[:8]}.txt"
        response = make_response(transcript_content)
        response.headers['Content-Disposition'] = f"attachment; filename*=UTF-8''{download_filename}"
        response.headers['Content-Type'] = 'text/plain; charset=utf-8'
        return response

    # --- Handle JSON API Request for displaying on the page ---
    response_json = {}

    # File Info
    file_info_text = f"- **파일:** `{audio_filename_base}`"
    if result_data.get('audio_info'):
        audio_info = result_data['audio_info']
        file_info_text += f"\n- **길이:** `{audio_info.get('duration_formatted')}`"
        file_info_text += f"\n- **크기:** `{audio_info.get('file_size_mb', 0):.1f}MB`"
    response_json['file_info'] = md.render(file_info_text)

    # Timing Info
    response_json['timing_info'] = md.render(f"### ⏱️ 처리 시간 정보\n- **시작 시간:** `{job_info.get('start_time_str', 'N/A')}`\n- **종료 시간:** `{job_info.get('end_time', 'N/A')}`\n- **총 소요 시간:** `{job_info.get('duration', 'N/A')}`\n")

    # Summary
    if result_data.get('summary'):
        response_json['summary_html'] = md.render(f"## 📜 요약 내용\n{result_data['summary']}")
        response_json['summary_download_url'] = f"/result/{job_id}?type=summary"

    # Transcript
    if result_data.get('transcript'):
        response_json['transcript_html'] = md.render(f"## ✍️ 전체 텍스트\n```\n{result_data['transcript']}\n```")
        response_json['transcript_download_url'] = f"/result/{job_id}?type=transcript"

    return jsonify(response_json)






if __name__ == '__main__':
    app.run(debug=True, port=5001, threaded=True)