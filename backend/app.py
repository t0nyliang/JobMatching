import json
import os
import uuid

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from dotenv import load_dotenv

from matcher import build_candidate_text, build_tfidf_index, rank_jobs
from feedback import parse_feedback

load_dotenv()

app = Flask(__name__)
CORS(app)

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
FRONTEND_PATH = os.path.join(os.path.dirname(
    __file__), '..', 'frontend', 'index.html')

with open(os.path.join(DATA_DIR, 'jobs.json'), encoding='utf-8') as f:
    jobs = json.load(f)

with open(os.path.join(DATA_DIR, 'candidates.json'), encoding='utf-8') as f:
    candidates = json.load(f)

# Build TF-IDF index once at startup; all requests share it read-only
build_tfidf_index(jobs)

sessions = {}  # keyed by 8-char session_id


@app.route('/')
def serve_frontend():
    return send_file(FRONTEND_PATH)


@app.route('/api/candidates')
def get_candidates():
    return jsonify(candidates)


@app.route('/api/session/start', methods=['POST'])
def session_start():
    body = request.get_json()
    candidate = body.get('candidate', {})

    session_id = uuid.uuid4().hex[:8]
    candidate_text = build_candidate_text(candidate)

    session = {
        'candidate': candidate,
        'candidate_text': candidate_text,
        'preferences': {'boost': [], 'penalize': [], 'require': []},
        'rounds': [],
    }

    recommendations = rank_jobs(candidate_text, session['preferences'])

    session['rounds'].append({
        'round': 1,
        'jobs': recommendations,
        'feedback': None,
        'delta': None,
    })

    sessions[session_id] = session

    return jsonify({
        'session_id': session_id,
        'round': 1,
        'recommendations': recommendations,
    })


@app.route('/api/session/feedback', methods=['POST'])
def session_feedback():
    body = request.get_json()
    session_id = body.get('session_id')
    feedback_text = body.get('feedback', '')
    candidate_update = body.get('candidate')

    if session_id not in sessions:
        return jsonify({'error': 'Session not found'}), 404

    session = sessions[session_id]

    if candidate_update:
        session['candidate'] = candidate_update
        session['candidate_text'] = build_candidate_text(candidate_update)

    # Claude merges old + new prefs and returns the combined result
    try:
        delta = parse_feedback(feedback_text, session['preferences'])
    except Exception as e:
        return jsonify({'error': f'Feedback parsing failed: {str(e)}'}), 500

    session['preferences'] = {
        'boost': delta.get('boost', []),
        'penalize': delta.get('penalize', []),
        'require': delta.get('require', []),
    }

    # Attach feedback + delta to the round that was just shown
    if session['rounds']:
        session['rounds'][-1]['feedback'] = feedback_text
        session['rounds'][-1]['delta'] = delta

    recommendations = rank_jobs(
        session['candidate_text'], session['preferences'])

    new_round = {
        'round': len(session['rounds']) + 1,
        'jobs': recommendations,
        'feedback': None,
        'delta': None,
    }
    session['rounds'].append(new_round)

    return jsonify({
        'round': new_round['round'],
        'preference_delta': delta,
        'recommendations': recommendations,
    })


@app.route('/api/session/<session_id>')
def get_session(session_id):
    if session_id not in sessions:
        return jsonify({'error': 'Session not found'}), 404

    session = sessions[session_id]

    return jsonify(session)


if __name__ == '__main__':
    app.run(debug=True, port=5000)