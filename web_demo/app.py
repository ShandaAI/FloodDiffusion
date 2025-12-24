"""
Flask server for real-time 3D motion generation demo
"""
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import json
import time
import threading
import argparse
from model_manager import get_model_manager

app = Flask(__name__)
CORS(app)

# Global model manager (lazy loaded)
model_manager = None
model_config_path = None  # Will be set once at startup

# Session tracking - only one active session can generate at a time
active_session_id = None  # The session ID currently generating
session_lock = threading.Lock()

# Frame consumption monitoring - detect if client disconnected by tracking frame consumption
last_frame_consumed_time = None
consumption_timeout = 5.0  # If no frame consumed for 5 seconds, assume client disconnected
consumption_monitor_thread = None
consumption_monitor_lock = threading.Lock()


def init_model():
    """Initialize model manager"""
    global model_manager
    if model_manager is None:
        if model_config_path is None:
            raise RuntimeError("model_config_path not set. Server not properly initialized.")
        print(f"Initializing model manager with config: {model_config_path}")
        model_manager = get_model_manager(config_path=model_config_path)
        print("Model manager ready!")
    return model_manager


def consumption_monitor():
    """Monitor frame consumption and auto-reset if client stops consuming"""
    global last_frame_consumed_time, active_session_id, model_manager
    
    while True:
        time.sleep(2.0)  # Check every 2 seconds
        
        # Read state with proper locking - no nested locks!
        should_reset = False
        current_session = None
        time_since_last_consumption = 0
        
        # First, check consumption time
        with consumption_monitor_lock:
            if last_frame_consumed_time is not None:
                time_since_last_consumption = time.time() - last_frame_consumed_time
                if time_since_last_consumption > consumption_timeout:
                    # Need to check if still generating before reset
                    if model_manager and model_manager.is_generating:
                        should_reset = True
        
        # Then, get current session (separate lock)
        if should_reset:
            with session_lock:
                current_session = active_session_id
        
        # Perform reset outside of locks to avoid deadlock
        if should_reset and current_session is not None:
            print(f"No frame consumed for {time_since_last_consumption:.1f}s - client disconnected, auto-resetting...")
            
            if model_manager:
                model_manager.reset()
                print("Generation reset due to client disconnect (no frame consumption)")
            
            # Clear state with proper locking - no nested locks!
            with session_lock:
                if active_session_id == current_session:
                    active_session_id = None
            
            with consumption_monitor_lock:
                last_frame_consumed_time = None


def start_consumption_monitor():
    """Start the consumption monitoring thread if not already running"""
    global consumption_monitor_thread
    
    if consumption_monitor_thread is None or not consumption_monitor_thread.is_alive():
        consumption_monitor_thread = threading.Thread(target=consumption_monitor, daemon=True)
        consumption_monitor_thread.start()
        print("Consumption monitor started")


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/api/start', methods=['POST'])
def start_generation():
    """Start generation with given text"""
    try:
        global active_session_id, last_frame_consumed_time
        
        data = request.json
        session_id = data.get('session_id')
        text = data.get('text', 'walk in a circle.')
        history_length = data.get('history_length', 30)
        smoothing_alpha = data.get('smoothing_alpha', None)  # Optional smoothing parameter
        denoise_steps = data.get('denoise_steps', None)  # Optional denoising steps
        force = data.get('force', False)  # Allow force takeover
        
        if not session_id:
            return jsonify({
                'status': 'error',
                'message': 'session_id is required'
            }), 400
        
        print(f"[Session {session_id}] Starting generation with text: {text}, history_length: {history_length}, force: {force}")
        
        # Initialize model if needed
        mm = init_model()
        
        # Check if another session is already generating
        need_force_takeover = False
        previous_session = None
        
        with session_lock:
            if active_session_id and active_session_id != session_id:
                if not force:
                    # Another session is active, return conflict
                    return jsonify({
                        'status': 'error',
                        'message': 'Another session is already generating.',
                        'conflict': True,
                        'active_session_id': active_session_id
                    }), 409
                else:
                    # Force takeover
                    print(f"[Session {session_id}] Force takeover from session {active_session_id}")
                    need_force_takeover = True
                    previous_session = active_session_id
            
            if mm.is_generating and active_session_id == session_id:
                return jsonify({
                    'status': 'error',
                    'message': 'Generation is already running for this session.'
                }), 400
            
            # Set this session as active
            active_session_id = session_id
        
        # Clear previous session's consumption tracking if force takeover (no nested locks)
        if need_force_takeover:
            with consumption_monitor_lock:
                last_frame_consumed_time = None
        
        # Reset and start generation with history length, smoothing, and denoise steps
        mm.reset(history_length=history_length, smoothing_alpha=smoothing_alpha, denoise_steps=denoise_steps)
        mm.start_generation(text, history_length=history_length)
        
        # Initialize consumption tracking (no nested locks)
        with consumption_monitor_lock:
            last_frame_consumed_time = time.time()
        
        # Start consumption monitoring
        start_consumption_monitor()
        print(f"[Session {session_id}] Consumption monitoring activated")
        
        return jsonify({
            'status': 'success',
            'message': f'Generation started with text: {text}, history_length: {history_length}',
            'session_id': session_id
        })
    except Exception as e:
        print(f"Error in start_generation: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/update_text', methods=['POST'])
def update_text():
    """Update the generation text"""
    try:
        data = request.json
        session_id = data.get('session_id')
        text = data.get('text', '')
        
        if not session_id:
            return jsonify({
                'status': 'error',
                'message': 'session_id is required'
            }), 400
        
        # Verify this is the active session
        with session_lock:
            if active_session_id != session_id:
                return jsonify({
                    'status': 'error',
                    'message': 'Not the active session'
                }), 403
        
        if model_manager is None:
            return jsonify({
                'status': 'error',
                'message': 'Model not initialized'
            }), 400
        
        model_manager.update_text(text)
        
        return jsonify({
            'status': 'success',
            'message': f'Text updated to: {text}'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/pause', methods=['POST'])
def pause_generation():
    """Pause generation (keeps state for resume)"""
    try:
        data = request.json if request.json else {}
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({
                'status': 'error',
                'message': 'session_id is required'
            }), 400
        
        # Verify this is the active session
        with session_lock:
            if active_session_id != session_id:
                return jsonify({
                    'status': 'error',
                    'message': 'Not the active session'
                }), 403
        
        if model_manager:
            model_manager.pause_generation()
        
        return jsonify({
            'status': 'success',
            'message': 'Generation paused'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/resume', methods=['POST'])
def resume_generation():
    """Resume generation from paused state"""
    try:
        global last_frame_consumed_time
        
        data = request.json if request.json else {}
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({
                'status': 'error',
                'message': 'session_id is required'
            }), 400
        
        # Verify this is the active session
        with session_lock:
            if active_session_id != session_id:
                return jsonify({
                    'status': 'error',
                    'message': 'Not the active session'
                }), 403
        
        if model_manager is None:
            return jsonify({
                'status': 'error',
                'message': 'Model not initialized'
            }), 400
        
        model_manager.resume_generation()
        
        # Reset consumption tracking when resuming
        with consumption_monitor_lock:
            last_frame_consumed_time = time.time()
        
        return jsonify({
            'status': 'success',
            'message': 'Generation resumed'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/reset', methods=['POST'])
def reset():
    """Reset generation state"""
    try:
        global active_session_id, last_frame_consumed_time
        
        data = request.json if request.json else {}
        session_id = data.get('session_id')
        history_length = data.get('history_length', 30)
        smoothing_alpha = data.get('smoothing_alpha', None)
        denoise_steps = data.get('denoise_steps', None)
        
        # If session_id provided, verify it's the active session
        if session_id:
            with session_lock:
                if active_session_id and active_session_id != session_id:
                    return jsonify({
                        'status': 'error',
                        'message': 'Not the active session'
                    }), 403
        
        if model_manager:
            model_manager.reset(history_length=history_length, smoothing_alpha=smoothing_alpha, denoise_steps=denoise_steps)
        
        # Clear the active session
        with session_lock:
            if active_session_id == session_id or not session_id:
                active_session_id = None
        
        # Clear consumption tracking
        with consumption_monitor_lock:
            last_frame_consumed_time = None
        
        params_msg = f", smoothing: {smoothing_alpha}" if smoothing_alpha is not None else ""
        params_msg += f", steps: {denoise_steps}" if denoise_steps is not None else ""
        print(f"[Session {session_id}] Reset complete, session cleared")
        
        return jsonify({
            'status': 'success',
            'message': f'Reset complete with history_length: {history_length}{params_msg}'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/get_frame', methods=['GET'])
def get_frame():
    """Get the next frame"""
    try:
        global last_frame_consumed_time
        
        session_id = request.args.get('session_id')
        
        if not session_id:
            return jsonify({
                'status': 'error',
                'message': 'session_id is required'
            }), 400
        
        # Verify this is the active session
        with session_lock:
            if active_session_id != session_id:
                return jsonify({
                    'status': 'error',
                    'message': 'Not the active session'
                }), 403
        
        if model_manager is None:
            return jsonify({
                'status': 'error',
                'message': 'Model not initialized'
            }), 400
        
        # Get next frame from buffer
        joints = model_manager.get_next_frame()
        
        if joints is not None:
            # Update last consumption time
            with consumption_monitor_lock:
                last_frame_consumed_time = time.time()
            
            # Convert numpy array to list for JSON
            joints_list = joints.tolist()
            return jsonify({
                'status': 'success',
                'joints': joints_list,
                'buffer_size': model_manager.frame_buffer.size()
            })
        else:
            return jsonify({
                'status': 'waiting',
                'message': 'No frame available yet',
                'buffer_size': model_manager.frame_buffer.size()
            })
    except Exception as e:
        print(f"Error in get_frame: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get generation status"""
    try:
        session_id = request.args.get('session_id')
        
        with session_lock:
            is_active_session = (session_id and active_session_id == session_id)
            current_active_session = active_session_id
        
        if model_manager is None:
            return jsonify({
                'initialized': False,
                'buffer_size': 0,
                'is_generating': False,
                'is_active_session': is_active_session,
                'active_session_id': current_active_session
            })
        
        status = model_manager.get_buffer_status()
        status['initialized'] = True
        status['is_active_session'] = is_active_session
        status['active_session_id'] = current_active_session
        
        return jsonify(status)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Flask server for real-time 3D motion generation')
    parser.add_argument('--config', type=str, default='configs/stream.yaml',
                        help='Path to config yaml file (default: configs/stream.yaml)')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to run the server on (default: 5000)')
    args = parser.parse_args()
    
    # Set config path (this is module-level code, no need for global declaration)
    model_config_path = args.config
    
    print("Starting Flask server...")
    print(f"Config file: {model_config_path}")
    print("Note: Model will be loaded on first generation request")
    app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True)

