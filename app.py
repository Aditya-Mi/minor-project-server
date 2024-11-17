# app.py
from flask import Flask, jsonify, request
from datetime import datetime
from streaming_manager import StreamingManager
from notification_manager import NotificationManager

app = Flask(__name__)
streaming_manager = StreamingManager()
notification_manager = NotificationManager('secrets/serviceAccountKey.json')

@app.route('/health')
def health_check():
    """
    Health check endpoint that returns streaming service status
    GET /health
    """
    try:
        health_data = {
            "status": "healthy",
            "streaming_status": {
                "is_active": streaming_manager.is_streaming,
                "server_thread_alive": bool(streaming_manager.server_thread and streaming_manager.server_thread.is_alive()) if streaming_manager.server_thread else False
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Include video server metrics if available
        if streaming_manager.video_server and streaming_manager.is_streaming:
            health_data.update({
                "camera_status": "connected" if streaming_manager.video_server.camera and 
                                              streaming_manager.video_server.camera.isOpened() else "disconnected",
                "metrics": {
                    "fps": round(streaming_manager.video_server.metrics['current_fps'], 2),
                    "processing_time_ms": round(streaming_manager.video_server.metrics['avg_processing_time'] * 1000, 2),
                    "dropped_frames": streaming_manager.video_server.metrics['dropped_frames'],
                    "queue_size": streaming_manager.video_server.metrics['queue_size'],
                    "total_frames": streaming_manager.video_server.frame_count,
                    "errors": streaming_manager.video_server.error_count
                }
            })
        
        return jsonify(health_data)
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/stream/start', methods=['POST'])
def start_stream():
    """Start the video stream"""
    success, message = streaming_manager.start_streaming()
    return jsonify({
        "success": success,
        "message": message
    }), 200 if success else 500

@app.route('/stream/stop', methods=['POST'])
def stop_stream():
    """Stop the video stream"""
    success, message = streaming_manager.stop_streaming()
    return jsonify({
        "success": success,
        "message": message
    }), 200 if success else 500
    
@app.route('/send-fire-alert', methods=['POST'])
def send_fire_alert():
    try:
        data = request.json
        fcm_token = data.get('fcm_token')
        
        if not fcm_token:
            return jsonify({
                'success': False,
                'error': 'FCM token is required'
            }), 400

        success, response = notification_manager.send_fire_alert(fcm_token)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Fire alert sent successfully',
                'response': response
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': response
            }), 500

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)