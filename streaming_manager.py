import asyncio
import logging
import threading

from server import VideoServer


class StreamingManager:
    def __init__(self):
        self.video_server = None
        self.server_thread = None
        self.event_loop = None
        self.is_streaming = False

    def start_server_in_thread(self):
        """Run the video server in a separate thread"""
        self.event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.event_loop)
        self.video_server = VideoServer()
        try:
            self.event_loop.run_until_complete(
                self.video_server.run(
                    websocket_host='0.0.0.0',
                    websocket_port=8765,
                    http_port=8766
                )
            )
        except Exception as e:
            logging.error(f"Error in video server thread: {str(e)}")
        finally:
            self.event_loop.close()

    def start_streaming(self):
        """Start the video streaming server"""
        if self.is_streaming:
            return False, "Stream is already running"
        
        try:
            self.server_thread = threading.Thread(
                target=self.start_server_in_thread,
                daemon=True
            )
            self.server_thread.start()
            self.is_streaming = True
            return True, "Streaming started successfully"
        except Exception as e:
            logging.error(f"Failed to start streaming: {str(e)}")
            return False, f"Failed to start streaming: {str(e)}"

    def stop_streaming(self):
        """Stop the video streaming server"""
        if not self.is_streaming:
            return False, "Stream is not running"
        
        try:
            if self.video_server:
                self.video_server.is_running = False
                if self.event_loop:
                    self.event_loop.call_soon_threadsafe(self.event_loop.stop)
                self.video_server.cleanup()
                self.server_thread.join(timeout=5)
                self.video_server = None
                self.server_thread = None
                self.event_loop = None
            self.is_streaming = False
            return True, "Streaming stopped successfully"
        except Exception as e:
            logging.error(f"Failed to stop streaming: {str(e)}")
            return False, f"Failed to stop streaming: {str(e)}"
