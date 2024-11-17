# import asyncio
# import cv2
# import base64
# import json
# import websockets
# import logging
# import signal
# import sys
# from datetime import datetime
# from aiohttp import web
# import threading
# from collections import deque
# import time
# from concurrent.futures import ThreadPoolExecutor

# class VideoServer:
#     def __init__(self):
#         self.clients = set()
#         self.camera = None
#         self.is_running = True
#         self.frame_count = 0
#         self.error_count = 0
#         self.start_time = None
#         self.lock = threading.Lock()
#         self.app = web.Application()
#         self.app.router.add_get('/health', self.health_check)
        
#         self.metrics = {
#             'avg_processing_time': 0,
#             'current_fps': 0,
#             'dropped_frames': 0,
#             'queue_size': 0,
#             'camera_errors': 0,
#             'last_error': None
#         }
        
#         self.camera_index = 0  # Default camera index
#         self.camera_retries = 0
#         self.max_camera_retries = 3
#         self.last_frame_time = None
#         self.target_fps = 30

#     async def initialize_camera(self):
#         """Initialize the camera with enhanced error handling"""
#         if self.camera_retries >= self.max_camera_retries:
#             print("Exceeded maximum camera retry attempts")
#             return False

#         try:
#             if self.camera is not None:
#                 self.camera.release()
#                 await asyncio.sleep(1)  # Wait before retrying

#             print(f"Attempting to initialize camera (attempt {self.camera_retries + 1})")
            
#             # Try different camera indices if default fails
#             for idx in range(2):  # Try camera index 0 and 1
#                 try:
#                     self.camera = cv2.VideoCapture(idx)
#                     if self.camera.isOpened():
#                         self.camera_index = idx
#                         break
#                 except Exception as e:
#                     print(f"Failed to open camera at index {idx}: {str(e)}")
#                     if self.camera:
#                         self.camera.release()
#                     continue

#             if not self.camera or not self.camera.isOpened():
#                 raise Exception(f"Failed to open any camera")

#             # Configure camera
#             self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#             self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#             self.camera.set(cv2.CAP_PROP_FPS, self.target_fps)
#             self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffering

#             # Verify camera settings
#             actual_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
#             actual_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
#             actual_fps = self.camera.get(cv2.CAP_PROP_FPS)

#             print(f"Camera initialized successfully:")
#             print(f"Resolution: {actual_width}x{actual_height}")
#             print(f"FPS: {actual_fps}")
            
#             # Test frame capture
#             success, frame = self.camera.read()
#             if not success or frame is None:
#                 raise Exception("Failed to capture test frame")

#             self.start_time = datetime.now()
#             self.camera_retries = 0  # Reset retry counter on success
#             return True

#         except Exception as e:
#             self.metrics['camera_errors'] += 1
#             self.metrics['last_error'] = str(e)
#             print(f"Camera initialization error: {str(e)}")
#             self.camera_retries += 1
#             return False

#     async def register(self, websocket):
#         """Handle WebSocket connections with enhanced error handling"""
#         client_info = f"{websocket.remote_address}"
#         print(f"New client connected from {client_info}")
        
#         try:
#             # Verify camera is working before accepting connection
#             if not self.camera or not self.camera.isOpened():
#                 if not await self.initialize_camera():
#                     raise Exception("Camera not available")

#             self.clients.add(websocket)
#             print(f"Client {client_info} registered successfully")

#             try:
#                 while self.is_running:
#                     try:
#                         message = await asyncio.wait_for(websocket.recv(), timeout=30)
#                         if message == "ping":
#                             await websocket.send("pong")
#                     except asyncio.TimeoutError:
#                         # Send ping to keep connection alive
#                         await websocket.ping()
#                     except websockets.ConnectionClosed:
#                         break
#                     except Exception as e:
#                         print(f"Error handling message from {client_info}: {str(e)}")
#                         break

#             finally:
#                 await self.unregister_client(websocket)
#                 print(f"Client {client_info} disconnected")

#         except Exception as e:
#             print(f"Error in register for {client_info}: {str(e)}")
#             await self.unregister_client(websocket)

#     async def unregister_client(self, websocket):
#         """Safely unregister a client"""
#         try:
#             if websocket in self.clients:
#                 self.clients.remove(websocket)
#             await websocket.close()
#         except Exception as e:
#             print(f"Error unregistering client: {str(e)}")

#     async def broadcast_frames(self):
#         """Broadcast frames with enhanced error handling"""
#         consecutive_errors = 0
#         max_consecutive_errors = 5

#         while self.is_running:
#             try:
#                 if not self.clients:
#                     await asyncio.sleep(0.01)
#                     continue

#                 if not self.camera or not self.camera.isOpened():
#                     if consecutive_errors >= max_consecutive_errors:
#                         print("Too many consecutive camera errors, attempting to reinitialize...")
#                         await self.initialize_camera()
#                         consecutive_errors = 0
#                     await asyncio.sleep(0.1)
#                     continue

#                 success, frame = self.camera.read()
#                 if not success or frame is None:
#                     consecutive_errors += 1
#                     continue

#                 consecutive_errors = 0  # Reset error counter on success
#                 current_time = time.time()
                
#                 if self.last_frame_time:
#                     self.metrics['current_fps'] = 1.0 / (current_time - self.last_frame_time)
#                 self.last_frame_time = current_time

#                 # Encode frame with error handling
#                 try:
#                     _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
#                     frame_bytes = base64.b64encode(buffer).decode('utf-8')
#                 except Exception as e:
#                     print(f"Frame encoding error: {str(e)}")
#                     continue

#                 message = json.dumps({
#                     'frame': frame_bytes,
#                     'timestamp': datetime.now().isoformat(),
#                     'metrics': self.metrics
#                 })

#                 disconnected_clients = set()
#                 for websocket in self.clients.copy():
#                     try:
#                         await asyncio.wait_for(websocket.send(message), timeout=5.0)
#                         self.frame_count += 1
#                     except Exception as e:
#                         print(f"Error sending frame: {str(e)}")
#                         disconnected_clients.add(websocket)
#                         self.error_count += 1

#                 self.clients -= disconnected_clients

#             except Exception as e:
#                 print(f"Broadcast error: {str(e)}")
#                 consecutive_errors += 1
#                 await asyncio.sleep(0.1)

#     async def health_check(self, request):
#         """Enhanced health check endpoint with detailed status"""
#         try:
#             camera_status = "connected" if self.camera and self.camera.isOpened() else "disconnected"
#             if camera_status == "disconnected" and self.metrics['last_error']:
#                 camera_status = f"disconnected: {self.metrics['last_error']}"

#             return web.json_response({
#                 "status": "healthy" if camera_status == "connected" else "degraded",
#                 "camera_status": camera_status,
#                 "metrics": {
#                     "fps": round(self.metrics['current_fps'], 2),
#                     "processing_time_ms": round(self.metrics['avg_processing_time'] * 1000, 2),
#                     "dropped_frames": self.metrics['dropped_frames'],
#                     "queue_size": self.metrics['queue_size'],
#                     "total_frames": self.frame_count,
#                     "errors": self.error_count,
#                     "camera_errors": self.metrics['camera_errors']
#                 },
#                 "streaming_status": {
#                     "is_active": self.is_running,
#                     "server_thread_alive": True
#                 },
#                 "timestamp": datetime.now().isoformat()
#             })
#         except Exception as e:
#             return web.json_response({
#                 "status": "unhealthy",
#                 "error": str(e)
#             }, status=500)

#     async def run(self, websocket_host='0.0.0.0', websocket_port=8765, http_port=8766):
#         """Run server with enhanced error handling"""
#         print("Starting server...")
        
#         # Initialize camera first
#         if not await self.initialize_camera():
#             print("Initial camera initialization failed, continuing with retry logic...")

#         try:
#             # Start HTTP server
#             runner = web.AppRunner(self.app)
#             await runner.setup()
#             site = web.TCPSite(runner, websocket_host, http_port)
#             await site.start()
#             print(f"Health check server running on http://{websocket_host}:{http_port}/health")

#             # Start WebSocket server
#             websocket_server = await websockets.serve(
#                 self.register,
#                 websocket_host,
#                 websocket_port,
#                 ping_interval=20,
#                 ping_timeout=30,
#                 max_size=None  # No limit on message size
#             )
            
#             print(f"WebSocket server running on ws://{websocket_host}:{websocket_port}")
            
#             # Run all tasks
#             await asyncio.gather(
#                 self.broadcast_frames(),
#                 asyncio.Future()  # Keep server running
#             )

#         except Exception as e:
#             print(f"Server error: {str(e)}")
#         finally:
#             self.cleanup()
#             await runner.cleanup()

#     def cleanup(self):
#         """Clean up resources"""
#         print("Cleaning up resources...")
#         self.is_running = False
#         if self.camera:
#             self.camera.release()
#         print("Cleanup completed")

# if __name__ == '__main__':
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s [%(levelname)s] %(message)s',
#         handlers=[
#             logging.FileHandler('video_server.log'),
#             logging.StreamHandler(sys.stdout)
#         ]
#     )
    
#     def signal_handler(sig, frame):
#         print("Shutting down...")
#         asyncio.get_event_loop().stop()
#         sys.exit(0)
    
#     for sig in (signal.SIGINT, signal.SIGTERM):
#         signal.signal(sig, signal_handler)
    
#     server = VideoServer()
#     asyncio.run(server.run())

import asyncio
import cv2
import base64
import json
import websockets
import logging
import signal
import sys
from datetime import datetime
from aiohttp import web
import threading
from collections import deque
import time
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO
import numpy as np

class VideoServer:
    def __init__(self):
        self.clients = set()
        self.camera = None
        self.is_running = True
        self.frame_count = 0
        self.error_count = 0
        self.start_time = None
        self.lock = threading.Lock()
        self.app = web.Application()
        self.app.router.add_get('/health', self.health_check)
        
        # Initialize YOLO model
        self.model = None
        
        self.metrics = {
            'avg_processing_time': 0,
            'current_fps': 0,
            'dropped_frames': 0,
            'queue_size': 0,
            'camera_errors': 0,
            'last_error': None,
            'inference_time': 0,
            'person_count': 0
        }
        
        self.camera_index = 0
        self.camera_retries = 0
        self.max_camera_retries = 3
        self.last_frame_time = None
        self.target_fps = 24
        
        # Create thread pool for inference
        self.executor = ThreadPoolExecutor(max_workers=1)

    async def initialize_model(self):
        """Initialize YOLO model"""
        try:
            print("Loading YOLO model...")
            self.model = YOLO('yolov8n.pt')  # or use 'yolov8s.pt' for better accuracy
            print("YOLO model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading YOLO model: {str(e)}")
            return False

    def process_detections(self, frame):
        """Run YOLO inference on frame for person detection"""
        try:
            start_time = time.time()
            
            # Run inference
            results = self.model(frame, classes=[0])  # class 0 is person in COCO dataset
            
            # Update inference time metric
            self.metrics['inference_time'] = time.time() - start_time
            
            # Get detections
            detections = []
            person_count = 0
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    
                    detection = {
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'label': 'person'
                    }
                    detections.append(detection)
                    person_count += 1
                    
                    # Draw detection on frame
                    cv2.rectangle(frame, 
                                (int(x1), int(y1)), 
                                (int(x2), int(y2)), 
                                (0, 255, 0), 2)
                    label = f"person {conf:.2f}"
                    cv2.putText(frame, label, 
                              (int(x1), int(y1) - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 
                              0.5, (0, 255, 0), 2)
            
            # Update person count metric
            self.metrics['person_count'] = person_count
            
            return frame, detections
            
        except Exception as e:
            print(f"Inference error: {str(e)}")
            return frame, []

    async def initialize_camera(self):
        """Initialize the camera with enhanced error handling"""
        if self.camera_retries >= self.max_camera_retries:
            print("Exceeded maximum camera retry attempts")
            return False

        try:
            if self.camera is not None:
                self.camera.release()
                await asyncio.sleep(1)

            print(f"Attempting to initialize camera (attempt {self.camera_retries + 1})")
            
            for idx in range(2):
                try:
                    self.camera = cv2.VideoCapture(idx)
                    if self.camera.isOpened():
                        self.camera_index = idx
                        break
                except Exception as e:
                    print(f"Failed to open camera at index {idx}: {str(e)}")
                    if self.camera:
                        self.camera.release()
                    continue

            if not self.camera or not self.camera.isOpened():
                raise Exception(f"Failed to open any camera")

            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, self.target_fps)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            actual_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)

            print(f"Camera initialized successfully:")
            print(f"Resolution: {actual_width}x{actual_height}")
            print(f"FPS: {actual_fps}")
            
            success, frame = self.camera.read()
            if not success or frame is None:
                raise Exception("Failed to capture test frame")

            self.start_time = datetime.now()
            self.camera_retries = 0
            return True

        except Exception as e:
            self.metrics['camera_errors'] += 1
            self.metrics['last_error'] = str(e)
            print(f"Camera initialization error: {str(e)}")
            self.camera_retries += 1
            return False

    async def broadcast_frames(self):
        """Broadcast frames with enhanced error handling and person detection"""
        consecutive_errors = 0
        max_consecutive_errors = 5

        while self.is_running:
            try:
                if not self.clients:
                    await asyncio.sleep(0.01)
                    continue

                if not self.camera or not self.camera.isOpened():
                    if consecutive_errors >= max_consecutive_errors:
                        print("Too many consecutive camera errors, attempting to reinitialize...")
                        await self.initialize_camera()
                        consecutive_errors = 0
                    await asyncio.sleep(0.1)
                    continue

                success, frame = self.camera.read()
                if not success or frame is None:
                    consecutive_errors += 1
                    continue

                consecutive_errors = 0
                current_time = time.time()
                
                if self.last_frame_time:
                    self.metrics['current_fps'] = 1.0 / (current_time - self.last_frame_time)
                self.last_frame_time = current_time

                # Run YOLO inference
                processed_frame, detections = await asyncio.get_event_loop().run_in_executor(
                    self.executor, 
                    self.process_detections, 
                    frame.copy()
                )

                try:
                    _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    frame_bytes = base64.b64encode(buffer).decode('utf-8')
                except Exception as e:
                    print(f"Frame encoding error: {str(e)}")
                    continue

                message = json.dumps({
                    'frame': frame_bytes,
                    'timestamp': datetime.now().isoformat(),
                    'metrics': self.metrics,
                    'detections': detections
                })

                disconnected_clients = set()
                for websocket in self.clients.copy():
                    try:
                        await asyncio.wait_for(websocket.send(message), timeout=5.0)
                        self.frame_count += 1
                    except Exception as e:
                        print(f"Error sending frame: {str(e)}")
                        disconnected_clients.add(websocket)
                        self.error_count += 1

                self.clients -= disconnected_clients

            except Exception as e:
                print(f"Broadcast error: {str(e)}")
                consecutive_errors += 1
                await asyncio.sleep(0.1)

    async def register(self, websocket):
        """Handle WebSocket connections with enhanced error handling"""
        client_info = f"{websocket.remote_address}"
        print(f"New client connected from {client_info}")
        
        try:
            if not self.camera or not self.camera.isOpened():
                if not await self.initialize_camera():
                    raise Exception("Camera not available")

            self.clients.add(websocket)
            print(f"Client {client_info} registered successfully")

            try:
                while self.is_running:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=30)
                        if message == "ping":
                            await websocket.send("pong")
                    except asyncio.TimeoutError:
                        await websocket.ping()
                    except websockets.ConnectionClosed:
                        break
                    except Exception as e:
                        print(f"Error handling message from {client_info}: {str(e)}")
                        break

            finally:
                await self.unregister_client(websocket)
                print(f"Client {client_info} disconnected")

        except Exception as e:
            print(f"Error in register for {client_info}: {str(e)}")
            await self.unregister_client(websocket)

    async def unregister_client(self, websocket):
        """Safely unregister a client"""
        try:
            if websocket in self.clients:
                self.clients.remove(websocket)
            await websocket.close()
        except Exception as e:
            print(f"Error unregistering client: {str(e)}")

    async def health_check(self, request):
        """Enhanced health check endpoint with detailed status"""
        try:
            camera_status = "connected" if self.camera and self.camera.isOpened() else "disconnected"
            if camera_status == "disconnected" and self.metrics['last_error']:
                camera_status = f"disconnected: {self.metrics['last_error']}"

            return web.json_response({
                "status": "healthy" if camera_status == "connected" else "degraded",
                "camera_status": camera_status,
                "metrics": {
                    "fps": round(self.metrics['current_fps'], 2),
                    "processing_time_ms": round(self.metrics['avg_processing_time'] * 1000, 2),
                    "dropped_frames": self.metrics['dropped_frames'],
                    "queue_size": self.metrics['queue_size'],
                    "total_frames": self.frame_count,
                    "errors": self.error_count,
                    "camera_errors": self.metrics['camera_errors'],
                    "person_count": self.metrics['person_count'],
                    "inference_time": round(self.metrics['inference_time'] * 1000, 2)
                },
                "streaming_status": {
                    "is_active": self.is_running,
                    "server_thread_alive": True
                },
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            return web.json_response({
                "status": "unhealthy",
                "error": str(e)
            }, status=500)

    async def run(self, websocket_host='0.0.0.0', websocket_port=8765, http_port=8766):
        """Run server with enhanced error handling"""
        print("Starting server...")
        
        # Initialize YOLO model first
        if not await self.initialize_model():
            print("Failed to initialize YOLO model")
            return
        
        # Initialize camera
        if not await self.initialize_camera():
            print("Initial camera initialization failed, continuing with retry logic...")

        try:
            # Start HTTP server
            runner = web.AppRunner(self.app)
            await runner.setup()
            site = web.TCPSite(runner, websocket_host, http_port)
            await site.start()
            print(f"Health check server running on http://{websocket_host}:{http_port}/health")

            # Start WebSocket server
            websocket_server = await websockets.serve(
                self.register,
                websocket_host,
                websocket_port,
                ping_interval=20,
                ping_timeout=30,
                max_size=None
            )
            
            print(f"WebSocket server running on ws://{websocket_host}:{websocket_port}")
            
            await asyncio.gather(
                self.broadcast_frames(),
                asyncio.Future()
            )

        except Exception as e:
            print(f"Server error: {str(e)}")
        finally:
            self.cleanup()
            await runner.cleanup()

    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up resources...")
        self.is_running = False
        if self.camera:
            self.camera.release()
        self.executor.shutdown(wait=True)
        print("Cleanup completed")

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler('video_server.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    def signal_handler(sig, frame):
        print("Shutting down...")
        asyncio.get_event_loop().stop()
        sys.exit(0)
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, signal_handler)
    
    server = VideoServer()
    asyncio.run(server.run())