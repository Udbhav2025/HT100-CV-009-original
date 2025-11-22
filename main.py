from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict

class VehicleCounter:
    def __init__(self, model_path='yolov8n.pt', skip_frames=2):
        """Initialize YOLOv8 model and tracking variables"""
        self.model = YOLO(model_path)
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        self.track_history = defaultdict(lambda: [])
        self.counted_ids = set()
        self.total_count = 0
        self.skip_frames = skip_frames  # Process every Nth frame
        self.detection_zone = None  # Will store the detection zone coordinates
        
    def detect_and_count(self, source, line_positions=[0.4, 0.6], detection_zone=None):
        """
        Detect and count vehicles crossing multiple lines
        
        Args:
            source: Video file path, camera index (0 for webcam), or image
            line_positions: List of line positions (0.0 to 1.0). Default [0.4, 0.6] creates two lines
            detection_zone: Dictionary defining the detection area. Options:
                - None: Detect entire frame (default)
                - {'type': 'rectangle', 'x1': 0.2, 'y1': 0.3, 'x2': 0.8, 'y2': 0.7}
                - {'type': 'polygon', 'points': [[0.2, 0.3], [0.8, 0.3], [0.9, 0.7], [0.1, 0.7]]}
                - {'type': 'center_strip', 'width': 0.6}  # Center strip of given width
                All coordinates are normalized (0.0 to 1.0)
        """
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print("Error: Could not open video source")
            return
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Calculate counting line positions
        line_y_positions = [int(frame_height * pos) for pos in line_positions]
        
        # Setup detection zone
        self.detection_zone = self._setup_detection_zone(detection_zone, frame_width, frame_height)
        
        print(f"Processing video... Press 'q' to quit")
        print(f"Video dimensions: {frame_width}x{frame_height} @ {fps}fps")
        print(f"Performance mode: Processing every {self.skip_frames} frame(s)")
        print(f"Counting lines at: {line_positions} ({line_y_positions} pixels)")
        if self.detection_zone:
            print(f"Detection zone: {detection_zone['type']}")
        
        # Create resizable window
        cv2.namedWindow('Vehicle Counter', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Vehicle Counter', 1280, 720)  # Initial size
        
        frame_count = 0
        last_results = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process only every Nth frame for better performance
            if frame_count % self.skip_frames == 0:
                # Run YOLOv8 tracking with optimized settings
                results = self.model.track(
                    frame, 
                    persist=True, 
                    classes=self.vehicle_classes, 
                    verbose=False,
                    imgsz=640,  # Smaller image size for faster processing
                    conf=0.3,   # Lower confidence threshold
                    iou=0.5     # IOU threshold for NMS
                )
                last_results = results
            else:
                # Skip processing, use last results for display
                if last_results is None:
                    continue
                results = last_results
            
            # Draw detection zone
            if self.detection_zone:
                self._draw_detection_zone(frame)
            
            # Draw counting lines (clipped to detection zone if exists)
            colors = [(0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 255, 0)]  # Yellow, Magenta, Green, Yellow-Green
            for idx, line_y in enumerate(line_y_positions):
                color = colors[idx % len(colors)]
                
                if self.detection_zone:
                    # Draw line only within detection zone
                    self._draw_clipped_line(frame, line_y, color, idx)
                else:
                    # Draw full line across frame
                    cv2.line(frame, (0, line_y), (frame_width, line_y), color, 2)
                    cv2.putText(frame, f"LINE {idx + 1}", (10, line_y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Process detections
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                confidences = results[0].boxes.conf.cpu().numpy()
                
                for box, track_id, cls, conf in zip(boxes, track_ids, classes, confidences):
                    x1, y1, x2, y2 = map(int, box)
                    center_y = (y1 + y2) // 2
                    center_x = (x1 + x2) // 2
                    
                    # Check if vehicle center is within detection zone
                    if self.detection_zone and not self._point_in_zone(center_x, center_y):
                        continue  # Skip vehicles outside detection zone
                    
                    # Store track history
                    track = self.track_history[track_id]
                    track.append((center_x, center_y))
                    if len(track) > 30:
                        track.pop(0)
                    
                    # Check if vehicle crossed any of the lines
                    if len(track) > 1 and track_id not in self.counted_ids:
                        prev_y = track[-2][1]
                        curr_y = track[-1][1]
                        
                        # Check each counting line
                        for line_y in line_y_positions:
                            # Count if crossed from top to bottom or bottom to top
                            if (prev_y < line_y <= curr_y) or (prev_y > line_y >= curr_y):
                                self.counted_ids.add(track_id)
                                self.total_count += 1
                                print(f"✓ Vehicle ID:{track_id} counted (crossed line at y={line_y})")
                                break  # Count only once even if multiple lines crossed simultaneously
                    
                    # Draw bounding box and label
                    class_names = {2: 'Car', 3: 'Motorcycle', 5: 'Bus', 7: 'Truck'}
                    label = f"{class_names.get(cls, 'Vehicle')} ID:{track_id}"
                    color = (0, 255, 0) if track_id in self.counted_ids else (255, 0, 0)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Draw tracking line
                    points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [points], False, color, 2)
            
            # Display count
            cv2.putText(frame, f"Total Count: {self.total_count}", (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
            # Add instructions text
            cv2.putText(frame, "Press 'Q' to quit | Drag corners to resize window", 
                       (10, frame_height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Resize frame if too large for screen (optional, can be removed if you want full size)
            screen_height = 720
            if frame_height > screen_height:
                scale = screen_height / frame_height
                new_width = int(frame_width * scale)
                frame = cv2.resize(frame, (new_width, screen_height))
            
            # Show frame
            cv2.imshow('Vehicle Counter', frame)
            
            # Control playback speed - use minimal delay for real-time feel
            delay = 1  # Minimal delay for faster playback
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\nFinal count: {self.total_count} vehicles")
        return self.total_count
    
    def _setup_detection_zone(self, zone_config, width, height):
        """Convert normalized zone coordinates to pixel coordinates"""
        if zone_config is None:
            return None
        
        zone_type = zone_config['type']
        
        if zone_type == 'rectangle':
            return {
                'type': 'rectangle',
                'x1': int(zone_config['x1'] * width),
                'y1': int(zone_config['y1'] * height),
                'x2': int(zone_config['x2'] * width),
                'y2': int(zone_config['y2'] * height)
            }
        elif zone_type == 'polygon':
            points = [[int(p[0] * width), int(p[1] * height)] for p in zone_config['points']]
            return {
                'type': 'polygon',
                'points': np.array(points, dtype=np.int32)
            }
        elif zone_type == 'center_strip':
            strip_width = zone_config.get('width', 0.6)
            margin = (1.0 - strip_width) / 2
            return {
                'type': 'rectangle',
                'x1': int(margin * width),
                'y1': 0,
                'x2': int((1.0 - margin) * width),
                'y2': height
            }
        return None
    
    def _point_in_zone(self, x, y):
        """Check if a point is inside the detection zone"""
        if self.detection_zone is None:
            return True
        
        zone_type = self.detection_zone['type']
        
        if zone_type == 'rectangle':
            return (self.detection_zone['x1'] <= x <= self.detection_zone['x2'] and
                    self.detection_zone['y1'] <= y <= self.detection_zone['y2'])
        elif zone_type == 'polygon':
            point = np.array([x, y], dtype=np.int32)
            return cv2.pointPolygonTest(self.detection_zone['points'], (float(x), float(y)), False) >= 0
        return True
    
    def _draw_detection_zone(self, frame):
        """Draw the detection zone on the frame"""
        if self.detection_zone is None:
            return
        
        zone_type = self.detection_zone['type']
        overlay = frame.copy()
        
        if zone_type == 'rectangle':
            x1, y1 = self.detection_zone['x1'], self.detection_zone['y1']
            x2, y2 = self.detection_zone['x2'], self.detection_zone['y2']
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "DETECTION ZONE", (x1 + 10, y1 + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif zone_type == 'polygon':
            points = self.detection_zone['points'].reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [points], (0, 255, 0))
            cv2.polylines(frame, [points], True, (0, 255, 0), 2)
            cv2.putText(frame, "DETECTION ZONE", (points[0][0][0] + 10, points[0][0][1] + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Blend overlay with original frame for transparency effect
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
    
    def _draw_clipped_line(self, frame, line_y, color, line_idx):
        """Draw counting line clipped to detection zone boundaries"""
        zone_type = self.detection_zone['type']
        
        if zone_type == 'rectangle':
            x1 = self.detection_zone['x1']
            x2 = self.detection_zone['x2']
            y1 = self.detection_zone['y1']
            y2 = self.detection_zone['y2']
            
            # Only draw line if it's within the zone's y bounds
            if y1 <= line_y <= y2:
                cv2.line(frame, (x1, line_y), (x2, line_y), color, 2)
                # Position label inside zone
                label_x = x1 + 10
                label_y = line_y - 10 if line_y - 10 > y1 else line_y + 20
                cv2.putText(frame, f"LINE {line_idx + 1}", (label_x, label_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        elif zone_type == 'polygon':
            points = self.detection_zone['points']
            
            # Find intersection points of horizontal line with polygon edges
            intersections = []
            num_points = len(points)
            
            for i in range(num_points):
                p1 = points[i]
                p2 = points[(i + 1) % num_points]
                
                # Check if edge crosses the horizontal line
                if (p1[1] <= line_y <= p2[1]) or (p2[1] <= line_y <= p1[1]):
                    # Avoid division by zero
                    if p2[1] != p1[1]:
                        # Calculate x coordinate where edge intersects the line
                        t = (line_y - p1[1]) / (p2[1] - p1[1])
                        x = p1[0] + t * (p2[0] - p1[0])
                        intersections.append(int(x))
            
            # Draw line segments between intersection pairs
            intersections.sort()
            for i in range(0, len(intersections), 2):
                if i + 1 < len(intersections):
                    cv2.line(frame, (intersections[i], line_y), 
                            (intersections[i + 1], line_y), color, 2)
            
            # Add label if line intersects polygon
            if intersections:
                label_x = intersections[0] + 10
                label_y = line_y - 10
                cv2.putText(frame, f"LINE {line_idx + 1}", (label_x, label_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

if __name__ == "__main__":
    # ============================================
    # CONFIGURATION SECTION - EDIT THESE VALUES
    # ============================================
    
    # VIDEO SOURCE
    VIDEO_PATH = 'videos/854897-hd_1920_1080_30fps.mp4'  # Change to your video path
    
    # NUMBER OF COUNTING LINES (adjust as needed)
    # Options:
    # - 1 line: [0.5]
    # - 2 lines: [0.4, 0.6]
    # - 3 lines: [0.3, 0.5, 0.7]
    # - 4 lines: [0.25, 0.4, 0.6, 0.75]
    LINE_POSITIONS = [0.4, 0.43, 0.45, 0.6, 0.7, 0.8]  # CHANGE THIS
    
    # DETECTION ZONE (choose one option below and uncomment it)
    
    # Option 1: No zone - detect entire frame
    # DETECTION_ZONE = None
    
    # Option 2: Rectangle zone
    # DETECTION_ZONE = {
    #     'type': 'rectangle',
    #     'x1': 0.2,   # Left edge (0.0 = left, 1.0 = right)
    #     'y1': 0.3,   # Top edge (0.0 = top, 1.0 = bottom)
    #     'x2': 0.8,   # Right edge
    #     'y2': 0.7    # Bottom edge
    # }
    
    # Option 3: Center strip
    # DETECTION_ZONE = {
    #     'type': 'center_strip',
    #     'width': 0.6  # Width of center strip (0.0 to 1.0)
    # }
    
    # Option 4: Left half only
    # DETECTION_ZONE = {
    #     'type': 'rectangle',
    #     'x1': 0.0,
    #     'y1': 0.0,
    #     'x2': 0.5,
    #     'y2': 1.0
    # }
    
    # Option 5: Right half only
    # DETECTION_ZONE = {
    #     'type': 'rectangle',
    #     'x1': 0.5,
    #     'y1': 0.0,
    #     'x2': 1.0,
    #     'y2': 1.0
    # }
    
    # Option 6: Polygon (custom shape)
    DETECTION_ZONE = {
        'type': 'polygon',
        'points': [
            [0.382, 0.375],  # Top-left
            [0.66, 0.375],  # Top-right
            [0.95, 0.95],  # Bottom-right
            [0.2, 0.95]   # Bottom-left
        ]
    }
    
    # PERFORMANCE SETTINGS
    SKIP_FRAMES = 2  # Process every Nth frame (1=all frames, 2=every 2nd, 3=every 3rd)
    MODEL = 'yolov8n.pt'  # Options: yolov8n.pt (fastest), yolov8s.pt, yolov8m.pt (most accurate)
    
    # ============================================
    # RUN THE COUNTER (don't change below)
    # ============================================
    print("=" * 50)
    print("VEHICLE COUNTER CONFIGURATION")
    print("=" * 50)
    print(f"Video: {VIDEO_PATH}")
    print(f"Counting Lines: {len(LINE_POSITIONS)} line(s) at positions {LINE_POSITIONS}")
    print(f"Detection Zone: {DETECTION_ZONE['type'] if DETECTION_ZONE else 'Full Frame'}")
    print(f"Model: {MODEL}")
    print(f"Performance: Processing every {SKIP_FRAMES} frame(s)")
    print("=" * 50)
    print()
    
    counter = VehicleCounter(MODEL, skip_frames=SKIP_FRAMES)
    counter.detect_and_count(VIDEO_PATH, 
                            line_positions=LINE_POSITIONS, 
                            detection_zone=DETECTION_ZONE)
