import streamlit as st
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image
import tempfile
import hashlib
import sqlite3
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Initialize session state
if 'user' not in st.session_state:
    st.session_state['user'] = None

# Database setup
conn = sqlite3.connect('users.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS users
             (username TEXT PRIMARY KEY, password TEXT)''')
conn.commit()

# Load the YOLO model
@st.cache_resource
def load_model():
    return YOLO('./model/best.pt')

model = load_model()

def draw_angled_corners(img, box, color, thickness=2, corner_length=20):
    pts = np.array(box).astype(np.int32).reshape((-1, 1, 2))
    for i in range(4):
        pt1 = tuple(pts[i][0])
        pt2 = tuple(pts[(i+1)%4][0])
        
        # Draw first half of the corner
        cv2.line(img, pt1, (int(pt1[0] + (pt2[0]-pt1[0])*0.2), int(pt1[1] + (pt2[1]-pt1[1])*0.2)), color, thickness)
        
        # Draw second half of the corner
        cv2.line(img, pt2, (int(pt2[0] - (pt2[0]-pt1[0])*0.2), int(pt2[1] - (pt2[1]-pt1[1])*0.2)), color, thickness)

def add_text_with_background(img, text, position, font_scale=0.7, thickness=2, text_color=(255,255,255), bg_color=(0,0,0), bg_alpha=0.5):
    if img is None or text is None or position is None:
        print("Error: Invalid input to add_text_with_background")
        return img

    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    
    if text_size is None or len(text_size) < 2 or text_size[0] is None:
        print("Error: Unable to get text size")
        return img

    (text_width, text_height) = text_size[0]
    text_offset_x, text_offset_y = position

    if any(v is None for v in [text_width, text_height, text_offset_x, text_offset_y]):
        print("Error: Invalid text dimensions or position")
        return img

    box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 10, text_offset_y - text_height - 10))

    # Check if box coordinates are within image boundaries
    if (box_coords[0][0] < 0 or box_coords[0][1] < 0 or 
        box_coords[1][0] > img.shape[1] or box_coords[1][1] > img.shape[0]):
        print("Error: Text box coordinates out of image boundaries")
        return img

    try:
        sub_img = img[box_coords[0][1]:box_coords[1][1], box_coords[0][0]:box_coords[1][0]]
        white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
        res = cv2.addWeighted(sub_img, 1-bg_alpha, white_rect, bg_alpha, 1.0)
        
        img[box_coords[0][1]:box_coords[1][1], box_coords[0][0]:box_coords[1][0]] = res
        cv2.putText(img, text, (text_offset_x+5, text_offset_y-5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
    except Exception as e:
        print(f"Error in adding text: {str(e)}")
        return img

    return img

def process_parking_lot(results):
    empty_count = 0
    filled_count = 0
    adjusted_boxes = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            # Use the rotated bounding box if available, otherwise use the regular box
            if hasattr(box, 'xyxyxyxy'):
                adjusted_box = box.xyxyxyxy[0].cpu().numpy()
            else:
                adjusted_box = box.xyxy[0].cpu().numpy()
                # Convert to rotated box format (4 corners)
                x1, y1, x2, y2 = adjusted_box
                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                width, height = x2 - x1, y2 - y1
                # Reduce the size of the box
                width *= 0.8
                height *= 0.8
                adjusted_box = np.array([
                    [center_x - width/2, center_y - height/2],
                    [center_x + width/2, center_y - height/2],
                    [center_x + width/2, center_y + height/2],
                    [center_x - width/2, center_y + height/2]
                ])
            
            if cls in [0, 3]:  # Empty
                empty_count += 1
                adjusted_boxes.append((adjusted_box, (0, 255, 0)))  # Green for empty
            elif cls in [1, 2, 4]:  # Filled
                filled_count += 1
                adjusted_boxes.append((adjusted_box, (0, 0, 255)))  # Red for filled
    return empty_count, filled_count, adjusted_boxes

def draw_boxes_and_counts(image, empty_count, filled_count, adjusted_boxes):
    if image is None:
        print("Error: Invalid input image")
        return image

    # Add total counts at the top with transparent background
    image = add_text_with_background(image, f"Empty: {empty_count} | Filled: {filled_count}", (10, 30), bg_color=(0, 0, 0))

    for box, color in adjusted_boxes:
        draw_angled_corners(image, box, color, 2)

    return image

def process_image(image):
    if image is None:
        print("Error: Invalid input image")
        return image, 0, 0

    results = model(image)
    empty_count, filled_count, adjusted_boxes = process_parking_lot(results)
    
    print(f"Debug: empty_count = {empty_count}, filled_count = {filled_count}")
    
    result_image = draw_boxes_and_counts(image.copy(), empty_count, filled_count, adjusted_boxes)
    return result_image, empty_count, filled_count

def process_video(video_file):
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(video_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    
    # Create a Streamlit video output
    output = st.empty()
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Process every 5th frame to reduce computation
        if frame_count % 5 == 0:
            results = model(frame)
            empty_count, filled_count, adjusted_boxes = process_parking_lot(results)
            
            # Draw boxes and add text
            frame_with_boxes = draw_boxes_and_counts(frame.copy(), empty_count, filled_count, adjusted_boxes)
            frame_with_boxes = add_text_with_background(frame_with_boxes, f"Frame: {frame_count}", (10, 70))
            
            # Display the frame
            output.image(frame_with_boxes, channels="BGR", use_column_width=True)
        
        # Check if the user has stopped the video
        if st.session_state.get('stop_video', False):
            break
    
    cap.release()

# New functions for user authentication
def hash_password(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_user(username, password):
    c.execute('SELECT * FROM users WHERE username=? AND password=?', (username, hash_password(password)))
    return c.fetchone() is not None

def create_user(username, password):
    try:
        c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hash_password(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

# Page functions
def landing_page():
    st.title('Welcome to Intelligent Parking Space Detection')
    st.write('This application helps you analyze parking spaces in images and videos.')
    st.write('Please log in or sign up to use the parking space analyzer.')
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button('Login'):
            st.session_state['page'] = 'login'
    with col2:
        if st.button('Sign Up'):
            st.session_state['page'] = 'signup'

def login_page():
    st.title('Login')
    username = st.text_input('Username')
    password = st.text_input('Password', type='password')
    if st.button('Login'):
        if check_user(username, password):
            st.session_state['user'] = username
            st.session_state['page'] = 'analyzer'
            st.success('Logged in successfully!')
        else:
            st.error('Invalid username or password')

def signup_page():
    st.title('Sign Up')
    username = st.text_input('Username')
    password = st.text_input('Password', type='password')
    if st.button('Sign Up'):
        if create_user(username, password):
            st.success('Account created successfully! Please log in.')
            st.session_state['page'] = 'login'
        else:
            st.error('Username already exists')

def analyzer_page():
    st.title('Parking Space Analyzer')
    st.write(f'Welcome, {st.session_state["user"]}!')
    
    if st.button('Logout'):
        st.session_state['user'] = None
        st.session_state['page'] = 'landing'
        st.experimental_rerun()

    uploaded_file = st.file_uploader("Choose an image or video file", type=["jpg", "jpeg", "png", "mp4"])

    if uploaded_file is not None:
        file_type = uploaded_file.type.split('/')[0]
        
        if file_type == 'image':
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            if st.button('Analyze Parking Spaces'):
                try:
                    result_image, empty_count, filled_count = process_image(np.array(image))
                    st.image(result_image, caption='Analyzed Image', use_column_width=True)
                    st.write(f"Empty spaces: {empty_count}")
                    st.write(f"Filled spaces: {filled_count}")
                except Exception as e:
                    st.error(f"An error occurred during image processing: {str(e)}")
        
        elif file_type == 'video':
            st.video(uploaded_file)
            
            if st.button('Analyze Video'):
                st.session_state['stop_video'] = False
                try:
                    process_video(uploaded_file)
                except Exception as e:
                    st.error(f"An error occurred during video processing: {str(e)}")
            
            if st.button('Stop Video'):
                st.session_state['stop_video'] = True

# Main app logic
def main():
    if 'page' not in st.session_state:
        st.session_state['page'] = 'landing'

    if st.session_state['page'] == 'landing':
        landing_page()
    elif st.session_state['page'] == 'login':
        login_page()
    elif st.session_state['page'] == 'signup':
        signup_page()
    elif st.session_state['page'] == 'analyzer':
        if st.session_state['user'] is None:
            st.session_state['page'] = 'landing'
            st.experimental_rerun()
        else:
            analyzer_page()

if __name__ == "__main__":
    main()