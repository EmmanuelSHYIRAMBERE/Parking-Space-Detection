# app.py

import time
import base64
import os
import io
import pandas as pd
import streamlit as st
from streamlit import session_state
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import sqlite3
import hashlib
import tempfile
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from vectors import EmbeddingsManager
from chatbot import ChatbotManager

# Initialize session state for parking system
if 'user' not in st.session_state:
    st.session_state.update({
        'user': None,
        'temp_pdf_path': None,
        'chatbot_manager': None,
        'messages': [],
        'parking_stats': {'empty': 0, 'filled': 0},
        'conn': None,
        'cursor': None
    })

# Database setup and management
def setup_database():
    """Initialize database connection and create necessary tables."""
    conn = sqlite3.connect('integrated_system.db', check_same_thread=False)
    c = conn.cursor()

    # Create users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, 
                  password TEXT,
                  created_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')

    # Create comprehensive parking analytics table
    c.execute('''CREATE TABLE IF NOT EXISTS parking_analytics
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  empty_spots INTEGER,
                  filled_spots INTEGER,
                  efficiency REAL,
                  revenue REAL,
                  peak_hour TEXT,
                  image_path TEXT,
                  notes TEXT)''')

    # Create daily summary table for faster querying
    c.execute('''CREATE TABLE IF NOT EXISTS daily_summaries
                 (date DATE PRIMARY KEY,
                  avg_empty_spots REAL,
                  avg_filled_spots REAL,
                  avg_efficiency REAL,
                  total_revenue REAL,
                  peak_hours TEXT,
                  total_vehicles INTEGER)''')

    # Create document processing table
    c.execute('''CREATE TABLE IF NOT EXISTS document_processing
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  file_path TEXT,
                  processing_date DATETIME,
                  status TEXT,
                  error_message TEXT)''')

    migrate_database(c, conn)
    return conn, c


def migrate_database(c, conn):
    """Perform necessary database migrations."""

    # Check if email column exists
    c.execute("PRAGMA table_info(users)")
    columns = [column[1] for column in c.fetchall()]

    if 'email' not in columns:
        c.execute("ALTER TABLE users ADD COLUMN email TEXT")

    if 'profile_pic' not in columns:
        c.execute("ALTER TABLE users ADD COLUMN profile_pic BLOB")

    # Check if columns exist in parking_analytics
    c.execute("PRAGMA table_info(parking_analytics)")
    columns = [column[1] for column in c.fetchall()]

    columns_to_add = {
        'efficiency': 'REAL',
        'revenue': 'REAL',
        'peak_hour': 'TEXT',
        'image_path': 'TEXT',
        'notes': 'TEXT'
    }

    for column, data_type in columns_to_add.items():
        if column not in columns:
            c.execute(f"ALTER TABLE parking_analytics ADD COLUMN {column} {data_type}")
            conn.commit()
            print(f"Added {column} column to parking_analytics table")

    # Ensure document_processing table exists
    c.execute('''CREATE TABLE IF NOT EXISTS document_processing
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  file_path TEXT,
                  processing_date DATETIME,
                  status TEXT,
                  error_message TEXT)''')
    conn.commit()
    print("Ensured document_processing table exists")

    return conn, c

# Load the YOLO model for parking detection
@st.cache_resource
def load_parking_model():
    """Load and cache the YOLO model."""
    return YOLO('./model/best.pt')

# Authentication functions
def hash_password(password):
    """Create a secure hash of the password."""
    return hashlib.sha256(str.encode(password)).hexdigest()

def create_user(username, password, email):
    try:
        c = st.session_state.cursor
        c.execute('INSERT INTO users (username, password, email) VALUES (?, ?, ?)',
                 (username, hash_password(password), email))
        st.session_state.conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def check_user(username, password):
    c = st.session_state.cursor
    c.execute('SELECT username, email, profile_pic FROM users WHERE username=? AND password=?',
              (username, hash_password(password)))
    user = c.fetchone()
    if user:
        st.session_state.user = {
            'username': user[0],
            'email': user[1],
            'profile_pic': user[2]
        }
        return True
    return False

def update_password(username, new_password):
    c = st.session_state.cursor
    c.execute('UPDATE users SET password=? WHERE username=?',
              (hash_password(new_password), username))
    st.session_state.conn.commit()

def update_profile_pic(username, profile_pic):
    c = st.session_state.cursor
    c.execute('UPDATE users SET profile_pic=? WHERE username=?',
              (profile_pic, username))
    st.session_state.conn.commit()

def get_profile_pic(username):
    c = st.session_state.cursor
    c.execute('SELECT profile_pic FROM users WHERE username=?', (username,))
    result = c.fetchone()
    return result[0] if result else None


# User profile management
def render_user_profile():
    st.title("User Profile")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Profile Information")
        st.write(f"Username: {st.session_state.user['username']}")
        st.write(f"Email: {st.session_state.user['email']}")

        new_email = st.text_input("New Email")
        if st.button("Update Email"):
            c = st.session_state.cursor
            c.execute('UPDATE users SET email=? WHERE username=?',
                      (new_email, st.session_state.user['username']))
            st.session_state.conn.commit()
            st.session_state.user['email'] = new_email
            st.success("Email updated successfully!")

        st.subheader("Change Password")
        current_password = st.text_input("Current Password", type="password")
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Confirm New Password", type="password")

        if st.button("Change Password"):
            if check_user(st.session_state.user['username'], current_password):
                if new_password == confirm_password:
                    update_password(st.session_state.user['username'], new_password)
                    st.success("Password changed successfully!")
                else:
                    st.error("New passwords do not match.")
            else:
                st.error("Current password is incorrect.")

    with col2:
        st.subheader("Profile Picture")
        profile_pic = get_profile_pic(st.session_state.user['username'])
        if profile_pic:
            st.image(profile_pic, width=200)
        else:
            st.info("No profile picture uploaded yet.")

        uploaded_file = st.file_uploader("Choose a profile picture", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()

            update_profile_pic(st.session_state.user['username'], img_byte_arr)
            st.session_state.user['profile_pic'] = img_byte_arr
            st.success("Profile picture updated successfully!")
            st.rerun()

# Parking Analysis Functions
def process_parking_image(image, notes=""):
    """Process parking image and store results in database."""
    model = load_parking_model()
    results = model(image)
    empty_count, filled_count = 0, 0
    
    # Process detection results
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            if cls in [0, 3]:  # Empty spots
                empty_count += 1
            elif cls in [1, 2, 4]:  # Filled spots
                filled_count += 1
    
    # Save image to disk
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = f"parking_images/{timestamp}.jpg"
    os.makedirs("parking_images", exist_ok=True)
    if isinstance(image, Image.Image):
        image.save(image_path)
    else:
        cv2.imwrite(image_path, image)
    
    # Calculate metrics
    total_spots = empty_count + filled_count
    efficiency = filled_count / total_spots if total_spots > 0 else 0
    current_hour = datetime.now().hour
    is_peak = 7 <= current_hour <= 19
    peak_hour = f"{current_hour}:00" if is_peak else "Non-peak"
    
    # Calculate revenue (example: $5 per filled spot per hour)
    hourly_rate = 5
    revenue = filled_count * hourly_rate
    
    # Store results in database
    c = st.session_state.cursor
    c.execute('''INSERT INTO parking_analytics 
                 (empty_spots, filled_spots, efficiency, revenue, peak_hour, image_path, notes)
                 VALUES (?, ?, ?, ?, ?, ?, ?)''',
              (empty_count, filled_count, efficiency, revenue, peak_hour, image_path, notes))
    
    # Update daily summary
    update_daily_summary(datetime.now().date())
    
    st.session_state.conn.commit()
    
    return empty_count, filled_count, efficiency, revenue, peak_hour

def update_daily_summary(date):
    """Update or create daily summary for the given date."""
    c = st.session_state.cursor
    
    # Get daily statistics
    c.execute('''SELECT 
                   avg(empty_spots) as avg_empty,
                   avg(filled_spots) as avg_filled,
                   avg(efficiency) as avg_eff,
                   sum(revenue) as total_rev,
                   group_concat(distinct peak_hour) as peak_hrs,
                   sum(filled_spots) as total_vehicles
                 FROM parking_analytics 
                 WHERE date(timestamp) = ?''', (date,))
    
    stats = c.fetchone()
    if stats[0] is not None:  # If we have data for this date
        c.execute('''INSERT OR REPLACE INTO daily_summaries
                     (date, avg_empty_spots, avg_filled_spots, avg_efficiency,
                      total_revenue, peak_hours, total_vehicles)
                     VALUES (?, ?, ?, ?, ?, ?, ?)''',
                  (date, *stats))
        st.session_state.conn.commit()

def get_historical_data(days=30):
    """Retrieve historical parking data from the database."""
    conn = st.session_state.conn
    
    query = '''
    SELECT 
        date as Date,
        avg_empty_spots as Empty_Spots,
        avg_filled_spots as Filled_Spots,
        avg_efficiency as Parking_Efficiency,
        peak_hours as Peak_Hours,
        total_revenue as Revenue,
        total_vehicles as Vehicles
    FROM daily_summaries
    WHERE date >= date('now', ?)
    ORDER BY date
    '''
    
    df = pd.read_sql_query(query, conn, params=(f'-{days} days',))
    
    if not df.empty:
        # Convert string columns to appropriate types
        df['Empty_Spots'] = df['Empty_Spots'].astype(float).round(0)
        df['Filled_Spots'] = df['Filled_Spots'].astype(float).round(0)
        df['Parking_Efficiency'] = df['Parking_Efficiency'].astype(float)
        df['Revenue'] = df['Revenue'].astype(float).round(2)
        df['Vehicles'] = df['Vehicles'].astype(int)
    
    return df

# PDF Processing Functions
def process_pdf_document(pdf_path, embeddings_manager):
    """Process and store PDF document contents."""
    try:
        result = embeddings_manager.create_embeddings(pdf_path)
        # Store PDF processing record in database
        c = st.session_state.cursor
        c.execute('''INSERT INTO document_processing 
                     (file_path, processing_date, status)
                     VALUES (?, ?, ?)''',
                  (pdf_path, datetime.now(), 'success'))
        st.session_state.conn.commit()
        return result
    except Exception as e:
        # Log error in database
        c = st.session_state.cursor
        c.execute('''INSERT INTO document_processing 
                     (file_path, processing_date, status, error_message)
                     VALUES (?, ?, ?, ?)''',
                  (pdf_path, datetime.now(), 'error', str(e)))
        st.session_state.conn.commit()
        raise e

# Data Analysis Functions
def render_data_analysis():
    """Render the comprehensive data analysis dashboard."""
    st.title("📊 Parking Data Analysis")
    
    # Get historical data
    df = get_historical_data()
    
    if df.empty:
        st.warning("No historical data available. Please analyze some parking images first.")
        return
    
    # Create dashboard tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Real-time Metrics",
        "🚗 Parking Occupancy", 
        "💰 Revenue Analysis",
        "📊 Advanced Analytics"
    ])
    
    with tab1:
        st.header("Real-time Parking Metrics")
        
        # Current metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            current_empty = df['Empty_Spots'].iloc[-1]
            previous_empty = df['Empty_Spots'].iloc[-2] if len(df) > 1 else current_empty
            st.metric(
                label="Empty Spots", 
                value=f"{current_empty:,.0f}",
                delta=f"{current_empty - previous_empty:,.0f}"
            )
        
        with col2:
            current_filled = df['Filled_Spots'].iloc[-1]
            previous_filled = df['Filled_Spots'].iloc[-2] if len(df) > 1 else current_filled
            st.metric(
                label="Filled Spots", 
                value=f"{current_filled:,.0f}",
                delta=f"{current_filled - previous_filled:,.0f}"
            )
        
        with col3:
            current_efficiency = df['Parking_Efficiency'].iloc[-1]
            previous_efficiency = df['Parking_Efficiency'].iloc[-2] if len(df) > 1 else current_efficiency
            st.metric(
                label="Parking Efficiency", 
                value=f"{current_efficiency:.1%}",
                delta=f"{(current_efficiency - previous_efficiency):.1%}"
            )
        
        # Real-time trend chart
        st.subheader("Today's Parking Trends")
        today_data = get_todays_data()
        if not today_data.empty:
            fig_today = px.line(today_data, x='timestamp', y=['empty_spots', 'filled_spots'],
                              title="Today's Parking Occupancy")
            st.plotly_chart(fig_today, use_container_width=True)
    
    with tab2:
        st.header("Parking Occupancy Analysis")
        
        # Occupancy trend
        fig_occupancy = px.bar(
            df, 
            x='Date', 
            y=['Empty_Spots', 'Filled_Spots'],
            title='Parking Occupancy Trend',
            barmode='stack'
        )
        fig_occupancy.update_layout(yaxis_title="Number of Spots")
        st.plotly_chart(fig_occupancy, use_container_width=True)
        
        # Efficiency trend
        fig_efficiency = px.line(
            df, 
            x='Date', 
            y='Parking_Efficiency',
            title='Parking Efficiency Over Time'
        )
        fig_efficiency.update_layout(yaxis_title="Efficiency")
        st.plotly_chart(fig_efficiency, use_container_width=True)
        
        # Vehicle count trend
        fig_vehicles = px.bar(
            df,
            x='Date',
            y='Vehicles',
            title='Daily Vehicle Count'
        )
        st.plotly_chart(fig_vehicles, use_container_width=True)
    
    with tab3:
        st.header("Revenue Analysis")
        
        # Daily revenue trend
        fig_revenue = px.line(
            df, 
            x='Date', 
            y='Revenue',
            title='Daily Parking Revenue'
        )
        fig_revenue.update_layout(yaxis_title="Revenue ($)")
        st.plotly_chart(fig_revenue, use_container_width=True)
        
        # Revenue by peak hours
        fig_peak_hours = px.bar(
            df, 
            x='Date', 
            y='Revenue',
            color='Peak_Hours',
            title='Revenue by Peak Hours'
        )
        fig_peak_hours.update_layout(yaxis_title="Revenue ($)")
        st.plotly_chart(fig_peak_hours, use_container_width=True)
        
        # Revenue metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            total_revenue = df['Revenue'].sum()
            st.metric("Total Revenue", f"${total_revenue:,.2f}")
        with col2:
            avg_daily_revenue = df['Revenue'].mean()
            st.metric("Average Daily Revenue", f"${avg_daily_revenue:,.2f}")
        with col3:
            revenue_trend = df['Revenue'].pct_change().mean()
            st.metric("Revenue Trend", f"{revenue_trend:.1%}")
    
    with tab4:
        st.header("Advanced Analytics")
        
        # Correlation analysis
        correlation = df[['Empty_Spots', 'Filled_Spots', 'Parking_Efficiency', 'Revenue']].corr()
        fig_corr = px.imshow(correlation, 
                            title='Correlation Matrix',
                            color_continuous_scale='RdBu')
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Weekly patterns
        df['WeekDay'] = pd.to_datetime(df['Date']).dt.day_name()
        weekly_stats = df.groupby('WeekDay')['Parking_Efficiency'].mean().reset_index()
        fig_weekly = px.bar(weekly_stats,
                           x='WeekDay',
                           y='Parking_Efficiency',
                           title='Average Parking Efficiency by Day of Week')
        st.plotly_chart(fig_weekly, use_container_width=True)

def get_todays_data():
    """Retrieve today's parking data from the database."""
    conn = st.session_state.conn
    query = '''
    SELECT timestamp, empty_spots, filled_spots, efficiency, revenue
    FROM parking_analytics
    WHERE date(timestamp) = date('now')
    ORDER BY timestamp
    '''
    return pd.read_sql_query(query, conn)

def main():
    """Main application function."""

    # Initialize database connection if not already done
    if 'conn' not in st.session_state or 'cursor' not in st.session_state:
        st.session_state.conn, st.session_state.cursor = setup_database()

    # Initialize database connection if not already done
    if st.session_state.conn is None or st.session_state.cursor is None:
        st.session_state.conn, st.session_state.cursor = setup_database()
    
    # Set page configuration
    st.set_page_config(
        page_title="Intelligent Parking Space Detection",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar navigation
    with st.sidebar:
        st.image("logo.png", use_column_width=True)
        st.markdown("### 🚗 Smart Parking Assistant")
        st.markdown("---")

        if st.session_state.user is None:
            login_tab, register_tab = st.tabs(["Login", "Register"])

            with login_tab:
                login_user = st.text_input("Username", key="login_user")
                login_pass = st.text_input("Password", type="password", key="login_pass")
                if st.button("Login"):
                    if check_user(login_user, login_pass):
                        st.success("Successfully logged in!")
                        st.rerun()
                    else:
                        st.error("Invalid credentials")

            with register_tab:
                reg_user = st.text_input("Username", key="reg_user")
                reg_email = st.text_input("Email", key="reg_email")
                reg_pass = st.text_input("Password", type="password", key="reg_pass")
                if st.button("Register"):
                    if create_user(reg_user, reg_pass, reg_email):
                        st.success("Registration successful! Please login.")
                    else:
                        st.error("Username already exists")

        else:
            st.write(f"Welcome, {st.session_state.user['username']}!")
            if st.session_state.user['profile_pic']:
                st.image(st.session_state.user['profile_pic'], width=100)
            menu = ["🏠 Home", "🚗 Parking Analysis", "🤖 Parking Assistant", "📊 Data Analysis", "👤 User Profile", "📧 Contact"]
            choice = st.selectbox("Navigate", menu)

            if st.button("Logout"):
                st.session_state.user = None
                st.rerun()

    # Main content based on navigation choice
    if st.session_state.user is None:
        st.title("Welcome to Smart Parking Assistant")
        st.markdown("Please login or register to continue.")
    
    else:
        if choice == "🏠 Home":
            st.title("Smart Parking Assistant Dashboard")
            st.markdown("""
            Welcome to your smart parking management system! 🚀
            
            Our system helps you:
            - Monitor parking spaces in real-time
            - Analyze parking patterns and trends
            - Generate detailed reports and insights
            - Optimize parking operations
            """)
            
            # Quick stats
            if 'parking_stats' in st.session_state:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Spaces Monitored", 
                             f"{st.session_state.parking_stats['empty'] + st.session_state.parking_stats['filled']}")
                with col2:
                    st.metric("Available Spaces", f"{st.session_state.parking_stats['empty']}")
                with col3:
                    st.metric("Occupied Spaces", f"{st.session_state.parking_stats['filled']}")

        elif choice == "🚗 Parking Analysis":
            st.title("🚗 Parking Space Analysis")
            
            uploaded_file = st.file_uploader("Upload parking lot image", type=['png', 'jpg', 'jpeg'])
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                notes = st.text_area("Add notes (optional)")
                
                if st.button("Analyze Parking Spaces"):
                    with st.spinner("Analyzing parking spaces..."):
                        empty_count, filled_count, efficiency, revenue, peak_hour = process_parking_image(image, notes)
                        
                        st.success("Analysis complete!")
                        
                        # Display results
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Empty Spots", empty_count)
                        with col2:
                            st.metric("Filled Spots", filled_count)
                        with col3:
                            st.metric("Efficiency", f"{efficiency:.1%}")
                        
                        st.metric("Estimated Revenue", f"${revenue:.2f}")
                        st.info(f"Current hour: {'Peak' if '7' <= peak_hour <= '19' else 'Off-peak'} ({peak_hour})")

        elif choice == "🤖 Parking Assistant":
            st.title("🤖 Parking Assistant")
            
            uploaded_pdf = st.file_uploader("Upload Parking Documentation", type=["pdf"])
            if uploaded_pdf is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_pdf.getvalue())
                    st.session_state.temp_pdf_path = tmp_file.name
                
                st.success("📄 Document uploaded successfully!")
                
                if st.button("Process Document"):
                    try:
                        embeddings_manager = EmbeddingsManager()
                        result = process_pdf_document(st.session_state.temp_pdf_path, embeddings_manager)
                        st.success(result)
                        
                        # Initialize chatbot
                        if st.session_state.chatbot_manager is None:
                            st.session_state.chatbot_manager = ChatbotManager()
                    
                    except Exception as e:
                        st.error(f"Error processing document: {str(e)}")
            
            # Chat interface
            st.markdown("### 💬 Chat with Parking Assistant")
            if st.session_state.chatbot_manager is not None:
                # Display chat history
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                
                # Chat input
                prompt = st.chat_input("Ask about parking...")
                if prompt:
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    
                    with st.chat_message("assistant"):
                        response = st.session_state.chatbot_manager.get_response(prompt)
                        st.markdown(response)
                    
                    st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                st.info("Please upload and process a document to start chatting.")

        elif choice == "📊 Data Analysis":
            render_data_analysis()

        elif choice == "👤 User Profile":
            render_user_profile()

        elif choice == "📧 Contact":
            st.title("📬 Contact Us")
            st.markdown("""
            We'd love to hear from you! Contact our support team:

            - **Email:** urbantrafficoptimizer@gmail.com
            - **Phone:** +250 (728) 254-819
            """)
            
            # Contact form
            with st.form("contact_form"):
                name = st.text_input("Name")
                email = st.text_input("Email")
                message = st.text_area("Message")
                submitted = st.form_submit_button("Send Message")
                
                if submitted:
                    st.success("Thank you for your message! We'll get back to you soon.")

if __name__ == "__main__":
    main()