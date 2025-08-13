import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect('co2_saver.db')
    c = conn.cursor()
    
    # Create trips table if it doesn't exist
    c.execute('''CREATE TABLE IF NOT EXISTS trips
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_name TEXT,
                  date TEXT,
                  mode TEXT,
                  distance REAL,
                  occupancy INTEGER,
                  emission_factor REAL,
                  co2_emitted REAL,
                  baseline_co2 REAL,
                  co2_saved REAL,
                  percent_improvement REAL,
                  suggested_mode TEXT,
                  trees_saved REAL)''')
    
    # Create emission factors table
    c.execute('''CREATE TABLE IF NOT EXISTS emission_factors
                 (mode TEXT PRIMARY KEY,
                  factor REAL,
                  unit TEXT)''')
    
    # Insert default emission factors if table is empty
    c.execute("SELECT COUNT(*) FROM emission_factors")
    if c.fetchone()[0] == 0:
        default_factors = [
            ('Petrol Car', 0.192, 'kg/km'),
            ('Diesel Car', 0.171, 'kg/km'),
            ('CNG Auto', 0.075, 'kg/km'),
            ('Bus', 0.082, 'kg/pkm'),
            ('Metro', 0.040, 'kg/pkm'),
            ('EV Car', 0.070, 'kg/km'),
            ('Cycle', 0.0, 'kg/km'),
            ('Walk', 0.0, 'kg/km')
        ]
        c.executemany("INSERT INTO emission_factors VALUES (?, ?, ?)", default_factors)
    
    conn.commit()
    conn.close()

# Initialize the database
init_db()

# Emission factors (will be loaded from DB)
def get_emission_factors():
    conn = sqlite3.connect('co2_saver.db')
    factors = pd.read_sql("SELECT * FROM emission_factors", conn)
    conn.close()
    return factors.set_index('mode')['factor'].to_dict()

# Calculate CO2 emissions
def calculate_co2(mode, distance, occupancy=1):
    factors = get_emission_factors()
    
    if mode not in factors:
        return 0, 0, 0
    
    # For public transport, we divide by occupancy (passenger-km)
    if mode in ['Bus', 'Metro']:
        co2_emitted = factors[mode] * distance / max(1, occupancy)
    else:
        co2_emitted = factors[mode] * distance
    
    # Baseline is petrol car solo
    baseline_co2 = factors['Petrol Car'] * distance
    
    # CO2 saved compared to baseline
    co2_saved = baseline_co2 - co2_emitted
    percent_improvement = (co2_saved / baseline_co2) * 100 if baseline_co2 > 0 else 0
    
    # Trees equivalent (assuming 1 tree absorbs ~21.77 kg CO2 per year, ~0.06 kg/day)
    trees_saved = co2_saved / 0.06
    
    return co2_emitted, co2_saved, percent_improvement, trees_saved

# Find better alternative mode
def suggest_better_mode(distance, current_mode):
    factors = get_emission_factors()
    
    # Remove current mode from suggestions
    available_modes = [m for m in factors.keys() if m != current_mode and factors[m] > 0]
    
    if not available_modes:
        return None, 0
    
    # Find mode with lowest emissions
    best_mode = min(available_modes, key=lambda x: factors[x])
    
    # Calculate potential savings
    current_co2 = factors[current_mode] * distance
    best_co2 = factors[best_mode] * distance
    savings = current_co2 - best_co2
    
    return best_mode, savings

# Save trip to database
def save_trip(user_name, mode, distance, occupancy):
    co2_emitted, co2_saved, percent_improvement, trees_saved = calculate_co2(mode, distance, occupancy)
    best_mode, _ = suggest_better_mode(distance, mode)
    
    conn = sqlite3.connect('co2_saver.db')
    c = conn.cursor()
    
    # Get emission factor for this mode
    factors = get_emission_factors()
    emission_factor = factors.get(mode, 0)
    
    # Insert trip
    c.execute('''INSERT INTO trips 
                 (user_name, date, mode, distance, occupancy, emission_factor, 
                  co2_emitted, baseline_co2, co2_saved, percent_improvement, 
                  suggested_mode, trees_saved)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (user_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), mode, distance, occupancy, emission_factor,
               co2_emitted, emission_factor * distance, co2_saved, percent_improvement,
               best_mode, trees_saved))
    
    conn.commit()
    conn.close()
    
    return co2_emitted, co2_saved, percent_improvement, best_mode, trees_saved

# Get all trips
def get_all_trips():
    conn = sqlite3.connect('co2_saver.db')
    trips = pd.read_sql("SELECT * FROM trips ORDER BY date DESC", conn)
    conn.close()
    return trips

# Get dashboard summary
def get_dashboard_summary():
    conn = sqlite3.connect('co2_saver.db')
    
    # Total CO2 emitted
    total_emitted = pd.read_sql("SELECT SUM(co2_emitted) FROM trips", conn).iloc[0,0] or 0
    
    # Total CO2 saved
    total_saved = pd.read_sql("SELECT SUM(co2_saved) FROM trips", conn).iloc[0,0] or 0
    
    # Total trips
    total_trips = pd.read_sql("SELECT COUNT(*) FROM trips", conn).iloc[0,0] or 0
    
    # Mode distribution
    mode_dist = pd.read_sql("SELECT mode, COUNT(*) as count FROM trips GROUP BY mode", conn)
    
    conn.close()
    
    return {
        'total_emitted': total_emitted,
        'total_saved': total_saved,
        'total_trips': total_trips,
        'mode_dist': mode_dist
    }

# Generate receipt image
def generate_receipt_image(user_name, mode, distance, co2_emitted, co2_saved, percent_improvement, best_mode, trees_saved):
    # Create a blank image
    img = Image.new('RGB', (600, 800), color=(240, 240, 240))
    d = ImageDraw.Draw(img)
    
    # Load a font (using default font for simplicity)
    try:
        font_large = ImageFont.truetype("arial.ttf", 28)
        font_medium = ImageFont.truetype("arial.ttf", 20)
        font_small = ImageFont.truetype("arial.ttf", 16)
    except:
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Add title
    d.text((50, 50), "🌱 Green Receipt", font=font_large, fill=(0, 100, 0))
    
    # Add user info
    d.text((50, 100), f"User: {user_name}", font=font_medium, fill=(0, 0, 0))
    d.text((50, 130), f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", font=font_medium, fill=(0, 0, 0))
    
    # Add trip details
    d.text((50, 180), "Trip Details:", font=font_medium, fill=(0, 0, 0))
    d.text((70, 210), f"Mode: {mode}", font=font_small, fill=(0, 0, 0))
    d.text((70, 235), f"Distance: {distance} km", font=font_small, fill=(0, 0, 0))
    
    # Add CO2 info
    d.text((50, 290), "CO₂ Impact:", font=font_medium, fill=(0, 0, 0))
    d.text((70, 320), f"Emitted: {co2_emitted:.2f} kg", font=font_small, fill=(150, 0, 0))
    d.text((70, 345), f"Saved vs baseline: {co2_saved:.2f} kg ({percent_improvement:.1f}%)", font=font_small, fill=(0, 100, 0))
    
    # Add suggestion
    if best_mode:
        d.text((50, 400), "Better Alternative:", font=font_medium, fill=(0, 0, 0))
        d.text((70, 430), f"Next time try: {best_mode}", font=font_small, fill=(0, 0, 150))
    
    # Add trees equivalent
    d.text((50, 480), "Environmental Impact:", font=font_medium, fill=(0, 0, 0))
    d.text((70, 510), f"Equivalent to saving {trees_saved:.1f} trees for a day", font=font_small, fill=(0, 100, 0))
    
    # Add footer
    d.text((150, 750), "Thank you for reducing your carbon footprint!", font=font_small, fill=(100, 100, 100))
    
    # Save image to bytes
    img_bytes = BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    return img_bytes

# Prediction tool
def predict_savings(current_mode, new_mode, trips_per_week, distance_per_trip):
    factors = get_emission_factors()
    
    if current_mode not in factors or new_mode not in factors:
        return 0
    
    # Weekly CO2 for current mode
    current_co2 = factors[current_mode] * distance_per_trip * trips_per_week
    
    # Weekly CO2 for new mode
    new_co2 = factors[new_mode] * distance_per_trip * trips_per_week
    
    # Weekly savings
    weekly_savings = current_co2 - new_co2
    
    # Annual savings
    annual_savings = weekly_savings * 52
    
    # Trees equivalent
    trees_saved = annual_savings / (0.06 * 365)  # 0.06 kg/day per tree
    
    return weekly_savings, annual_savings, trees_saved

# Admin page for editing emission factors
def admin_page():
    st.title("Admin - Emission Factors")
    
    conn = sqlite3.connect('co2_saver.db')
    factors_df = pd.read_sql("SELECT * FROM emission_factors", conn)
    conn.close()
    
    edited_df = st.data_editor(factors_df, num_rows="dynamic")
    
    if st.button("Save Changes"):
        conn = sqlite3.connect('co2_saver.db')
        edited_df.to_sql('emission_factors', conn, if_exists='replace', index=False)
        conn.close()
        st.success("Emission factors updated successfully!")

# Main app
def main():
    st.set_page_config(page_title="CO₂ Saver - Green Receipt", layout="wide")
    
    # Sidebar navigation
    st.sidebar.title("CO₂ Saver")
    app_mode = st.sidebar.radio("Navigation", 
                               ["Log Trip", "Dashboard", "History", "Prediction Tool", "Admin"])
    
    if app_mode == "Log Trip":
        st.title("🌱 Log Your Trip")
        
        # Form for input
        with st.form("trip_form"):
         user_name = st.text_input("Your Name", "Anonymous")
         mode = st.selectbox("Mode of Transport", ["Petrol Car", "Diesel Car", "CNG Auto","Bus", "Metro", "EV Car", "Cycle", "Walk"])
         distance = st.number_input("Distance (km)", min_value=0.1, value=5.0)
         occupancy = st.number_input("Occupancy", min_value=1, value=1)
         submitted = st.form_submit_button("Generate Green Receipt")

# Processing and display outside the form
        if submitted:
         co2_emitted, co2_saved, percent_improvement, best_mode, trees_saved = save_trip(
         user_name, mode, distance, occupancy)
    
         # Display receipt
         st.success("Here's your Green Receipt!")
    
         col1, col2 = st.columns([1, 2])
    
         with col1:
            # Display receipt card (unchanged)
    
           with col2:
             # Generate and display receipt image
             img_bytes = generate_receipt_image(user_name, mode, distance, co2_emitted, 
                                         co2_saved, percent_improvement, best_mode, trees_saved)
        
         st.image(img_bytes, caption="Your Green Receipt", use_container_width =True)
        
         # Download button now outside the form
         st.download_button(
            label="Download Receipt",
            data=img_bytes,
            file_name="green_receipt.png",
            mime="image/png"
         )
    
    elif app_mode == "Dashboard":
        st.title("🌍 Live Dashboard")
        
        # Get dashboard data
        summary = get_dashboard_summary()
        
        # Display KPIs
        col1, col2, col3 = st.columns(3)
        col1.metric("Total CO₂ Emitted", f"{summary['total_emitted']:.1f} kg")
        col2.metric("Total CO₂ Saved", f"{summary['total_saved']:.1f} kg")
        col3.metric("Total Trips Recorded", summary['total_trips'])
        
        # Mode distribution chart
        if not summary['mode_dist'].empty:
            st.subheader("Transport Mode Distribution")
            fig = px.pie(summary['mode_dist'], names='mode', values='count', 
                         title="Breakdown of Transport Modes Used")
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent trips
        st.subheader("Recent Trips")
        trips = get_all_trips()
        if not trips.empty:
            st.dataframe(trips.head(10)[['user_name', 'date', 'mode', 'distance', 'co2_emitted', 'co2_saved']])
        else:
            st.info("No trips recorded yet. Log a trip to see data here.")
    
    elif app_mode == "History":
        st.title("Trip History")
        
        trips = get_all_trips()
        
        if not trips.empty:
            # Filter options
            col1, col2 = st.columns(2)
            with col1:
                user_filter = st.selectbox("Filter by user", 
                                         ["All"] + sorted(trips['user_name'].unique().tolist()))
            with col2:
                date_range = st.date_input("Filter by date range", 
                                         [datetime.now().date() - pd.Timedelta(days=7), datetime.now().date()])
            
            # Apply filters
            filtered_trips = trips.copy()
            if user_filter != "All":
                filtered_trips = filtered_trips[filtered_trips['user_name'] == user_filter]
            
            if len(date_range) == 2:
                filtered_trips = filtered_trips[
                    (pd.to_datetime(filtered_trips['date']).dt.date >= date_range[0]) &
                    (pd.to_datetime(filtered_trips['date']).dt.date <= date_range[1])
                ]
            st.dataframe(filtered_trips)
            
            # Summary statistics
            st.subheader("Summary Statistics")
            if not filtered_trips.empty:
                total_co2 = filtered_trips['co2_emitted'].sum()
                total_saved = filtered_trips['co2_saved'].sum()
                avg_saving = filtered_trips['percent_improvement'].mean()
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total CO₂ Emitted", f"{total_co2:.1f} kg")
                col2.metric("Total CO₂ Saved", f"{total_saved:.1f} kg")
                col3.metric("Average Improvement", f"{avg_saving:.1f}%")
        else:
            st.info("No trips match your filters.")
    
    elif app_mode == "Prediction Tool":
        st.title("🌿 CO₂ Savings Predictor")
        
        st.markdown("""
        See how much CO₂ you could save by changing your transportation habits.
        """)
        
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                current_mode = st.selectbox("Your Current Mode", 
                                          ["Petrol Car", "Diesel Car", "CNG Auto", "Bus", "Metro", "EV Car", "Cycle", "Walk"])
            with col2:
                new_mode = st.selectbox("Alternative Mode", 
                                       ["Petrol Car", "Diesel Car", "CNG Auto", "Bus", "Metro", "EV Car", "Cycle", "Walk"])
            
            trips_per_week = st.slider("Number of trips per week", 1, 20, 5)
            distance_per_trip = st.number_input("Average distance per trip (km)", min_value=0.1, max_value=100.0, value=5.0)
            
            submitted = st.form_submit_button("Calculate Savings")
            
            if submitted:
                weekly_savings, annual_savings, trees_saved = predict_savings(
                    current_mode, new_mode, trips_per_week, distance_per_trip)
                
                st.success(f"""
                **Potential Savings:**
                
                - **Weekly CO₂ Savings:** {weekly_savings:.2f} kg
                - **Annual CO₂ Savings:** {annual_savings:.2f} kg
                - **Equivalent to saving {trees_saved:.1f} trees for a year**
                """)
                
                # Visualization
                fig, ax = plt.subplots(figsize=(8, 4))
                modes = [current_mode, new_mode]
                factors = get_emission_factors()
                co2_values = [
                    factors[current_mode] * distance_per_trip * trips_per_week,
                    factors[new_mode] * distance_per_trip * trips_per_week
                ]
                
                bars = ax.bar(modes, co2_values, color=['#d9534f', '#5cb85c'])
                ax.set_ylabel('Weekly CO₂ Emissions (kg)')
                ax.set_title('Comparison of Weekly CO₂ Emissions')
                
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.1f} kg',
                            ha='center', va='bottom')
                
                st.pyplot(fig)
    
    elif app_mode == "Admin":
        admin_page()

if __name__ == "__main__":
    main()