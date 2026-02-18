import os
import csv
import sys
import traceback
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import joblib
import plotly
import plotly.express as px
import json

app = Flask(__name__)

# ==========================================
# 1. SETUP LOGGING
# ==========================================
LOG_FILE = 'usage_logs.csv'
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'Item', 'Category', 'Distance', 'Cost_Saved', 'CO2_Saved', 'Standard_CO2'])

# ==========================================
# 2. LOAD DATABASE
# ==========================================
try:
    if os.path.exists('engineered_material_data.csv'):
        df = pd.read_csv('engineered_material_data.csv')
        # Clean column names
        df.rename(columns={
            'Product_Category': 'Category', 'product_category': 'Category',
            'Material_Name': 'Material', 'material_name': 'Material',
            'Cost_per_kg': 'Cost', 'cost_per_kg': 'Cost',
            'CO2_Emission_kg': 'CO2_Emissions_kg', 'co2_emission_kg': 'CO2_Emissions_kg',
        }, inplace=True)
        df['Category'] = df['Category'].astype(str).str.strip()
        df['Material'] = df['Material'].astype(str).str.strip()
        print("✅ Database Loaded.")
    else:
        # Fallback dummy data prevents crash
        df = pd.DataFrame({
            'Category': ['Electronics', 'Electronics', 'Fashion'],
            'Material': ['Cardboard Box', 'Bubble Wrap', 'Poly Bag'],
            'Cost': [10.0, 15.0, 2.0],
            'CO2_Emissions_kg': [0.5, 1.2, 0.2]
        })
        print("⚠️ Using dummy data (CSV missing).")
except Exception as e:
    print(f"❌ Data Load Error: {e}")
    df = pd.DataFrame()

# ==========================================
# 3. ROUTES
# ==========================================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    if not os.path.exists(LOG_FILE) or os.stat(LOG_FILE).st_size == 0:
        return "No data available yet."

    try:
        data = pd.read_csv(LOG_FILE)
        
        # FIX TIME DATA CRASH
        data['Timestamp'] = pd.to_datetime(data['Timestamp'], format='mixed', errors='coerce')
        data = data.dropna(subset=['Timestamp'])
        data = data.sort_values(by='Timestamp') 

        total_cost_saved = round(data['Cost_Saved'].sum(), 2)
        total_co2_saved = round(data['CO2_Saved'].sum(), 2)
        total_std_co2 = data['Standard_CO2'].sum()

        # Calculate Percentage
        pct = round((total_co2_saved / total_std_co2) * 100, 1) if total_std_co2 > 0 else 0

        # Graph 1: Pie Chart
        fig_pie = px.pie(
            data, names='Category',
            title='Material Usage by Category',
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        graph_pie = json.dumps(fig_pie, cls=plotly.utils.PlotlyJSONEncoder)

        # Graph 2: Line Chart
        fig_line = px.line(
            data, x='Timestamp', y='Cost_Saved',
            title='Cumulative Cost Savings (₹)', markers=True,
            color_discrete_sequence=['#00b894']
        )
        graph_line = json.dumps(fig_line, cls=plotly.utils.PlotlyJSONEncoder)

        return render_template(
            'dashboard.html',
            cost=total_cost_saved,
            co2=total_co2_saved,
            pct=pct,
            graph_pie=graph_pie,
            graph_line=graph_line
        )
    except Exception as e:
        traceback.print_exc()
        return f"Dashboard Error: {str(e)}"

@app.route('/download_report')
def download_report():
    if os.path.exists(LOG_FILE):
        return send_file(LOG_FILE, as_attachment=True)
    return "No report available."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Receive Data
        data = request.get_json()
        item_name = data.get('item_name', 'Unknown')
        category = data.get('category', 'General Use')
        weight = float(data.get('weight', 1))
        distance = float(data.get('distance', 500))

        # 2. Filter Data
        category_data = df[df['Category'] == category].copy()
        if category_data.empty:
            category_data = df.copy()

        # 3. Calculations
        category_data['Weight_kg'] = weight
        category_data['Base_Cost'] = category_data['Cost'] * weight
        category_data['Base_CO2'] = category_data['CO2_Emissions_kg'] * weight
        
        # Shipping Impact
        distance_factor = distance / 500.0
        category_data['Final_Cost'] = category_data['Base_Cost'] + (category_data['Base_Cost'] * 0.10 * distance_factor)
        category_data['Final_CO2'] = category_data['Base_CO2'] + (category_data['Base_CO2'] * 0.15 * distance_factor)

        # 4. Scoring
        def safe_score(series, higher_is_better=False):
            if series.nunique() <= 1: return pd.Series([85.0]*len(series), index=series.index)
            norm = (series - series.min()) / (series.max() - series.min())
            return (norm * 100).round(1) if higher_is_better else ((1 - norm) * 100).round(1)

        category_data['Sustainability_Score'] = safe_score(category_data['Final_CO2'])
        category_data['Cost_Efficiency_Score'] = safe_score(category_data['Final_Cost'])
        category_data['Suitability_Score'] = (category_data['Sustainability_Score']*0.6 + category_data['Cost_Efficiency_Score']*0.4).round(1)

        # 5. Best Match
        best = category_data.sort_values(by='Final_CO2').iloc[0]

        # 6. Savings Calc
        baseline_cost = float(best['Final_Cost']) * 1.5
        baseline_co2 = float(best['Final_CO2']) * 1.5
        cost_saved = round(baseline_cost - float(best['Final_Cost']), 2)
        co2_saved = round(baseline_co2 - float(best['Final_CO2']), 2)

        # 7. Log
        with open(LOG_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), item_name, category, distance, cost_saved, co2_saved, baseline_co2])

        # 8. Response
        recommendations = []
        for _, row in category_data.sort_values(by='Final_CO2').head(3).iterrows():
            recommendations.append({
                'material_name': str(row['Material']),
                'cost': round(float(row['Final_Cost']), 2),
                'co2': round(float(row['Final_CO2']), 2)
            })

        return jsonify({
            'category': category,
            'baseline': "Conventional Packaging",
            'recommendations': recommendations,
            'scores': {
                'suitability': float(best['Suitability_Score']),
                'sustainability': float(best['Sustainability_Score']),
                'cost_efficiency': float(best['Cost_Efficiency_Score']),
                'confidence': 100.0
            },
            'savings': {'cost': cost_saved, 'co2': co2_saved}
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ==========================================
# 4. RUN SERVER
# ==========================================
if __name__ == '__main__':
    # Get the port from Render, or use 5000 for local testing
    port = int(os.environ.get("PORT", 5000))
    
    # Check if we are on Render (Render sets 'RENDER' env var)
    if os.environ.get("RENDER"):
        # PRODUCTION: Turn off debug, listen on 0.0.0.0
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        # LOCAL: Keep debug on for easier testing
        app.run(debug=True)
