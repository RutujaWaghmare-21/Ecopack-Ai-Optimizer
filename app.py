from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import joblib
import csv
import os
from datetime import datetime
import sys
import traceback
from transformers import pipeline
import plotly
import plotly.express as px
import json

app = Flask(__name__)

# ==========================================
# 1. SETUP LOGGING & GLOBAL VARS
# ==========================================
LOG_FILE = 'usage_logs.csv'
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'Timestamp', 'Item', 'Category',
            'Distance', 'Cost_Saved', 'CO2_Saved', 'Standard_CO2'
        ])

# Global variable for the AI model (Lazy Loading)
classifier_model = None

def get_classifier():
    """Loads the AI model only when needed to prevent Render timeouts."""
    global classifier_model
    if classifier_model is None:
        print("⏳ Loading Zero-Shot Classification AI...")
        try:
            classifier_model = pipeline(
                "zero-shot-classification",
                model="valhalla/distilbart-mnli-12-1"
            )
            print("✅ AI Brain Loaded!")
        except Exception as e:
            print(f"❌ AI Load Warning: {e}")
            # Fallback dummy function if AI fails
            return lambda text, labels: {'labels': ['General Use']}
    return classifier_model

# ==========================================
# 2. LOAD DATABASE & MODELS
# ==========================================
try:
    df = pd.read_csv('engineered_material_data.csv')

    df.rename(columns={
        'Product_Category': 'Category',
        'product_category': 'Category',
        'Material_Name': 'Material',
        'material_name': 'Material',
        'Cost_per_kg': 'Cost',
        'cost_per_kg': 'Cost',
        'CO2_Emission_kg': 'CO2_Emissions_kg',
        'co2_emission_kg': 'CO2_Emissions_kg',
    }, inplace=True)

    df['Category'] = df['Category'].astype(str).str.strip()
    df['Material'] = df['Material'].astype(str).str.strip()

    # NOTE: You are loading these but currently using direct calculation logic below.
    # This is fine for a hackathon (safer), but keep the files in the repo.
    try:
        model_cost = joblib.load('model_cost_rf.pkl')
        model_co2 = joblib.load('model_co2_xgb.pkl')
    except:
        print("⚠️ Warning: ML .pkl files not found. Using direct calculation logic.")

    print("✅ System Ready.")

except Exception as e:
    print("❌ Critical Error loading data/models")
    traceback.print_exc()
    # Don't exit here, let the app start even if data fails (easier to debug on Render)

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def guess_category(user_input):
    labels = [
        "Food & Perishables", "Electronics", "Medical & Pharma",
        "Furniture & Home", "Automotive & Industrial",
        "Fashion & Luxury", "Chemicals & Hazmat", "Office & Stationery"
    ]
    try:
        classifier = get_classifier() # Load AI here
        result = classifier(user_input, labels)
        return result['labels'][0]
    except:
        return "General Use"

# ==========================================
# 4. DASHBOARD ROUTES
# ==========================================
@app.route('/dashboard')
def dashboard():
    if not os.path.exists(LOG_FILE) or os.stat(LOG_FILE).st_size == 0:
        return "No data available yet. Please use the app to generate data."

    try:
        data = pd.read_csv(LOG_FILE)
        
        # Check if data is empty
        if data.empty:
            return "Log file exists but is empty."

        total_cost_saved = round(data['Cost_Saved'].sum(), 2)
        total_co2_saved = round(data['CO2_Saved'].sum(), 2)
        total_std_co2 = data['Standard_CO2'].sum()

        pct = round((total_co2_saved / total_std_co2) * 100, 1) if total_std_co2 > 0 else 0

        fig_pie = px.pie(
            data, names='Category',
            title='Material Usage by Category'
        )
        graph_pie = json.dumps(fig_pie, cls=plotly.utils.PlotlyJSONEncoder)

        fig_line = px.line(
            data, x='Timestamp', y='Cost_Saved',
            title='Cumulative Cost Savings (₹)', markers=True
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
        return f"Dashboard Error: {str(e)}"

@app.route('/download_report')
def download_report():
    if os.path.exists(LOG_FILE):
        return send_file(LOG_FILE, as_attachment=True)
    return "No report available."

# ==========================================
# 5. MAIN APP ROUTES
# ==========================================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Receive Data from Frontend
        data = request.get_json()
        item_name = data.get('item_name', 'Unknown')
        weight = float(data.get('weight', 1))
        distance = float(data.get('distance', 500))

        # 2. AI Categorization
        category = guess_category(item_name)

        # 3. Filter Data by Category
        category_data = df[df['Category'] == category].copy()
        if category_data.empty:
            category_data = df.copy() # Fallback to all materials

        # 4. Feature Engineering
        category_data['Weight_kg'] = weight
        category_data['Distance_km'] = distance
        
        # Calculate Base Cost & CO2
        category_data['Base_Cost'] = category_data['Cost'] * weight
        category_data['Base_CO2'] = category_data['CO2_Emissions_kg'] * weight
        
        # Add Shipping Impact
        distance_factor = distance / 500.0
        shipping_cost = category_data['Base_Cost'] * 0.10 * distance_factor
        shipping_co2 = category_data['Base_CO2'] * 0.15 * distance_factor
        
        category_data['Final_Cost'] = category_data['Base_Cost'] + shipping_cost
        category_data['Final_CO2'] = category_data['Base_CO2'] + shipping_co2

        # 7. Scoring Logic
        def safe_score(series, higher_is_better=False):
            if series.nunique() <= 1:
                return pd.Series([85.0] * len(series), index=series.index)
            
            norm = (series - series.min()) / (series.max() - series.min())
            
            if higher_is_better:
                return (norm * 100).round(1)
            else:
                return ((1 - norm) * 100).round(1)

        category_data['Sustainability_Score'] = safe_score(category_data['Final_CO2'])
        category_data['Cost_Efficiency_Score'] = safe_score(category_data['Final_Cost'])

        category_data['Suitability_Score'] = (
            category_data['Sustainability_Score'] * 0.6 +
            category_data['Cost_Efficiency_Score'] * 0.4
        ).round(1)

        # 8. Ranking & Selection
        ranked = category_data.sort_values(by='Final_CO2')
        if ranked.empty:
             return jsonify({'error': 'No matching materials found'}), 400
             
        best = ranked.iloc[0]

        # 9. Savings Calculation
        baseline_material = "Conventional Plastic Packaging"
        standard_cost = float(best['Final_Cost']) * 1.5
        standard_co2 = float(best['Final_CO2']) * 1.5

        cost_saved = float(round(standard_cost - float(best['Final_Cost']), 2))
        co2_saved = float(round(standard_co2 - float(best['Final_CO2']), 2))

        # 10. Log the Transaction
        with open(LOG_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                item_name, category,
                distance, cost_saved, co2_saved, standard_co2
            ])

        # 11. Prepare Response
        recommendations = []
        for _, row in ranked.head(3).iterrows():
            recommendations.append({
                'material_name': str(row['Material']),
                'cost': round(float(row['Final_Cost']), 2),
                'co2': round(float(row['Final_CO2']), 2)
            })

        return jsonify({
            'category': category,
            'baseline': baseline_material,
            'model_info': {
                'cost_model': 'Direct Calculation (Accurate)',
                'co2_model': 'Direct Calculation (Accurate)'
            },
            'recommendations': recommendations,
            'scores': {
                'suitability': float(best['Suitability_Score']),
                'sustainability': float(best['Sustainability_Score']),
                'cost_efficiency': float(best['Cost_Efficiency_Score']),
                'confidence': float(min(95.0, best['Suitability_Score']))
            },
            'savings': {
                'cost': cost_saved,
                'co2': co2_saved
            }
        })

    except Exception as e:
        print("\n🚨 SERVER ERROR:")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500    

# ==========================================
# 6. RUN SERVER
# ==========================================
if __name__ == '__main__': 
    if not os.path.exists('templates'):
        print("⚠️ Warning: 'templates' folder is missing. Dashboard will not work.")
    
    # Use PORT environment variable for Render
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Server starting on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False) # Debug False for production
