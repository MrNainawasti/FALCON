import streamlit as st
import numpy as np
import pandas as pd
import time
import threading
import pickle
import os
from flask import Flask, request, send_file, jsonify
from server_engine import load_server_assets, evaluate_global_metrics

# --- 1. ENTERPRISE CSS ---
st.set_page_config(page_title="FALCON Server", page_icon="🦅", layout="wide")
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    .block-container { padding-top: 2rem; padding-bottom: 0rem; }
    [data-testid="stMetricValue"] { color: #ff2b2b; font-family: monospace; font-size: 1.8rem; }
    [data-testid="stMetric"] { background-color: #161a25; border: 1px solid #2e3440; padding: 10px; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)

# --- 2. THREAD-SAFE GLOBAL STATE ---
@st.cache_resource
def get_server_state():
    return {
        'registered_nodes': set(), 
        'pending_updates': {},     
        'current_round': 0,
        'history': [], 
        'rejections': 0, 
        'quality_rollbacks': 0, 
        'threats_detected': 0, 
        'latest_consensus_log': [],
        'last_action': 'success' 
    }
state = get_server_state()

# --- 3. LOAD SIMULATION ASSETS ---
@st.cache_resource
def init_server():
    model, X_test, y_true = load_server_assets()
    with open("global_weights.pkl", "wb") as f:
        pickle.dump(model.get_weights(), f)
    initial_metrics, _ = evaluate_global_metrics(model, X_test, y_true)
    state['history'].append({
        "Round": 0, "Accuracy": initial_metrics['Accuracy'], "F1 Score": initial_metrics['F1 Score'],
        "Precision": initial_metrics['Precision'], "Recall": initial_metrics['Recall']
    })
    return model, X_test, y_true

global_model, X_test_global, y_true_numeric = init_server()

# --- 4. THE LATENT CONSENSUS ALGORITHM ---
def run_latent_consensus():
    try:
        nodes = list(state['pending_updates'].keys())
        local_weights_list = list(state['pending_updates'].values())
        global_weights = global_model.get_weights()
        
        distances = [float(np.sum([np.linalg.norm(cw - gw) for cw, gw in zip(c, global_weights)])) for c in local_weights_list]
        round_logs, accepted_weights = [], []
        
        # Outlier Detection (Strict 2.0 Multiplier)
        if len(nodes) < 3:
            for i, node in enumerate(nodes):
                round_logs.append({"Node": node, "Divergence Score": round(distances[i], 4), "Status": "✅ Accepted (Bypass: <3 Nodes)"})
                accepted_weights.append(local_weights_list[i])
        else:
            threshold = float(np.median(distances) * 2.0)
            for i, node in enumerate(nodes):
                if distances[i] <= threshold:
                    round_logs.append({"Node": node, "Divergence Score": round(distances[i], 4), "Status": "✅ Accepted"})
                    accepted_weights.append(local_weights_list[i])
                else:
                    round_logs.append({"Node": node, "Divergence Score": round(distances[i], 4), "Status": "🚨 REJECTED (Outlier)"})
                    state['rejections'] += 1

        state['latest_consensus_log'] = round_logs

        # Quality Control Gate
        if accepted_weights:
            last_history = state['history'][-1]
            best_f1 = last_history['F1 Score']
            old_weights = global_model.get_weights()
            
            proposed_weights = [np.mean(np.array(layer), axis=0) for layer in zip(*accepted_weights)]
            global_model.set_weights(proposed_weights)
            
            proposed_metrics, proposed_threats = evaluate_global_metrics(global_model, X_test_global, y_true_numeric)
            state['current_round'] += 1
            
            if proposed_metrics['F1 Score'] >= best_f1:
                state['last_action'] = 'success'
                with open("global_weights.pkl", "wb") as f: pickle.dump(proposed_weights, f)
                state['history'].append({
                    "Round": state['current_round'], 
                    "Accuracy": proposed_metrics['Accuracy'], "F1 Score": proposed_metrics['F1 Score'],
                    "Precision": proposed_metrics['Precision'], "Recall": proposed_metrics['Recall']
                })
                state['threats_detected'] += proposed_threats
            else:
                state['last_action'] = 'rollback'
                global_model.set_weights(old_weights)
                state['quality_rollbacks'] += 1
                state['history'].append({
                    "Round": state['current_round'], 
                    "Accuracy": last_history['Accuracy'], "F1 Score": last_history['F1 Score'],
                    "Precision": last_history['Precision'], "Recall": last_history['Recall']
                })
            
        state['pending_updates'].clear()
    except Exception as e: print(f"Consensus Error: {e}")

# --- 5. FLASK BACKGROUND SERVER ---
app = Flask(__name__)

@app.route('/register', methods=['POST'])
def register_node():
    node_id = request.form.get('node_id')
    if node_id: 
        state['registered_nodes'].add(node_id)
        return "Registered", 200
    return "Missing node_id", 400

@app.route('/update_weights', methods=['POST'])
def update_weights():
    node_id = request.form.get('node_id')
    if node_id in state['registered_nodes']:
        state['pending_updates'][node_id] = pickle.loads(request.files['weights'].read())
        if len(state['pending_updates']) == len(state['registered_nodes']): 
            run_latent_consensus()
        return "Weights queued", 200
    return "Node not registered", 403

@app.route('/status', methods=['GET'])
def get_status():
    return jsonify({
        "round": state['current_round'],
        "last_action": state['last_action']
    })

@app.route('/get_weights', methods=['GET'])
def get_weights():
    return send_file("global_weights.pkl", as_attachment=True)

if 'flask_started' not in st.session_state:
    threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False), daemon=True).start()
    st.session_state['flask_started'] = True

# --- 6. MAIN UI DASHBOARD ---
with st.sidebar:
    st.title("🦅 FEDERATED AUTOENCODER-BASED LATENT CONSENSUS OUTLIER-RESILIENT NETWORK FOR CYBER THREAT DETECTION")
    st.success("🟢 Global AI Core Online")
    st.write(f"**Federated Round:** {state['current_round']}")
    st.divider()
    
    st.markdown("### 📡 Live Mesh Network")
    st.caption("Monitoring connected sentinels...")
    if state['registered_nodes']:
        for node in state['registered_nodes']:
            status_icon = "🟢" if node in state['pending_updates'] else "🟡"
            st.write(f"{status_icon} **{node}**")
    else: 
        st.info("No sentinels connected.")

st.header("Global Defense Dashboard")

latest = state['history'][-1]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Global F1 Score", f"{latest['F1 Score'] * 100:.2f}%")
c2.metric("Global Accuracy", f"{latest['Accuracy'] * 100:.2f}%")
c3.metric("Consensus Rejections", state['rejections'], help="Nodes blocked due to outlier divergence.")
c4.metric("Quality Rollbacks", state['quality_rollbacks'], help="Updates blocked because they degraded the Global F1 Score.")

st.divider()

st.subheader("Latent Consensus Diagnostics")
if state['latest_consensus_log']: 
    st.dataframe(pd.DataFrame(state['latest_consensus_log']), use_container_width=True)
else: 
    st.info("Awaiting Round 1 updates...")

st.subheader("Federated Performance Curve")
if len(state['history']) > 1:
    chart_data = [{"Round": d["Round"], "Accuracy": d["Accuracy"]*100, "F1": d["F1 Score"]*100, "Recall": d["Recall"]*100} for d in state['history']]
    st.line_chart(pd.DataFrame(chart_data).set_index("Round"), color=["#00ff88", "#ff2b2b", "#3388ff"])
else: 
    st.info("Awaiting synchronized field data...")

time.sleep(2)
st.rerun()