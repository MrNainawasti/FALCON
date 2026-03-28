import streamlit as st
import numpy as np
import pandas as pd
import requests
import pickle
import os
import time
import random
import tempfile
import json
from client_engine import load_and_calibrate_client, evaluate_full_metrics
from packet_pipeline import process_pcap, process_csv

# --- 1. ENTERPRISE CSS ---
st.set_page_config(page_title="FALCON Sentinel", page_icon="🛡️", layout="wide")
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    .block-container { padding-top: 2rem; padding-bottom: 0rem; }
    [data-testid="stMetricValue"] { color: #00ff88; font-family: monospace; font-size: 1.8rem; }
    [data-testid="stMetric"] { background-color: #161a25; border: 1px solid #2e3440; padding: 10px; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)

SERVER_IP = "127.0.0.1" 
SERVER_URL = f"http://{SERVER_IP}:5050"

# --- 2. LOAD SIMULATION MATH ---
@st.cache_resource
def init_system():
    return load_and_calibrate_client()

base_path = os.path.dirname(os.path.abspath(__file__))
model, X_train, X_test, y_true_numeric, normal_data, attack_data, initial_threshold = init_system()

def calculate_confidence(mse, thresh):
    if mse <= thresh:
        conf = 99.0 - (49.0 * (mse / thresh))
        return max(50.1, conf), "Benign"
    else:
        ratio = min((mse - thresh) / (3 * thresh), 1.0)
        conf = 50.0 + (49.9 * ratio)
        return min(99.9, conf), "Threat"

# --- 3. STATE MANAGEMENT ---
if 'scan_history' not in st.session_state: st.session_state['scan_history'] = []
if 'total_packets' not in st.session_state: st.session_state['total_packets'] = 0
if 'threats_blocked' not in st.session_state: st.session_state['threats_blocked'] = 0
if 'local_buffer' not in st.session_state: st.session_state['local_buffer'] = 0
if 'is_scanning' not in st.session_state: st.session_state['is_scanning'] = False
if 'current_round' not in st.session_state: st.session_state['current_round'] = 0
if 'target_buffer' not in st.session_state: st.session_state['target_buffer'] = 500
if 'manual_sample' not in st.session_state: st.session_state['manual_sample'] = ", ".join([str(float(x)) for x in normal_data[0]])
if 'threshold' not in st.session_state: st.session_state['threshold'] = initial_threshold

if 'local_metrics' not in st.session_state:
    with st.spinner("Calibrating Simulation Parameters..."):
        st.session_state['local_metrics'], st.session_state['threshold'] = evaluate_full_metrics(model, X_test, y_true_numeric)

# --- 4. SIDEBAR: FEDERATION CONTROL ---
with st.sidebar:
    st.title("🛡️ FALCON Sentinel")
    node_name = st.text_input("Node ID", value="Node-Alpha")
    
    if 'registered' not in st.session_state or st.session_state.get('last_node') != node_name:
        try:
            requests.post(f"{SERVER_URL}/register", data={'node_id': node_name})
            st.session_state['registered'] = True
            st.session_state['last_node'] = node_name
        except:
            st.caption("⚠️ Cannot connect to Global Core")
            
    st.divider()
    
    st.markdown("### Local AI Training Buffer")
    auto_sync = st.toggle("⚙️ Auto-Push Updates", value=True)
    
    buffer_cap = st.session_state['target_buffer']
    current_buffer = st.session_state['local_buffer']
    progress_val = min(current_buffer / buffer_cap, 1.0)
    
    if current_buffer >= buffer_cap:
        trigger_sync = False
        if auto_sync:
            st.progress(progress_val, text=f"🔄 Auto-Syncing: {current_buffer} Packets...")
            trigger_sync = True
        else:
            st.progress(progress_val, text=f"✅ Ready: {current_buffer} Packets")
            if st.button("🚀 Push Local Update", type="primary", use_container_width=True):
                trigger_sync = True

        if trigger_sync:
            with st.status("Executing Synchronous Federation...", expanded=True) as status:
                try:
                    status.write("Training on local packet capture...")
                    train_size = min(current_buffer, len(X_train))
                    idx = np.random.randint(0, len(X_train) - train_size + 1)
                    model.fit(X_train[idx:idx+train_size], X_train[idx:idx+train_size], epochs=1, verbose=0)
                    
                    status.write("Pushing weights to Central Server...")
                    weights = model.get_weights()
                    temp = f"weights_{node_name}.pkl"
                    with open(temp, 'wb') as f: pickle.dump(weights, f)
                    
                    with open(temp, 'rb') as f: 
                        r = requests.post(f"{SERVER_URL}/update_weights", files={'weights': f}, data={'node_id': node_name})
                    os.remove(temp)
                    
                    if r.status_code == 200:
                        status.write("Waiting for other network nodes to sync...")
                        while True:
                            try:
                                resp = requests.get(f"{SERVER_URL}/status").json()
                                if resp['round'] > st.session_state['current_round']:
                                    status.write("Consensus reached. Pulling new Global AI...")
                                    r_pull = requests.get(f"{SERVER_URL}/get_weights")
                                    model.set_weights(pickle.loads(r_pull.content))
                                    
                                    st.session_state['local_metrics'], st.session_state['threshold'] = evaluate_full_metrics(model, X_test, y_true_numeric)
                                    st.session_state['current_round'] = resp['round']
                                    
                                    if resp.get('last_action') == 'rollback':
                                        st.session_state['target_buffer'] += 500
                                        st.warning(f"Server rejected update! Expanding batch size to {st.session_state['target_buffer']}...", icon="⚠️")
                                    else:
                                        st.session_state['local_buffer'] = 0 
                                        st.session_state['target_buffer'] = 500
                                        st.toast("🧬 Global Model Auto-Updated!", icon="✅")
                                        
                                    break
                            except: pass
                            time.sleep(1.5) 
                        st.rerun()
                    else: st.error("Server rejected the push request.")
                except Exception as e: st.error(f"Error: {e}")
    else:
        st.progress(progress_val, text=f"Gathering: {current_buffer} / {buffer_cap} Min Packets")
        st.button("🔒 Push Locked (Awaiting Data)", disabled=True, use_container_width=True)

    st.divider()
    
    st.markdown("### 🔬 Node Parameters")
    m1, m2 = st.columns(2)
    m1.metric("Accuracy", f"{st.session_state['local_metrics']['Accuracy'] * 100:.2f}%")
    m2.metric("F1 Score", f"{st.session_state['local_metrics']['F1 Score'] * 100:.2f}%")

# --- 5. MAIN UI ---
st.header("FEDERATED AUTOENCODER-BASED LATENT CONSENSUS OUTLIER-RESILIENT NETWORK FOR CYBER THREAT DETECTION")
st.write(f"**Security Tripwire:** `{st.session_state['threshold']:.6f}`")

tab1, tab2 = st.tabs(["📡 Live Network Stream", "🔬 Analyst Sandbox"])

with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Packets Analyzed", st.session_state['total_packets'])
    c2.metric("Threats Blocked", st.session_state['threats_blocked'])
    
    scan_button_text = "🛑 Stop Sniffer" if st.session_state['is_scanning'] else "▶️ Start Live Sniffer"
    if c3.button(scan_button_text, use_container_width=True):
        st.session_state['is_scanning'] = not st.session_state['is_scanning']
        st.rerun()
        
    st.subheader("Network Reconstruction Error Matrix")
    if st.session_state['scan_history']:
        df = pd.DataFrame(st.session_state['scan_history']).set_index("Packet")
        st.line_chart(df[['Error', 'Threshold']], color=["#00ff88", "#ff2b2b"]) 
    else:
        st.info("Awaiting traffic stream...")

    if st.session_state['is_scanning']:
        time.sleep(0.5) 
        packet = normal_data[np.random.randint(0, len(normal_data))] if random.random() > 0.10 else attack_data[np.random.randint(0, len(attack_data))]
        packet_2d = packet.reshape(1, -1)
        recon = model.predict(packet_2d, verbose=0)
        mse = float(np.mean(np.power(packet_2d - recon, 2)))
        
        conf, status = calculate_confidence(mse, st.session_state['threshold'])
        
        st.session_state['total_packets'] += 1
        if status == "Benign": st.session_state['local_buffer'] += 1
        else: st.session_state['threats_blocked'] += 1
            
        st.session_state['scan_history'].append({"Packet": st.session_state['total_packets'], "Error": mse, "Threshold": st.session_state['threshold'], "Status": status})
        if len(st.session_state['scan_history']) > 50: st.session_state['scan_history'].pop(0)
        st.rerun()

# ==========================================
# TAB 2: MANUAL TESTING SANDBOX
# ==========================================
with tab2:
    st.info("Test specific network signatures. The Autoencoder will assess the mathematical anomaly score and calculate a confidence percentage.")
    
    input_mode = st.radio("Select Testing Data Source:", ["🎲 Random Pull from Dataset", "📁 Upload Raw Capture (.pcap / .csv)"], horizontal=True)
    
    if input_mode == "🎲 Random Pull from Dataset":
        col_a, col_b = st.columns([1, 2])
        with col_a:
            target_type = st.radio("Select Target Signature:", ["Benign Traffic", "DDoS / Malicious Traffic"])
            if st.button("🎲 Pull New Signature"):
                if target_type == "Benign Traffic":
                    pkt = normal_data[np.random.randint(0, len(normal_data))]
                else:
                    pkt = attack_data[np.random.randint(0, len(attack_data))]
                st.session_state['manual_sample'] = ", ".join([str(float(x)) for x in pkt])
                st.rerun()
        with col_b:
            raw_input = st.text_area("Extracted Features (Scaled):", value=st.session_state['manual_sample'], height=120)

    else:
        uploaded_file = st.file_uploader("Upload Traffic Dump", type=["pcap", "csv"])
        if uploaded_file is not None:
            with st.spinner("Processing through FALCON Packet Pipeline..."):
                ext = ".csv" if uploaded_file.name.endswith(".csv") else ".pcap"
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_in:
                    tmp_in.write(uploaded_file.getbuffer())
                    in_path = tmp_in.name
                    
                with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp_out:
                    out_path = tmp_out.name
                    
                try:
                    config_path = os.path.join(base_path, "config.json")
                    with open(config_path, "r") as f:
                        config = json.load(f)
                    
                    if ext == ".csv": process_csv(in_path, config, out_path)
                    else: process_pcap(in_path, config, out_path)
                    
                    with open(out_path, "r") as f:
                        flows = json.load(f)
                        
                    if not flows:
                        st.warning("No complete TCP/UDP flows could be extracted from this capture.")
                    else:
                        st.success(f"✅ Pipeline Extracted {len(flows)} unique network flows!")
                        flow_options = {f["flow_id"]: f["features"] for f in flows}
                        selected_flow_id = st.selectbox("Select Flow to Analyze:", list(flow_options.keys()))
                        
                        # Dynamically update the text area with the pipeline's output
                        current_features = flow_options[selected_flow_id]
                        st.session_state['manual_sample'] = ", ".join([str(float(x)) for x in current_features])
                        
                finally:
                    os.remove(in_path)
                    os.remove(out_path)
                    
        raw_input = st.text_area("Extracted Features (Scaled):", value=st.session_state['manual_sample'], height=120)

    # --- COMMON SCAN EXECUTION ---
    if st.button("🔍 Scan Signature against Local AI", type="primary"):
        try:
            import joblib
            
            features = [float(x.strip()) for x in raw_input.split(',')]
            packet = np.array(features).reshape(1, -1)
            
            if packet.shape[1] != X_train.shape[1]:
                st.error(f"Shape Mismatch: Model requires {X_train.shape[1]} features. You provided {packet.shape[1]}.")
            else:
                # --- STRICT SCALING GATEKEEPER ---
                scaler_path = os.path.join(base_path, "../models/scaler.pkl")
                if not os.path.exists(scaler_path):
                    st.error("❌ CRITICAL ERROR: 'scaler.pkl' is missing from your models folder! You must export it from your Jupyter Notebook.")
                else:
                    scaler = joblib.load(scaler_path)
                    
                    # 1. Clip negative values (Matches Notebook Cell 6)
                    packet_clipped = np.clip(packet, a_min=0, a_max=None)
                    
                    # 2. Apply Log-Transformation (Matches Notebook Cell 6)
                    packet_log = np.log1p(packet_clipped)
                    
                    # 3. Apply the MinMaxScaler
                    packet_scaled = scaler.transform(packet_log) 
                    
                    # Feed the SCALED packet to the AI
                    recon = model.predict(packet_scaled, verbose=0)
                    error = float(np.mean(np.power(packet_scaled - recon, 2)))
                    conf, status = calculate_confidence(error, st.session_state['threshold'])
                    
                    st.markdown("### 🧬 Analysis Result")
                    if status == "Threat":
                        st.error(f"🚨 **THREAT BLOCKED** | **Confidence:** {conf:.1f}% | **Reconstruction Error:** {error:.5f}")
                    else:
                        st.success(f"✅ **TRAFFIC ALLOWED** | **Confidence:** {conf:.1f}% | **Reconstruction Error:** {error:.5f}")
        except Exception as e:
            st.error(f"❌ Error processing signature: {e}")
# --- FOOTER ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.caption("🛡️ **ProjectFALCON:** Federated Autoencoder-Based Latent Consensus Outlier-Resilient Network")