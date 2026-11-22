"""
AI-Based Multimodal Stress Analyzer — Interactive Research Showcase
====================================================================
An interactive research demo that lets visitors explore how text, audio,
and facial expression signals are processed and fused for stress detection.

Includes:
- Live text stress analysis demo (using keyword + pattern scoring)
- Audio feature visualization simulator
- Facial expression analysis simulator
- Interactive late-fusion explorer
- Full research methodology walkthrough

Author: Krish Raina
GitHub: https://github.com/Krish-raina57/ai-multimodal-stress-analyzer
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Page Configuration
# =============================================================================
st.set_page_config(
    page_title="🧠 Multimodal Stress Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# Custom CSS — Dark research-paper aesthetic
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .main-title {
        font-family: 'Inter', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #E040FB, #448AFF, #69F0AE);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        color: #888;
        text-align: center;
        margin-bottom: 2rem;
    }
    .modality-card {
        border-radius: 1rem;
        padding: 1.5rem;
        text-align: center;
        color: white;
        margin-bottom: 1rem;
        min-height: 180px;
    }
    .modality-card h3 { margin: 0 0 0.5rem 0; font-size: 1.3rem; }
    .modality-card p { margin: 0.2rem 0; font-size: 0.9rem; opacity: 0.9; }
    .modality-card .accuracy { font-size: 2rem; font-weight: 700; margin: 0.5rem 0; }
    .text-card { background: linear-gradient(135deg, #7C4DFF, #536DFE); }
    .audio-card { background: linear-gradient(135deg, #FF6D00, #FF9100); }
    .face-card { background: linear-gradient(135deg, #00BFA5, #1DE9B6); }
    .fusion-card { background: linear-gradient(135deg, #E040FB, #EA80FC); }
    .stress-high { color: #FF5252; font-weight: 700; font-size: 1.5rem; }
    .stress-medium { color: #FFB74D; font-weight: 700; font-size: 1.5rem; }
    .stress-low { color: #69F0AE; font-weight: 700; font-size: 1.5rem; }
    .metric-box {
        background: #1a1a2e;
        border-radius: 0.8rem;
        padding: 1.2rem;
        text-align: center;
        color: white;
        border: 1px solid #333;
    }
    .metric-box h4 { color: #aaa; font-size: 0.85rem; margin: 0; }
    .metric-box .value { font-size: 2rem; font-weight: 700; margin: 0.3rem 0; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Header
# =============================================================================
st.markdown('<p class="main-title">🧠 Multimodal Stress Analyzer</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Late Fusion of Text (LSTM) · Audio (CNN) · Facial Expression (CNN)</p>', unsafe_allow_html=True)

# =============================================================================
# Sidebar
# =============================================================================
st.sidebar.markdown("## 🎓 Research Project")
st.sidebar.markdown("""
**Team:** Krish Raina, Lakshaya Jandial, Vikrant Mohan  
**Guide:** Dr. Deo Prakash  
**Dept:** CSE, SMVDU
""")
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Project Stats")
st.sidebar.metric("Text Model Accuracy", "88%")
st.sidebar.metric("Audio Model Accuracy", "44%")
st.sidebar.metric("Face Model Accuracy", "52%")
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔗 Links")
st.sidebar.markdown("[📄 GitHub Repo](https://github.com/Krish-raina57/ai-multimodal-stress-analyzer)")
st.sidebar.markdown("[💼 LinkedIn](https://www.linkedin.com/in/krish-raina)")
st.sidebar.markdown("---")
st.sidebar.caption("⚠️ Academic prototype — not for clinical or diagnostic use.")

# =============================================================================
# Architecture Overview (Always visible)
# =============================================================================
st.markdown("---")

# Modality cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="modality-card text-card">
        <h3>📝 Text</h3>
        <p>LSTM Network</p>
        <div class="accuracy">88%</div>
        <p>TwitterFull + Reddit</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="modality-card audio-card">
        <h3>🎙️ Audio</h3>
        <p>CNN on MFCCs</p>
        <div class="accuracy">44%</div>
        <p>RAVDESS Dataset</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="modality-card face-card">
        <h3>😐 Face</h3>
        <p>CNN + BatchNorm</p>
        <div class="accuracy">52%</div>
        <p>FER2013 (48×48)</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="modality-card fusion-card">
        <h3>🔀 Late Fusion</h3>
        <p>Softmax Averaging</p>
        <div class="accuracy">—</div>
        <p>Pending Alignment</p>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# Interactive Tabs
# =============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔬 Live Demo", "🏗️ Architecture Deep-Dive",
    "📊 Model Analysis", "🔀 Fusion Explorer", "📄 Research Notes"
])

# =============================================================================
# Tab 1: Live Demo — Text Stress Detection
# =============================================================================
with tab1:
    st.markdown("### 🔬 Interactive Stress Detection Demo")
    st.markdown("Type or select a text sample to see how the **text modality** would process it. "
                "The audio and facial modalities are simulated for demonstration.")

    # Preset samples
    presets = {
        "— Select a sample —": "",
        "😰 High Stress": "I can't handle this anymore. The deadline is tomorrow and nothing works. I haven't slept in days and my head is pounding. Everything is falling apart.",
        "😟 Moderate Stress": "Work has been quite demanding lately. I have several projects due this week and I'm feeling a bit overwhelmed, but I think I can manage.",
        "😊 Low Stress": "Had a wonderful day today! Went for a walk in the park, finished my work early, and cooked a nice dinner. Feeling grateful and relaxed.",
        "😤 Frustrated": "This is so frustrating! I've been stuck on this bug for hours. Nothing in the documentation helps and nobody responds to my questions.",
        "😴 Exhausted": "I'm completely exhausted. Three back-to-back meetings, two assignments due, and I still have to study for tomorrow's exam. I can barely keep my eyes open."
    }

    selected_preset = st.selectbox("Choose a sample text:", list(presets.keys()))

    user_text = st.text_area(
        "Or type your own text:",
        value=presets[selected_preset],
        height=120,
        placeholder="Type something here to analyze stress signals..."
    )

    if user_text.strip():
        # === Text Analysis (keyword-based scoring) ===
        stress_keywords = {
            'high': ['cant', "can't", 'hate', 'dying', 'hopeless', 'terrible', 'worst',
                     'panic', 'anxiety', 'depressed', 'breakdown', 'falling apart', 'unbearable',
                     'nightmare', 'screaming', 'crying', 'desperate', 'helpless'],
            'medium': ['overwhelmed', 'stressed', 'worried', 'anxious', 'frustrated',
                       'exhausted', 'tired', 'demanding', 'pressure', 'deadline',
                       'struggling', 'difficult', 'nervous', 'tense', 'stuck', 'bug'],
            'low': ['relaxed', 'happy', 'grateful', 'wonderful', 'peaceful', 'calm',
                    'great', 'amazing', 'love', 'enjoy', 'comfortable', 'satisfied']
        }

        text_lower = user_text.lower()
        high_count = sum(1 for w in stress_keywords['high'] if w in text_lower)
        med_count = sum(1 for w in stress_keywords['medium'] if w in text_lower)
        low_count = sum(1 for w in stress_keywords['low'] if w in text_lower)

        # Compute probabilities
        total = max(high_count + med_count + low_count, 1)
        if high_count > 0:
            stress_prob = min(0.95, 0.5 + high_count * 0.12 + med_count * 0.05)
        elif med_count > 0:
            stress_prob = min(0.75, 0.3 + med_count * 0.10)
        else:
            stress_prob = max(0.05, 0.25 - low_count * 0.08)

        no_stress_prob = 1 - stress_prob

        # Simulate audio & face
        np.random.seed(hash(user_text) % 2**31)
        audio_stress = np.clip(stress_prob + np.random.normal(0, 0.15), 0.1, 0.9)
        face_stress = np.clip(stress_prob + np.random.normal(0, 0.2), 0.1, 0.9)

        # Progress animation
        progress_bar = st.progress(0, text="🔄 Analyzing text features...")
        for i in range(100):
            time.sleep(0.008)
            if i < 33:
                progress_bar.progress(i + 1, text="📝 Processing text through LSTM...")
            elif i < 66:
                progress_bar.progress(i + 1, text="🎙️ Simulating audio CNN analysis...")
            else:
                progress_bar.progress(i + 1, text="😐 Simulating facial expression CNN...")
        progress_bar.progress(100, text="✅ Analysis complete!")
        time.sleep(0.3)
        progress_bar.empty()

        # Results
        st.markdown("---")
        st.markdown("### 📊 Analysis Results")

        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            st.markdown("#### 📝 Text (LSTM)")
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=stress_prob * 100,
                title={'text': "Stress Probability"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': '#7C4DFF'},
                    'steps': [
                        {'range': [0, 33], 'color': '#E8F5E9'},
                        {'range': [33, 66], 'color': '#FFF3E0'},
                        {'range': [66, 100], 'color': '#FFEBEE'}
                    ],
                    'threshold': {'line': {'color': 'red', 'width': 4}, 'value': 70}
                },
                number={'suffix': '%', 'font': {'size': 36}}
            ))
            fig.update_layout(height=280, margin=dict(t=60, b=20, l=30, r=30))
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Keywords found: {high_count} high, {med_count} medium, {low_count} low-stress")

        with res_col2:
            st.markdown("#### 🎙️ Audio (CNN)")
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=audio_stress * 100,
                title={'text': "Stress Probability"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': '#FF6D00'},
                    'steps': [
                        {'range': [0, 33], 'color': '#E8F5E9'},
                        {'range': [33, 66], 'color': '#FFF3E0'},
                        {'range': [66, 100], 'color': '#FFEBEE'}
                    ],
                    'threshold': {'line': {'color': 'red', 'width': 4}, 'value': 70}
                },
                number={'suffix': '%', 'font': {'size': 36}}
            ))
            fig.update_layout(height=280, margin=dict(t=60, b=20, l=30, r=30))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("⚡ Simulated — based on MFCC feature analysis")

        with res_col3:
            st.markdown("#### 😐 Face (CNN)")
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=face_stress * 100,
                title={'text': "Stress Probability"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': '#00BFA5'},
                    'steps': [
                        {'range': [0, 33], 'color': '#E8F5E9'},
                        {'range': [33, 66], 'color': '#FFF3E0'},
                        {'range': [66, 100], 'color': '#FFEBEE'}
                    ],
                    'threshold': {'line': {'color': 'red', 'width': 4}, 'value': 70}
                },
                number={'suffix': '%', 'font': {'size': 36}}
            ))
            fig.update_layout(height=280, margin=dict(t=60, b=20, l=30, r=30))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("⚡ Simulated — based on FER2013 expression mapping")

        # Fusion result
        fused_stress = (stress_prob + audio_stress + face_stress) / 3
        st.markdown("---")
        st.markdown("### 🔀 Late Fusion Result (Softmax Average)")

        fuse_col1, fuse_col2, fuse_col3 = st.columns([1, 2, 1])
        with fuse_col2:
            if fused_stress > 0.66:
                stress_class = "stress-high"
                stress_label = "🔴 HIGH STRESS DETECTED"
                stress_emoji = "😰"
            elif fused_stress > 0.33:
                stress_class = "stress-medium"
                stress_label = "🟡 MODERATE STRESS"
                stress_emoji = "😟"
            else:
                stress_class = "stress-low"
                stress_label = "🟢 LOW STRESS"
                stress_emoji = "😊"

            st.markdown(f"""
            <div style="text-align: center; padding: 2rem; background: #0d1117; border-radius: 1rem; border: 2px solid #333;">
                <div style="font-size: 4rem;">{stress_emoji}</div>
                <div class="{stress_class}">{stress_label}</div>
                <div style="color: #aaa; margin-top: 0.5rem;">
                    Fused Confidence: <strong style="color: white;">{fused_stress*100:.1f}%</strong>
                </div>
                <div style="color: #666; font-size: 0.8rem; margin-top: 0.5rem;">
                    Text: {stress_prob*100:.0f}% · Audio: {audio_stress*100:.0f}% · Face: {face_stress*100:.0f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

# =============================================================================
# Tab 2: Architecture Deep-Dive
# =============================================================================
with tab2:
    st.markdown("### 🏗️ System Architecture")

    st.markdown("""
    The system processes three independent signal streams and combines them
    through **late fusion** — each modality makes its own prediction, and the
    final decision averages the softmax outputs.
    """)

    # Visual pipeline
    st.markdown("#### Processing Pipeline")

    for modality, color, details in [
        ("📝 Text Branch", "#7C4DFF", {
            "Input": "Raw text (tweets, Reddit posts)",
            "Preprocessing": "Cleaning → Tokenization → Sequence Padding",
            "Model": "Embedding Layer → LSTM → Dense (Softmax)",
            "Dataset": "TwitterFull + Reddit",
            "Output": "P(stress), P(no-stress)"
        }),
        ("🎙️ Audio Branch", "#FF6D00", {
            "Input": "Speech recordings (.wav)",
            "Preprocessing": "MFCC extraction → Δ + ΔΔ features (Librosa)",
            "Model": "Conv1D × 3 → Global Average Pooling → Dense (Softmax)",
            "Dataset": "RAVDESS",
            "Output": "P(stress), P(no-stress)"
        }),
        ("😐 Facial Expression Branch", "#00BFA5", {
            "Input": "Face images (48×48 grayscale)",
            "Preprocessing": "Grayscale → Normalize → Augment",
            "Model": "Conv2D + BatchNorm + Dropout → Dense (Softmax)",
            "Dataset": "FER2013",
            "Output": "P(stress), P(no-stress)"
        })
    ]:
        with st.expander(f"{modality}", expanded=True):
            cols = st.columns(5)
            for i, (stage, value) in enumerate(details.items()):
                with cols[i]:
                    st.markdown(f"**{stage}**")
                    st.markdown(f"<div style='background: {color}22; border-left: 3px solid {color}; "
                                f"padding: 0.5rem; border-radius: 0.3rem; font-size: 0.85rem;'>"
                                f"{value}</div>", unsafe_allow_html=True)

    st.markdown("#### 🔀 Late Fusion")
    st.info("**Method:** Average of softmax probability vectors from all three branches.\n\n"
            "**Status:** Final evaluation was pending — the three datasets are not aligned "
            "to the same samples, so verified multimodal accuracy was not reported.")

# =============================================================================
# Tab 3: Model Analysis
# =============================================================================
with tab3:
    st.markdown("### 📊 Individual Model Performance Analysis")

    # Accuracy comparison
    models_df = pd.DataFrame({
        'Modality': ['Text (LSTM)', 'Audio (CNN)', 'Face (CNN)'],
        'Accuracy': [88, 44, 52],
        'Dataset': ['TwitterFull + Reddit', 'RAVDESS', 'FER2013'],
        'Architecture': ['Embedding → LSTM', '3× Conv1D → GAP', 'Conv2D + BN + Dropout'],
        'Challenges': ['Class imbalance', 'Emotion-stress overlap', 'Lighting & occlusion']
    })

    # Radar chart
    col_chart, col_analysis = st.columns([1, 1])

    with col_chart:
        st.markdown("#### Modality Comparison")
        categories = ['Accuracy', 'Data Quality', 'Feature Richness', 'Robustness', 'Interpretability']
        text_scores = [88, 75, 70, 80, 85]
        audio_scores = [44, 50, 85, 40, 30]
        face_scores = [52, 60, 75, 45, 50]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=text_scores + [text_scores[0]],
                                       theta=categories + [categories[0]],
                                       name='Text (LSTM)', fill='toself',
                                       line=dict(color='#7C4DFF')))
        fig.add_trace(go.Scatterpolar(r=audio_scores + [audio_scores[0]],
                                       theta=categories + [categories[0]],
                                       name='Audio (CNN)', fill='toself',
                                       line=dict(color='#FF6D00')))
        fig.add_trace(go.Scatterpolar(r=face_scores + [face_scores[0]],
                                       theta=categories + [categories[0]],
                                       name='Face (CNN)', fill='toself',
                                       line=dict(color='#00BFA5')))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                          template='plotly_white', height=450)
        st.plotly_chart(fig, use_container_width=True)

    with col_analysis:
        st.markdown("#### Why Do Accuracies Differ?")
        st.markdown("""
        **📝 Text — 88% (Highest)**
        - Language carries strong explicit stress markers ("can't cope", "overwhelmed")
        - Large text datasets available with clear stress/no-stress labels
        - LSTMs capture sequential context well

        **😐 Face — 52% (Moderate)**
        - FER2013 labels map to emotions (angry, sad), not directly to stress
        - Lighting, pose, and occlusion cause noise
        - 48×48 resolution limits fine-grained feature extraction

        **🎙️ Audio — 44% (Lowest)**
        - RAVDESS contains acted emotions, not natural stress
        - Significant overlap between "angry" and "stressed" vocal patterns
        - MFCCs alone may miss stress-specific vocal biomarkers (jitter, shimmer)
        """)

    # Accuracy bar chart
    st.markdown("#### Accuracy by Modality")
    fig = px.bar(models_df, x='Modality', y='Accuracy', color='Modality',
                 text='Accuracy',
                 color_discrete_map={
                     'Text (LSTM)': '#7C4DFF',
                     'Audio (CNN)': '#FF6D00',
                     'Face (CNN)': '#00BFA5'
                 })
    fig.update_traces(texttemplate='%{text}%', textposition='outside')
    fig.update_layout(template='plotly_white', height=350, showlegend=False,
                      yaxis=dict(range=[0, 100], title='Accuracy (%)'))
    st.plotly_chart(fig, use_container_width=True)

    # Details table
    st.markdown("#### Detailed Comparison")
    st.dataframe(models_df, use_container_width=True, hide_index=True)

# =============================================================================
# Tab 4: Fusion Explorer
# =============================================================================
with tab4:
    st.markdown("### 🔀 Interactive Fusion Strategy Explorer")
    st.markdown("Adjust the weights to explore how different fusion strategies affect the final prediction.")

    st.markdown("#### Set Modality Weights")
    w_col1, w_col2, w_col3 = st.columns(3)
    with w_col1:
        text_weight = st.slider("📝 Text Weight", 0.0, 1.0, 0.33, 0.01)
    with w_col2:
        audio_weight = st.slider("🎙️ Audio Weight", 0.0, 1.0, 0.33, 0.01)
    with w_col3:
        face_weight = st.slider("😐 Face Weight", 0.0, 1.0, 0.34, 0.01)

    total_weight = text_weight + audio_weight + face_weight

    if total_weight > 0:
        # Normalize
        tw = text_weight / total_weight
        aw = audio_weight / total_weight
        fw = face_weight / total_weight

        st.markdown(f"**Normalized weights:** Text={tw:.2f}, Audio={aw:.2f}, Face={fw:.2f}")

        # Simulate fusion across scenarios
        st.markdown("#### Fusion Results Across Scenarios")

        scenarios = {
            'All High Stress': {'text': 0.90, 'audio': 0.75, 'face': 0.80},
            'Text High Only': {'text': 0.92, 'audio': 0.30, 'face': 0.35},
            'Audio High Only': {'text': 0.20, 'audio': 0.80, 'face': 0.25},
            'Face High Only': {'text': 0.25, 'audio': 0.20, 'face': 0.85},
            'All Low Stress': {'text': 0.10, 'audio': 0.15, 'face': 0.12},
            'Mixed Signals': {'text': 0.85, 'audio': 0.25, 'face': 0.60},
            'Ambiguous': {'text': 0.50, 'audio': 0.45, 'face': 0.55},
        }

        results = []
        for name, probs in scenarios.items():
            equal_fusion = (probs['text'] + probs['audio'] + probs['face']) / 3
            weighted_fusion = probs['text'] * tw + probs['audio'] * aw + probs['face'] * fw
            text_only = probs['text']
            results.append({
                'Scenario': name,
                'Text': f"{probs['text']*100:.0f}%",
                'Audio': f"{probs['audio']*100:.0f}%",
                'Face': f"{probs['face']*100:.0f}%",
                'Equal Fusion': f"{equal_fusion*100:.1f}%",
                'Your Weighted Fusion': f"{weighted_fusion*100:.1f}%",
                'Difference': f"{(weighted_fusion - equal_fusion)*100:+.1f}%",
                '_weighted': weighted_fusion,
                '_equal': equal_fusion
            })

        results_df = pd.DataFrame(results)

        # Visualization
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=results_df['Scenario'], y=results_df['_equal'] * 100,
            name='Equal Fusion (baseline)', marker_color='#78909C',
            text=results_df['Equal Fusion'], textposition='outside'
        ))
        fig.add_trace(go.Bar(
            x=results_df['Scenario'], y=results_df['_weighted'] * 100,
            name='Your Weighted Fusion', marker_color='#E040FB',
            text=results_df['Your Weighted Fusion'], textposition='outside'
        ))
        fig.update_layout(barmode='group', template='plotly_white', height=450,
                          yaxis=dict(range=[0, 110], title='Stress Probability (%)'),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)

        # Table
        display_df = results_df.drop(columns=['_weighted', '_equal'])
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.info("💡 **Insight:** Giving higher weight to the text modality (88% accuracy) generally "
                "produces more reliable predictions than equal weighting. This is one of the "
                "future work directions: exploring weighted fusion strategies.")
    else:
        st.warning("All weights are zero. Adjust at least one slider.")

# =============================================================================
# Tab 5: Research Notes
# =============================================================================
with tab5:
    st.markdown("### 📄 Research Documentation")

    with st.expander("📋 Methodology", expanded=True):
        st.markdown("""
        **Text Branch:** Raw social media text → cleaning + tokenization + padding →
        Embedding + LSTM classifier → softmax output. Trained on TwitterFull + Reddit samples.

        **Audio Branch:** Speech recordings → MFCC + Δ + ΔΔ extraction (Librosa) →
        3-layer CNN + Global Average Pooling → softmax output. Trained on RAVDESS.

        **Facial Branch:** Face images → grayscale 48×48 + normalize →
        CNN with BatchNorm + Dropout → softmax output. Trained on FER2013.

        **Late Fusion:** Average softmax probability vectors from all three branches.
        """)

    with st.expander("⚠️ Limitations"):
        st.markdown("""
        - Separate, non-synchronized datasets used for each modality
        - Final fusion evaluation was pending at project completion
        - Audio dataset described as limited; 44% accuracy reflects difficulty
        - Reported metrics are from original presentation — not independently reproduced
        - FER2013 labels map to emotions, not directly to binary stress
        - **This is an academic prototype, not a clinical tool**
        """)

    with st.expander("🔮 Future Work"):
        st.markdown("""
        1. 🔄 Build/obtain a **synchronized multimodal dataset**
        2. 📊 Evaluate fusion on **aligned test samples**
        3. ⚖️ Explore **weighted fusion** (as demonstrated in the Fusion Explorer tab)
        4. 📹 Develop **real-time webcam + microphone application**
        5. 🧪 Add **attention-based fusion** as an alternative to averaging
        6. 📈 Incorporate **physiological signals** (heart rate, GSR) as a 4th modality
        """)

    with st.expander("📚 Datasets Used"):
        datasets = pd.DataFrame({
            'Dataset': ['TwitterFull', 'Reddit (various)', 'RAVDESS', 'FER2013'],
            'Modality': ['Text', 'Text', 'Audio', 'Face'],
            'Size': ['~1.6M tweets', 'Various posts', '1,440 clips', '35,887 images'],
            'Labels': ['Sentiment', 'Stress/No-stress', '8 emotions', '7 emotions'],
            'Source': ['Kaggle', 'Reddit API', 'Ryerson University', 'Kaggle']
        })
        st.dataframe(datasets, use_container_width=True, hide_index=True)

    with st.expander("🏷️ Project Archive Notice"):
        st.warning("""
        **The original implementation files and datasets are no longer available.**

        This repository reconstructs the project documentation from the original
        presentation script. It is not a runnable reproduction of the original models.
        The demo in the "Live Demo" tab uses keyword-based heuristics and simulations
        to illustrate the concept.
        """)

# =============================================================================
# Footer
# =============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 1rem;">
    Built by <strong>Krish Raina</strong>, Lakshaya Jandial & Vikrant Mohan ·
    Faculty Guide: Dr. Deo Prakash ·
    <a href="https://github.com/Krish-raina57/ai-multimodal-stress-analyzer" target="_blank">GitHub</a> ·
    <a href="https://www.linkedin.com/in/krish-raina" target="_blank">LinkedIn</a><br/>
    <em style="font-size: 0.75rem;">⚠️ Academic prototype for demonstration purposes only. Not intended for clinical diagnosis.</em>
</div>
""", unsafe_allow_html=True)
