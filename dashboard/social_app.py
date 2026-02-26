"""
SENTRY-X Demo: Integrated Social Media Platform ("Vibe")
Shows how SENTRY-X works silently in the background of a social app.
"""
import streamlit as st
import json
from datetime import datetime
import io
import uuid
import time
import requests
import os

# Configuration
API_BASE = "http://localhost:8000"
DB_DIR = "dashboard/db"
POSTS_FILE = os.path.join(DB_DIR, "posts.json")
UPLOAD_DIR = os.path.join(DB_DIR, "uploads")

os.makedirs(UPLOAD_DIR, exist_ok=True)
if not os.path.exists(POSTS_FILE):
    with open(POSTS_FILE, "w") as f:
        json.dump([], f)

st.set_page_config(
    page_title="Vibe | Social Feed",
    page_icon="📱",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Dark Mode Compatible Custom CSS
st.markdown("""
<style>
/* Reset and Font Setup */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* Post Container */
.post-container {
    background-color: #1e293b; /* Dark theme default */
    border: 1px solid #334155;
    border-radius: 12px;
    margin-bottom: 25px;
    overflow: hidden;
    position: relative;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
}

.post-header {
    display: flex;
    align-items: center;
    padding: 12px 16px;
    background-color: #1e293b;
    border-bottom: 1px solid #334155;
}

.user-avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    margin-right: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: 600;
    font-size: 14px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.user-name {
    font-weight: 600;
    color: #f1f5f9;
    font-size: 14px;
    letter-spacing: -0.01em;
}

.post-media {
    width: 100%;
    background: #0f172a; /* Darker bg for media */
    display: flex;
    justify-content: center;
}

.post-footer {
    padding: 16px;
    background-color: #1e293b;
}

.timestamp {
    color: #94a3b8;
    font-size: 12px;
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 500;
}

.post-caption {
    color: #e2e8f0;
    font-size: 14px;
    line-height: 1.5;
}

.caption-username {
    font-weight: 600;
    margin-right: 6px;
    color: #f1f5f9;
}

/* SENTRY-X Warning Badges */
.sentry-badge {
    margin: 12px 16px;
    padding: 12px 16px;
    border-radius: 8px;
    font-size: 13px;
    line-height: 1.5;
    display: flex;
    align-items: flex-start;
    gap: 12px;
    animation: fadeIn 0.5s ease-out;
}

.sentry-badge-icon {
    font-size: 18px;
}

.sentry-badge.yellow { 
    background-color: #422006; 
    border: 1px solid #854d0e;
    color: #fde047;
}

.sentry-badge.orange { 
    background-color: #431407; 
    border: 1px solid #9a3412;
    color: #fed7aa;
}

/* Admin Dashboard Elements */
.admin-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-left: 4px solid #ef4444;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 16px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

.admin-card.orange { border-left-color: #f97316; }
.admin-card.yellow { border-left-color: #eab308; }

.status-pill {
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.status-blocked { background: #450a0a; color: #fca5a5; }
.status-restricted { background: #431407; color: #fdba74; }
.status-labeled { background: #422006; color: #fde047; }

/* Login Container */
.login-container {
    background: #1e293b;
    padding: 30px;
    border-radius: 12px;
    border: 1px solid #334155;
    margin-top: 50px;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(-5px); }
    to { opacity: 1; transform: translateY(0); }
}

</style>
""", unsafe_allow_html=True)

# ── Data Management ──

def get_posts():
    try:
        with open(POSTS_FILE, "r") as f:
            return json.load(f)
    except:
        return []

def save_posts(posts):
    with open(POSTS_FILE, "w") as f:
        json.dump(posts, f, indent=4)

# ── Session State ──
if "user" not in st.session_state:
    st.session_state.user = None
if "role" not in st.session_state:
    st.session_state.role = "user"
if "avatar_color" not in st.session_state:
    st.session_state.avatar_color = "#3b82f6" # Default blue

# Helper to assign random colors to avatars
def get_user_color(username):
    colors = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899", "#14b8a6"]
    hash_val = sum(ord(c) for c in username)
    return colors[hash_val % len(colors)]

# ── Pages ──

def login_page():
    # Remove default styling wrapper for cleaner login
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            '<h1 style="text-align:center; font-weight:800; font-size:3rem; margin-bottom:0; letter-spacing:-0.05em; color:white;">Vibe</h1>', 
            unsafe_allow_html=True
        )
        st.markdown(
            '<p style="text-align:center; color:#94a3b8; margin-bottom:30px;">Social connection, secured by SENTRY-X.</p>', 
            unsafe_allow_html=True
        )
        
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        st.markdown("### Log In")
        username = st.text_input("Username (any name)")
        password = st.text_input("Password (optional)", type="password") 
        
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Log In as User", use_container_width=True, type="primary"):
                if username:
                    st.session_state.user = username
                    st.session_state.role = "user"
                    st.session_state.avatar_color = get_user_color(username)
                    st.rerun()
                else:
                    st.error("Please enter a username!")
        with c2:
            if st.button("SENTRY-X Admin Log In", use_container_width=True):
                st.session_state.user = "Trust & Safety Team"
                st.session_state.role = "admin"
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.info("💡 **Demo Strategy:**\n1. Log in as a normal user to upload authentic and manipulated media.\n2. See how the platform responds automatically.\n3. Log in as Admin to see the backend moderation queue.")

def feed_page():
    st.markdown('<h2 style="font-weight:700; margin-bottom:20px;">Feed</h2>', unsafe_allow_html=True)
    
    posts = get_posts()
    # Show newest first
    posts = list(reversed(posts))
    
    # Filter out auto-blocked posts for the normal timeline!
    visible_posts = [p for p in posts if p.get("risk_level") != "red"]
    
    if not visible_posts:
        st.info("No posts on your timeline yet! Be the first to share something.")
        
    for post in visible_posts:
        author = post['author']
        initial = author[0].upper()
        color = get_user_color(author)
        
        st.markdown(f'<div class="post-container">', unsafe_allow_html=True)
        
        # 1. Header
        st.markdown(f"""
        <div class="post-header">
            <div class="user-avatar" style="background-color: {color};">{initial}</div>
            <div class="user-name">{author}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # 2. Media Image/Video Wrapper 
        # Since Streamlit doesn't let us easily insert native media inside raw HTML, we close HTML, render media, reopen HTML
        st.markdown('<div class="post-media">', unsafe_allow_html=True)
        
        media_path = post["media_path"]
        if os.path.exists(media_path):
            if media_path.lower().endswith(('.mp4', '.mov', '.avi', '.webm', '.mkv')):
                st.video(media_path)
            else:
                st.image(media_path, use_column_width=True)
        else:
            st.error("Media file missing.")
            
        st.markdown('</div>', unsafe_allow_html=True)

        # 3. Badges applied by SENTRY-X
        risk = post.get("risk_level", "green")
        visibility = post.get("visibility_multiplier", 1.0)
        
        if risk == "yellow":
            st.markdown(f'''
            <div class="sentry-badge yellow">
                <div class="sentry-badge-icon">🤖</div>
                <div>
                    <b>Recognized as AI-Generated Context</b><br>
                    Independent systems flagged this media as likely generated or heavily modified by AI algorithms.
                    <br><span style="font-size: 11px; opacity: 0.8; margin-top: 4px; display: block;">Reach Throttled: {visibility*100}%</span>
                </div>
            </div>
            ''', unsafe_allow_html=True)
        elif risk == "orange":
            st.markdown(f'''
            <div class="sentry-badge orange">
                <div class="sentry-badge-icon">⚠️</div>
                <div>
                    <b>Manipulated Media Warning</b><br>
                    SENTRY-X forensic analysis detected significant digital alteration (Deepfake) in this content. Viewer discretion advised.
                    <br><span style="font-size: 11px; opacity: 0.8; margin-top: 4px; display: block;">Reach Heavily Suppressed: {visibility*100}% (Removed from feed suggestions)</span>
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        # 4. Footer (Caption)
        st.markdown(f"""
        <div class="post-footer">
            <div class="timestamp">{post["timestamp"]}</div>
            <div class="post-caption">
                <span class="caption-username">{author}</span> {post["caption"]}
            </div>
        </div>
        </div>  <!-- End post container -->
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)

def upload_page():
    st.markdown('<h2 style="font-weight:700; margin-bottom:20px;">Create Post</h2>', unsafe_allow_html=True)
    
    st.markdown("<p style='color:#94a3b8;'>Share a photo or video to your followers. SENTRY-X analyzes all uploads in real-time before they reach the feed.</p>", unsafe_allow_html=True)
    
    with st.container(border=True):
        uploaded_file = st.file_uploader("Select Media", type=["jpg", "jpeg", "png", "heic", "heif", "mp4", "mov", "avi", "webm"], label_visibility="hidden")
        
        caption = st.text_area("Caption:", placeholder="Write something about this...")
        
        if st.button("Share Post", type="primary", use_container_width=True):
            if uploaded_file is None:
                st.error("Select a photo or video first.")
                return
                
            is_video = uploaded_file.name.lower().endswith(('mp4', 'mov', 'avi', 'webm'))
            file_ext = uploaded_file.name.split(".")[-1]
            file_bytes = uploaded_file.read()
            
            # --- UPLOAD FLOW ANIMATION ---
            progress_container = st.container()
            with progress_container:
                st.markdown("---")
                status = st.empty()
                status.info("📤 Uploading media...")
                
                file_id = str(uuid.uuid4())
                media_path = os.path.join(UPLOAD_DIR, f"{file_id}.{file_ext}")
                with open(media_path, "wb") as f:
                    f.write(file_bytes)
                    
                time.sleep(1) # Fake delay for effect
                status.info("🛡️ SENTRY-X Trust Engine analyzing for deepfakes...")
                
                # --- SENTRY-X API CALL ---
                endpoint = f"{API_BASE}/v1/analyze/video" if is_video else f"{API_BASE}/v2/analyze"
                
                import mimetypes
                guess_mime, _ = mimetypes.guess_type(uploaded_file.name)
                if not guess_mime:
                    guess_mime = f"video/{file_ext}" if is_video else f"image/{file_ext}"
                if file_ext in ["heic", "heif"]:
                    guess_mime = f"image/{file_ext}"
                mime_type = guess_mime
                
                files = {"file": (uploaded_file.name, io.BytesIO(file_bytes), mime_type)}
                data = {"platform_id": "vibe-social", "sample_fps": 1.0} if is_video else {"platform_id": "vibe-social", "uploader_id": st.session_state.user, "caption": caption}
                timeout = 300 if is_video else 60
                
                try:
                    r = requests.post(endpoint, files=files, data=data, timeout=timeout)
                    result = r.json()
                    
                    if r.status_code != 200:
                        status.error(f"SENTRY-X Error: {result}")
                        return
                        
                    # Format standard schema whether video or image
                    if is_video:
                        v_info = result.get("verdict", {})
                        risk = v_info.get("risk_level", "green")
                        confidence = v_info.get("max_confidence", 0)
                        reason = v_info.get("description", "Authentic")
                        action_taken = result.get("action", "publish")
                        fingerprint_sha = result.get("fingerprint", {}).get("sha256", "N/A")
                        visibility = 1.0
                        generator = "N/A"
                        latency = v_info.get("processing_time_ms", 0.0)
                    else:
                        policy = result.get("amplification_policy", {})
                        risk = policy.get("tier", "green")
                        
                        # In fast_path, confidence might not be inside detection_signals
                        signals = result.get("detection_signals", {})
                        confidence = signals.get("fusion_threat_score", 1.0 if risk in ["red", "orange"] else 0)
                        
                        reason = policy.get("policy_enforcement", "Authentic")
                        action_taken = policy.get("action", "publish")
                        fingerprint_sha = result.get("fingerprints", {}).get("sha256", "N/A")
                        visibility = policy.get("visibility_multiplier", 1.0)
                        
                        generator = "Proven Ensemble" if signals else "Known Threat"
                        
                        latency_profile = result.get("latency_profile_ms", {})
                        # fast_path might only have phase1_triage_ms
                        latency = latency_profile.get("total_pipeline_ms") or latency_profile.get("phase1_triage_ms", 0.0)
                        
                        # Let's extract any multilingual matched context language if available
                        intent_sigs = result.get("intent_classification", {}).get("signals", [])
                        for sig in intent_sigs:
                            if "Matched Context Language" in sig:
                                reason += f" | {sig}"
                                
                    # Save to JSON database
                    post = {
                        "id": file_id,
                        "author": st.session_state.user,
                        "media_path": media_path,
                        "caption": caption,
                        "timestamp": datetime.now().strftime("%B %d, %Y at %I:%M %p"),
                        "risk_level": risk,
                        "confidence": confidence,
                        "sentry_reason": reason,
                        "fingerprint": fingerprint_sha,
                        "action_taken": action_taken,
                        "visibility_multiplier": visibility,
                        "generator": generator,
                        "latency": latency
                    }
                    
                    posts = get_posts()
                    posts.append(post)
                    save_posts(posts)
                    
                    status.empty()
                    
                    # --- ACTION BASED ON RISK LEVEL (VISUAL DEMO) ---
                    if risk == "red":
                        st.error(f"#### 🛑 SENTRY-X Intervention: Upload Blocked\n\nHigh-confidence manipulation detected (**{confidence*100:.1f}% confidence**). This violates our Community Guidelines regarding deceptive media.\n\nYour post has been blocked and the cryptographic hash permanently registered in the provenance ledger.")
                        with st.expander("View Forensic Output Details"):
                            st.write(f"**Analysis Reason:** {reason}")
                            st.write(f"**Ledger Hash ID:** `{post['fingerprint']}`")
                            
                    elif risk == "orange":
                        st.warning(f"#### ⚠️ Warning: Manipulated Media\n\nSENTRY-X forensic analysis detected significant digital alteration (**{confidence*100:.1f}% confidence**). \n\nYour post has been published, but algorithmic reach has been restricted and a visible warning label has been applied to viewers.")
                        st.success("Post uploaded with restrictions.")
                        
                    elif risk == "yellow":
                        st.info(f"#### 🤖 Notice: AI-Generated Content Label\n\nSENTRY-X origin detection flagged this media as likely synthetic. To maintain transparency, a disclosure label has been automatically appended to your post.")
                        st.success("Post uploaded successfully.")
                        
                    else:
                        st.success("✅ **Upload Complete!** SENTRY-X verified content integrity.")
                        st.balloons()
                        
                except Exception as e:
                    status.error(f"Cannot reach SENTRY-X internal firewall. Service offline. ({e})")

def admin_page():
    st.markdown('<h2 style="font-weight:800; color:#38bdf8; margin-bottom:5px;">🛡️ Moderation Dashboard</h2>', unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8; font-size:14px; margin-bottom:20px;'>SENTRY-X Advanced Trust & Safety Center</p>", unsafe_allow_html=True)
    
    posts = get_posts()
    
    total_scans = len(posts)
    blocked = len([p for p in posts if p.get("risk_level") == "red"])
    restricted = len([p for p in posts if p.get("risk_level") == "orange"])
    labeled = len([p for p in posts if p.get("risk_level") == "yellow"])
    
    # Hero metric row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="admin-card" style="border-left:none; text-align:center;"><div style="font-size:2rem; font-weight:800; color:white;">{total_scans}</div><div style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase;">Network Scans</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="admin-card yellow" style="text-align:center;"><div style="font-size:2rem; font-weight:800; color:#fde047;">{labeled}</div><div style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase;">AI Labeled</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="admin-card orange" style="text-align:center;"><div style="font-size:2rem; font-weight:800; color:#fdba74;">{restricted}</div><div style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase;">Restricted</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="admin-card red" style="border-left-color:#ef4444; text-align:center;"><div style="font-size:2rem; font-weight:800; color:#fca5a5;">{blocked}</div><div style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase;">Auto-Blocked</div></div>', unsafe_allow_html=True)

    st.markdown("### 🚨 SENTRY-X Incident Log")
    
    flagged_posts = [p for p in reversed(posts) if p.get("risk_level") != "green"]
    
    if not flagged_posts:
        st.info("Platform integrity verified. No malicious media detected in queue.")
        return
        
    for post in flagged_posts:
        risk = post["risk_level"]
        card_class = f"admin-card {risk}"
        
        status_map = {
            "red": ("status-blocked", "BLOCKED: DEEPFAKE"),
            "orange": ("status-restricted", "RESTRICTED: MANIPULATED"),
            "yellow": ("status-labeled", "LABELED: SYNTHETIC")
        }
        pill_class, label_text = status_map[risk]
        
        st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
        
        # Header of incident
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; margin-bottom:12px;">
            <div style="font-weight:600; color:white;">Incident ID: {post['id'].split('-')[0].upper()}</div>
            <div class="status-pill {pill_class}">{label_text}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Content split
        col1, col2 = st.columns([1, 2.5])
        
        with col1:
            media_path = post["media_path"]
            if os.path.exists(media_path):
                if media_path.lower().endswith(('.mp4', '.mov', '.avi', '.webm', '.mkv')):
                    st.video(media_path)
                else:
                    st.image(media_path)
            else:
                st.markdown("*(Media removed from disk)*")
                
        with col2:
            st.markdown(f"""
            <div style="font-size:14px; color:#cbd5e1; line-height:1.6;">
                <b>Uploader:</b> @{post['author']}<br>
                <b>Timestamp:</b> {post['timestamp']}<br>
                <b>Confidence Score:</b> <span style="color:white; font-weight:700;">{post['confidence']*100:.1f}%</span><br>
                <b>Ledger / SHA-256 Hash:</b> <code style="font-size:11px; background:#0f172a; padding:2px 4px;">{post['fingerprint']}</code><br>
                <b>Detection Paradigm:</b> <span style="color:#fbbf24; font-weight:600;">{post.get('generator', 'Unknown')}</span><br>
                <b>Processing Latency:</b> <span style="color:#94a3b8;">{post.get('latency', 0.0):.2f} ms</span><br><br>
                <b>SENTRY Engine Audit Log:</b><br>
                <div style="background:#0f172a; padding:8px 12px; border-radius:6px; border:1px solid #334155; margin-top:4px; font-size:12px; color:#94a3b8;">
                    {post['sentry_reason']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)

# ── Main Entrypoint ──

def main():
    if not st.session_state.user:
        login_page()
        return

    # User Sidebar
    with st.sidebar:
        st.markdown(f"""
        <div style="display:flex; align-items:center; margin-bottom:20px;">
            <div class="user-avatar" style="background-color:{st.session_state.avatar_color}; width:40px; height:40px; font-size:18px;">
                {st.session_state.user[0].upper()}
            </div>
            <div style="margin-left:12px;">
                <div style="font-weight:bold; font-size:18px;">{st.session_state.user}</div>
                <div style="color:#94a3b8; font-size:12px; text-transform:uppercase;">
                    {st.session_state.role} ACCOUNT
                </div>
            </div>
        </div>
        <hr style="border-color:#334155; margin-top:0;">
        """, unsafe_allow_html=True)
        
        if st.session_state.role == "admin":
            st.markdown("<p style='color:#38bdf8; font-size:13px; font-weight:bold; letter-spacing:0.05em; margin-bottom:10px;'>TRUST & SAFETY</p>", unsafe_allow_html=True)
            menu = ["Admin Dashboard", "Logout"]
        else:
            st.markdown("<p style='color:#94a3b8; font-size:13px; font-weight:bold; letter-spacing:0.05em; margin-bottom:10px;'>NAVIGATION</p>", unsafe_allow_html=True)
            menu = ["Feed", "New Post", "Logout"]
            
        choice = st.radio("Menu", menu, label_visibility="collapsed")
        
        st.markdown("""
        <div style="position: absolute; bottom: 20px; font-size: 11px; color:#475569;">
            Powered by SENTRY-X<br>
            Media Integrity Firewall
        </div>
        """, unsafe_allow_html=True)
        
    # Routing
    if choice == "Feed":
        feed_page()
    elif choice == "New Post":
        upload_page()
    elif choice == "Admin Dashboard":
        admin_page()
    elif choice == "Logout":
        st.session_state.user = None
        st.session_state.role = "user"
        st.session_state.avatar_color = "#3b82f6"
        st.rerun()

if __name__ == "__main__":
    main()
