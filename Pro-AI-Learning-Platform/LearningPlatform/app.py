# Pro version: Attractive AI Learning Platform
# Works with Python 3.11 + latest Streamlit
# Uses requests to call DeepSeek/OpenRouter-style Chat Completions

import streamlit as st
import requests
import pandas as pd
import numpy as np
import base64
from io import BytesIO
from datetime import datetime
import random
import os
import textwrap
import time

# ---------------- Page & CSS ----------------
st.set_page_config(page_title="Pro AI Learning Platform", layout="wide", page_icon="🎓")

CSS = """
<style>
:root {
  --bg1: #0f172a;
  --card: rgba(255,255,255,0.03);
  --accent1: #5ab0ff;
  --accent2: #ff6ec7;
  --text: #e6eef8;
  --muted: #9aa6b2;
}
body { background: linear-gradient(180deg, #07102a 0%, #0f172a 100%); color: var(--text); }
.block-container{padding-top:1rem;}
.header-card{
  background: linear-gradient(90deg, rgba(90,176,255,0.12), rgba(255,110,199,0.09));
  border-radius:14px; padding:18px; margin-bottom:12px; box-shadow:0 8px 30px rgba(0,0,0,0.35);
}
.card{
  background: rgba(255,255,255,0.02); border-radius:12px; padding:14px; margin-bottom:12px;
  border: 1px solid rgba(255,255,255,0.03);
}
.h1 { color: #bfe9ff; font-weight:700; }
.h2 { color: #ffd1ea; font-weight:700; }
.small { color: var(--muted); font-size:12px; }
.kpi { padding:12px; border-radius:10px; background: linear-gradient(90deg, rgba(90,176,255,0.06), rgba(255,110,199,0.04)); text-align:center; }
.profile-photo { border-radius: 12px; border:1px solid rgba(255,255,255,0.04); }
.question-card{ background: rgba(255,255,255,0.01); padding:10px; border-radius:10px; margin-bottom:8px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------------- Session defaults ----------------
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'profile' not in st.session_state:
    st.session_state.profile = {}
if 'scores' not in st.session_state:
    st.session_state.scores = []  # list of dicts: {date, program, score}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# ---------------- Static Data ----------------
PROGRAMS = [
    "AI","ML","Business Analytics","BBA","Data Analytics","Robotics",
    "Biotechnology","Agriculture","Law","Hospital Management","Digital Marketing"
]

AI_TOOLS = {
    "AI": [("GPT/DeepSeek","Explanation & code"), ("Kaggle","Datasets & notebooks"), ("Colab","Free notebooks")],
    "ML": [("Scikit-learn","Classical ML"), ("TensorFlow","Deep Learning"), ("Weights & Biases","Experiment tracking")],
    "Business Analytics": [("Power BI","Dashboards"), ("Tableau","Viz"), ("Excel","Reporting")],
    "BBA": [("Excel","Finance models"), ("Notion","Notes"), ("Grammarly","Writing")],
    "Data Analytics": [("Pandas","Data manipulation"), ("SQL","Queries"), ("Plotly","Interactive charts")],
    "Robotics": [("ROS/ROS2","Middleware"), ("Gazebo","Simulation"), ("OpenCV","Vision")],
    "Biotechnology": [("Biopython","Bioinformatics"), ("NCBI/PubMed","Research"), ("BLAST","Sequence search")],
    "Agriculture": [("GIS/QGIS","Mapping"), ("Remote sensing","Crop monitoring")],
    "Law": [("Legal search","Case lookup"), ("Citation tools","References")],
    "Hospital Management": [("EMR/HIS","Records"), ("Power BI","Operations dashboards")],
    "Digital Marketing": [("Google Analytics","Metrics"), ("Meta Ads","Ads optimization")]
}

RESOURCES = {
    "AI":["fast.ai course","DeepLearning.AI nanodegree"],
    "ML":["Hands-On ML book","Scikit-learn docs"],
    "Business Analytics":["Power BI guide","Kaggle BA datasets"],
    "Robotics":["ROS tutorials","Gazebo docs"],
    "Data Analytics":["SQL tutorials","Pandas docs"]
}

MOTIVATION = [
    "Small progress each day adds up to big results.",
    "Consistency > intensity — show up daily.",
    "Practice is how expertise is built.",
    "Mistakes are proof that you are trying."
]

# ---------------- Utility functions ----------------
def img_to_b64(file):
    if not file:
        return None
    data = file.read()
    return base64.b64encode(data).decode('utf-8')

def call_ai_chat(messages, api_key, base_url="https://api.deepseek.com/v1", model="deepseek-chat"):
    if not api_key:
        return None, "API key missing. Paste your DeepSeek/OpenRouter API key in sidebar to enable live AI."
    url = base_url.rstrip('/') + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code != 200:
            return None, f"API error {r.status_code}: {r.text[:300]}"
        data = r.json()
        # safe access
        txt = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return txt, None
    except Exception as e:
        return None, f"Request failed: {e}"

def generate_daily_quiz(program):
    # Lightweight quiz generation using static pools + program specific tweak
    # We'll create 20 MCQs by sampling and customizing first few for program
    base_mcq = [
        ("Which library is commonly used for data analysis in Python?", ["pandas","NumPy","Matplotlib","Flask"], "pandas"),
        ("Which tool is popular for BI dashboards?", ["Power BI","Git","Linux","Docker"], "Power BI"),
        ("What does ETL stand for?", ["Extract Transform Load","Enter Test Leave","Edit Transfer Log","None"], "Extract Transform Load"),
        ("Which is a supervised learning algorithm?", ["K-Means","Linear Regression","DBSCAN","PCA"], "Linear Regression"),
        ("Which library is used for computer vision tasks?", ["OpenCV","pandas","Flask","Requests"], "OpenCV"),
        ("SLAM stands for?", ["Simultaneous Localization and Mapping","Single Loc And Map","Source Local Area Map","None"], "Simultaneous Localization and Mapping"),
        ("Which cloud provider is common?", ["AWS","Pandas","NumPy","Scikit"], "AWS"),
        ("Which file format is common for data?", ["CSV","PNG","MP3","EXE"], "CSV"),
        ("Which is used for experiment tracking?", ["Weights & Biases","VSCode","Excel","PowerPoint"], "Weights & Biases"),
        ("Which is a deep learning framework?", ["TensorFlow","Excel","PowerPoint","Word"], "TensorFlow")
    ]
    # Program-specific seed Qs (first two)
    program_qs = {
        "AI": [("Which model family is from OpenAI?", ["GPT-4","BERT","ResNet","AlexNet"], "GPT-4")],
        "ML": [("Which algorithm is best for classification?", ["Linear Regression","Logistic Regression","PCA","KNN"], "Logistic Regression")],
        "Business Analytics": [("Which metric is KPI?", ["Key Performance Indicator","Key Program Interface","Kernel Process Input","None"], "Key Performance Indicator")],
        "Data Analytics": [("Which SQL clause filters rows?", ["WHERE","GROUP BY","ORDER BY","HAVING"], "WHERE")],
        "Robotics": [("Which sensor measures distance?", ["Lidar","Microphone","Thermometer","GPS"], "Lidar")],
        "Digital Marketing":[("What is SEO?", ["Search Engine Optimization","Simple Email Output","Software Engineering Option","None"], "Search Engine Optimization")]
    }
    qs = []
    # add program-specific if available
    if program in program_qs:
        qs.extend(program_qs[program])
    # fill from base_mcq shuffled
    shuffled = base_mcq.copy()
    random.shuffle(shuffled)
    for q in shuffled:
        if len(qs) >= 20: break
        qs.append(q)
    # ensure length 20 (repeat if necessary)
    while len(qs) < 20:
        qs.append(random.choice(base_mcq))
    return qs[:20]

# ---------------- Sidebar (API key, profile quick) ----------------
with st.sidebar:
    st.markdown("## 🔧 Settings & API")
    api_key_input = st.text_input("Paste DeepSeek/OpenRouter API Key (optional)", type="password")
    base_url_input = st.text_input("Base URL", value=os.getenv("AI_BASE_URL","https://api.deepseek.com/v1"),
                                  help="DeepSeek default: https://api.deepseek.com/v1. Or use OpenRouter base if you have that.")
    st.markdown("---")
    st.markdown("### 👤 Quick login (demo)")
    demo_user = st.selectbox("Choose demo user", options=["neel","soumy","vivek","student","new"])
    if demo_user != "new":
        # demo credential note
        st.caption("Demo creds: username = demo user, password = demo user (or check mentor list).")
    st.markdown("---")
    st.markdown("📧 Helpline: dubevivek50@gmail.com")
    st.markdown("Tip: For live AI put your key above. If missing, app gives offline helpful tips.")

# ---------------- Login Screen ----------------
def show_login():
    st.markdown('<div class="header-card"><h1 class="h1">🎓 Pro AI Learning Platform</h1></div>', unsafe_allow_html=True)
    st.markdown('<div class="card"><h3>🔐 Student Login</h3>', unsafe_allow_html=True)
    col1, col2 = st.columns([2,1])
    with col1:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
    with col2:
        if st.button("Login"):
            u = username.strip().lower()
            p = password.strip()
            # demo simple auth (mentor provided list)
            demo_accounts = {"neel":"1234","soumy":"1111","vivek":"2222","student":"student"}
            if u in demo_accounts and demo_accounts[u] == p:
                st.session_state.logged_in = True
                st.session_state.username = u
                st.success(f"Welcome {u.capitalize()}! 🚀")
                time.sleep(0.6)
                return
            else:
                st.error("Incorrect username or password. Use demo accounts or Save profile then proceed.")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Main Dashboard / App UI ----------------
def show_dashboard():
    st.markdown('<div class="header-card"><h2 class="h2">Welcome back — Learn with AI, quizzes & projects</h2></div>', unsafe_allow_html=True)
    # Top KPIs
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown('<div class="kpi"><h3>Daily Quiz</h3><div class="small">20 Q MCQ</div></div>', unsafe_allow_html=True)
    with k2:
        total_pts = sum(item.get("total",0) for item in st.session_state.scores) if st.session_state.scores else 0
        st.markdown(f'<div class="kpi"><h3>{total_pts}</h3><div class="small">Total Points</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown('<div class="kpi"><h3>AI Tutor</h3><div class="small">Ask Doubts</div></div>', unsafe_allow_html=True)
    with k4:
        st.markdown('<div class="kpi"><h3>Motivation</h3><div class="small">Daily Quote</div></div>', unsafe_allow_html=True)

    # Tabs
    tab_dashboard, tab_quiz, tab_practice, tab_ai, tab_tools, tab_resources, tab_leader = st.tabs(
        ["🏠 Home", "📝 Daily Quiz", "🖋 Practice", "🤖 AI Tutor", "🧰 AI Tools", "📚 Resources", "🏆 Leaderboard"]
    )

    # ---------- HOME ----------
    with tab_dashboard:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        col_left, col_right = st.columns([2,1])
        with col_left:
            st.subheader(f"Hello, {st.session_state.username.capitalize() if st.session_state.username else 'Student'}")
            st.write("Use the tabs to practice, take daily quizzes, ask AI tutor, and check resources.")
            st.write("Tip: Complete daily quiz to earn XP & badges.")
            st.markdown("### 🔔 Today's suggestion")
            st.info(random.choice(MOTIVATION))
        with col_right:
            # profile summary
            prof = st.session_state.profile
            if prof.get("photo"):
                st.image(BytesIO(base64.b64decode(prof["photo"])), width=140, output_format="auto")
            else:
                st.image("https://dummyimage.com/140x140/223/77a6ff&text=Profile", width=140)
            st.markdown(f"**{prof.get('name', st.session_state.username or '—')}**")
            st.markdown(f"*{prof.get('program','—')} • {prof.get('year','—')}*")
            st.markdown(f"<div class='small'>Fav food: {prof.get('fav_food','—')}</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------- DAILY QUIZ ----------
    with tab_quiz:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📝 Daily 20 MCQ Quiz")
        prog = st.selectbox("Select Program", PROGRAMS, index=0)
        # ensure we store per-day quiz to prevent resubmit (demo local logic)
        today_key = f"{prog}|{datetime.now().strftime('%Y-%m-%d')}"
        if 'quiz_store' not in st.session_state:
            st.session_state.quiz_store = {}
        if today_key not in st.session_state.quiz_store:
            qs = generate_daily_quiz(prog)
            # Save structure: list of tuples (q,opts,correct)
            st.session_state.quiz_store[today_key] = qs
            st.session_state[f"answers_{today_key}"] = {}
        qs = st.session_state.quiz_store[today_key]

        # Render MCQs
        answers_local = {}
        for i, (q, opts, correct) in enumerate(qs, start=1):
            st.markdown(f"<div class='question-card'><b>{i}. {q}</b></div>", unsafe_allow_html=True)
            choice = st.radio("", options=opts, key=f"{today_key}_q{i}")
            answers_local[f"q{i}"] = choice

        if st.button("Submit Quiz"):
            # compute score
            score = 0
            for i, (q, opts, correct) in enumerate(qs, start=1):
                chosen = st.session_state.get(f"{today_key}_q{i}")
                if chosen == correct:
                    score += 5  # 5 points per correct MCQ
            st.success(f"Score: {score} / {len(qs)*5}")
            st.session_state.scores.append({"date": datetime.now().strftime("%Y-%m-%d"),
                                           "program": prog, "total": score})
            st.balloons()
            # award badge if full marks
            if score == len(qs)*5:
                st.success("Perfect! You earned a Platinum Badge 🏆")

        st.markdown('</div>', unsafe_allow_html=True)

    # ---------- PRACTICE ----------
    with tab_practice:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🖋 Practice Exercises")
        prog_p = st.selectbox("Choose program for practice", PROGRAMS, index=0, key="practice_prog")
        practice_bank = {
            "AI":[ "Explain difference between AI & ML (2 lines).", "List 3 AI applications."],
            "ML":[ "Write steps to split dataset for train/val/test.", "Explain bias vs variance."],
            "Robotics":[ "Describe PID controller in 2 lines.", "What is SLAM?" ],
            "Business Analytics":[ "List five KPIs for an e-commerce store.", "Sketch a dashboard layout for sales."],
            "Data Analytics":[ "Write a SQL query to get top 5 customers by revenue.", "Explain ETL pipeline." ]
        }
        items = practice_bank.get(prog_p, ["Write one short note on your topic."])
        for p_q in items:
            st.markdown(f"- {p_q}")
        st.markdown("You can write answers below and ask AI for feedback.")
        ans = st.text_area("Write your practice answer (2-5 lines)")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save Practice Locally"):
                st.success("Saved locally for this session.")
        with col2:
            if st.button("Ask AI to Review Answer"):
                if not api_key_input:
                    st.info("No API key. AI offline mode: quick tips -> Keep it concise, include examples.")
                else:
                    prompts = [
                        {"role":"system","content":"You are a friendly tutor who grades short answers 0-10 and gives 2 improvements."},
                        {"role":"user","content": f"Question: {items[0]}\nAnswer: {ans}"}
                    ]
                    with st.spinner("Getting AI feedback..."):
                        txt, err = call_ai_chat(prompts, api_key_input, base_url_input)
                        if err:
                            st.error(err)
                        else:
                            st.write(txt)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------- AI TUTOR ----------
    with tab_ai:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🤖 AI Tutor — Ask any study question")
        prog_ai = st.selectbox("Program context (helps AI tailor)", PROGRAMS, index=0, key="ai_prog")
        level_ai = st.selectbox("Student Level", ["Beginner","Intermediate","Advanced"], index=0)
        user_q = st.text_area("Type your question (be specific for best results)", height=120)
        if st.button("Ask AI"):
            if not user_q.strip():
                st.warning("Please write a question first.")
            else:
                system_prompt = (f"You are an expert tutor in {prog_ai}. Answer for a {level_ai} student. "
                                 "Give: 1) short explanation, 2) one example, 3) small code snippet if helpful, "
                                 "4) two study resources. Keep it concise.")
                msgs = [{"role":"system","content":system_prompt},
                        {"role":"user","content":user_q}]
                with st.spinner("AI is thinking..."):
                    txt, err = call_ai_chat(msgs, api_key_input, base_url_input)
                    if err:
                        # fallback offline helpful answer
                        st.error(err)
                        st.info("Offline tip: Break the topic into definitions, steps, and one example.")
                    else:
                        st.markdown("**AI Answer:**")
                        st.write(txt)
                        # save history
                        st.session_state.chat_history.append({"q":user_q,"a":txt,"ts":datetime.now().isoformat()})
        # show recent history
        if st.session_state.chat_history:
            st.markdown("**Recent Questions**")
            for item in st.session_state.chat_history[-6:]:
                st.markdown(f"- **Q:** {item['q']}  \n  **A:** {item['a'][:500]}...")

        st.markdown('</div>', unsafe_allow_html=True)

    # ---------- AI TOOLS ----------
    with tab_tools:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🧰 AI Tools by Program")
        prog_tool = st.selectbox("Select Program", PROGRAMS, index=0, key="tools_prog")
        st.markdown("**Recommended tools & why**")
        for name, why in AI_TOOLS.get(prog_tool, []):
            st.markdown(f"- **{name}** — {why}")
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------- RESOURCES ----------
    with tab_resources:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📚 Resources")
        p = st.selectbox("Choose program", PROGRAMS, index=0, key="res_prog")
        for r in RESOURCES.get(p, ["No resources yet."]):
            st.markdown(f"- {r}")
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------- LEADERBOARD ----------
    with tab_leader:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🏆 Leaderboard (demo)")
        # demo leaderboard: combine demo names + session points
        demo_names = ["Soumya","Vivek","Satyam","Alyssa","Rohit","Arman"]
        demo_pts = list(np.random.randint(200,800,len(demo_names)))
        # add current user points
        my_pts = sum(s.get("total",0) for s in st.session_state.scores)
        demo_names.append(st.session_state.username.capitalize() if st.session_state.username else "You")
        demo_pts.append(my_pts)
        df = pd.DataFrame({"Name": demo_names, "Points": demo_pts}).sort_values("Points", ascending=False).reset_index(drop=True)
        st.dataframe(df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Profile Editor ----------------
def show_profile_editor():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧑‍🎓 Edit Profile")
    name = st.text_input("Full name", value=st.session_state.profile.get("name",""))
    program = st.selectbox("Program", PROGRAMS, index=PROGRAMS.index(st.session_state.profile.get("program", PROGRAMS[0])) if st.session_state.profile.get("program") in PROGRAMS else 0)
    year = st.selectbox("Year", ["1st","2nd","3rd","4th","Other"], index=0)
    fav_song = st.text_input("Favorite song", value=st.session_state.profile.get("fav_song",""))
    fav_food = st.text_input("Favorite food", value=st.session_state.profile.get("fav_food",""))
    interests = st.text_area("Interests (comma separated)", value=st.session_state.profile.get("interests",""))
    photo = st.file_uploader("Upload photo (png/jpg)", type=["png","jpg","jpeg"])
    if photo:
        b64 = base64.b64encode(photo.read()).decode('utf-8')
        st.session_state.profile["photo"] = b64
    if st.button("Save Profile"):
        st.session_state.profile.update({
            "name": name, "program": program, "year": year, "fav_song": fav_song,
            "fav_food": fav_food, "interests": interests
        })
        st.success("Profile saved ✅")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Entry point ----------------
st.sidebar.markdown("## Navigation")
if not st.session_state.logged_in:
    show_login()
    st.sidebar.markdown("---")
    st.sidebar.info("Demo users: neel/1234, soumy/1111, vivek/2222, student/student")
else:
    # allow editing profile via sidebar quick link
    if st.sidebar.button("Edit Profile"):
        show_profile_editor()
    if st.sidebar.button("Home / Dashboard"):
        show_dashboard()
    else:
        show_dashboard()
    st.sidebar.markdown("---")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.experimental_rerun()

# footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div class='small'>Pro AI Learning Platform — built for hackathons. Add real DB and hosting later for persistence.</div>", unsafe_allow_html=True)
