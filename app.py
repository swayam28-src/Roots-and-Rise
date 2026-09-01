import gradio as gr
import pandas as pd
import sqlite3
import re
import os
from datetime import datetime

import plotly.express as px

from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN


# =========================================================
# ROOTS & RISE
# SIH 2026
# ADVANCED AI CITIZEN GRIEVANCE PLATFORM
# =========================================================


DB_NAME = "roots_and_rise.db"


# =========================================================
# LOAD AI EMBEDDING MODEL
# =========================================================

print("🧠 Loading AI Embedding Model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("✅ AI Model Loaded Successfully!")


# =========================================================
# DATABASE
# =========================================================

def get_connection():

    return sqlite3.connect(
        DB_NAME,
        check_same_thread=False
    )


def initialize_database():

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute("""

        CREATE TABLE IF NOT EXISTS complaints (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            name TEXT,

            email TEXT,

            complaint TEXT,

            location TEXT,

            latitude REAL,

            longitude REAL,

            category TEXT,

            department TEXT,

            priority TEXT,

            duplicate_of TEXT,

            similarity REAL,

            status TEXT,

            created_at TEXT

        )

    """)

    conn.commit()

    conn.close()


initialize_database()


# =========================================================
# LANGUAGE NORMALIZATION
# =========================================================

def normalize_text(text):

    if not text:

        return ""

    text = text.lower().strip()


    replacements = {

        # English abbreviations

        "pls": "please",

        "plz": "please",

        "govt": "government",

        "rd": "road",


        # Hinglish

        "pani nahi aa raha": "no water supply",

        "pani nahi hai": "water shortage",

        "road kharab hai": "road damaged",

        "sadak kharab hai": "road damaged",

        "bijli nahi hai": "electricity problem",

        "light nahi hai": "power cut",

        "kachra": "garbage",

        "ganda": "dirty",

        "naali": "drain",

        "bahut dangerous": "very dangerous",

        "jaldi": "urgent",

        "madad chahiye": "emergency help"

    }


    for old, new in replacements.items():

        text = text.replace(
            old,
            new
        )


    text = re.sub(
        r"\s+",
        " ",
        text
    )


    return text.strip()


# =========================================================
# CATEGORY KNOWLEDGE BASE
# =========================================================

CATEGORY_EXAMPLES = {

    "Road & Infrastructure": [

        "There is a pothole on the road.",

        "The road is damaged and unsafe.",

        "The bridge requires urgent repair.",

        "The footpath is broken.",

        "Road construction is incomplete."

    ],


    "Water Supply": [

        "There is no drinking water supply.",

        "Water shortage in our locality.",

        "The water pipeline is damaged.",

        "No water is coming from the tap.",

        "Residents are facing a water supply issue."

    ],


    "Sanitation & Garbage": [

        "Garbage has not been collected.",

        "Waste is overflowing on the street.",

        "The area is dirty and unhygienic.",

        "Drainage and sewage are overflowing.",

        "Cleaning services are not working."

    ],


    "Electricity": [

        "There is a power cut.",

        "Electricity is not available.",

        "Street lights are not working.",

        "The transformer is damaged.",

        "There is a dangerous electric wire."

    ],


    "Public Safety": [

        "This area is dangerous and unsafe.",

        "There is a risk of accident.",

        "An emergency situation requires help.",

        "There is a serious public safety risk.",

        "People may get injured."

    ],


    "Health & Public Services": [

        "The hospital does not have medicine.",

        "Healthcare services are unavailable.",

        "The clinic requires more doctors.",

        "An ambulance is required.",

        "There is a public health issue."

    ]

}


DEPARTMENT_MAPPING = {

    "Road & Infrastructure":
        "Public Works Department",

    "Water Supply":
        "Water Supply Department",

    "Sanitation & Garbage":
        "Municipal Sanitation Department",

    "Electricity":
        "Electricity Department",

    "Public Safety":
        "Police & Emergency Services",

    "Health & Public Services":
        "Public Health Department",

    "General Public Grievance":
        "Citizen Grievance Department"

}


# =========================================================
# CREATE CATEGORY EMBEDDINGS
# =========================================================

print("📚 Preparing AI Category Knowledge Base...")

CATEGORY_TEXTS = []

CATEGORY_LABELS = []


for category, examples in CATEGORY_EXAMPLES.items():

    for example in examples:

        CATEGORY_TEXTS.append(
            example
        )

        CATEGORY_LABELS.append(
            category
        )


CATEGORY_EMBEDDINGS = embedding_model.encode(

    CATEGORY_TEXTS,

    normalize_embeddings=True

)


print("✅ Category Knowledge Base Ready!")


# =========================================================
# AI CLASSIFICATION WITH CONFIDENCE
# =========================================================

def classify_complaint_ai(text):

    normalized_text = normalize_text(
        text
    )


    complaint_embedding = embedding_model.encode(

        [normalized_text],

        normalize_embeddings=True

    )[0]


    similarity_scores = (

        CATEGORY_EMBEDDINGS
        @ complaint_embedding

    )


    category_scores = {}


    for category in CATEGORY_EXAMPLES:

        category_indices = [

            i

            for i, label in enumerate(
                CATEGORY_LABELS
            )

            if label == category

        ]


        category_scores[category] = max(

            float(
                similarity_scores[i]
            )

            for i in category_indices

        )


    best_category = max(

        category_scores,

        key=category_scores.get

    )


    confidence = category_scores[
        best_category
    ]


    confidence_percentage = round(

        max(
            0,
            confidence
        ) * 100,

        2

    )


    # If confidence is very low

    if confidence < 0.25:

        best_category = (
            "General Public Grievance"
        )

        confidence_percentage = round(
            confidence_percentage,
            2
        )


    department = DEPARTMENT_MAPPING.get(

        best_category,

        "Citizen Grievance Department"

    )


    return (

        best_category,

        department,

        confidence_percentage

    )


# =========================================================
# PRIORITY SCORING
# =========================================================

HIGH_PRIORITY_WORDS = [

    "accident",

    "emergency",

    "danger",

    "dangerous",

    "fire",

    "injury",

    "life threatening",

    "serious",

    "urgent",

    "critical",

    "exposed wire"

]


MEDIUM_PRIORITY_WORDS = [

    "problem",

    "not working",

    "overflow",

    "several days",

    "major",

    "blocked",

    "shortage",

    "damaged"

]


def calculate_priority(text):

    text = normalize_text(
        text
    )


    high_score = sum(

        1

        for word in HIGH_PRIORITY_WORDS

        if word in text

    )


    medium_score = sum(

        1

        for word in MEDIUM_PRIORITY_WORDS

        if word in text

    )


    if high_score >= 1:

        return "High"


    elif medium_score >= 1:

        return "Medium"


    return "Low"


# =========================================================
# GET ALL COMPLAINTS
# =========================================================

def get_all_complaints():

    conn = get_connection()


    df = pd.read_sql_query(

        "SELECT * FROM complaints",

        conn

    )


    conn.close()


    return df


# =========================================================
# SEMANTIC DUPLICATE DETECTION
# =========================================================

def detect_semantic_duplicate(new_complaint):

    df = get_all_complaints()


    if df.empty:

        return (

            "No Duplicate",

            0.0

        )


    existing_complaints = (

        df["complaint"]

        .fillna("")

        .tolist()

    )


    normalized_existing = [

        normalize_text(
            complaint
        )

        for complaint in existing_complaints

    ]


    normalized_new = normalize_text(

        new_complaint

    )


    all_texts = (

        normalized_existing

        + [

            normalized_new

        ]

    )


    embeddings = embedding_model.encode(

        all_texts,

        normalize_embeddings=True

    )


    new_embedding = embeddings[-1]

    existing_embeddings = embeddings[:-1]


    similarity_scores = (

        existing_embeddings

        @ new_embedding

    )


    max_index = int(

        similarity_scores.argmax()

    )


    max_similarity = float(

        similarity_scores[max_index]

    )


    similarity_percentage = round(

        max_similarity * 100,

        2

    )


    if max_similarity >= 0.72:

        duplicate_id = int(

            df.iloc[max_index]["id"]

        )


        return (

            f"Possible Semantic Duplicate of Complaint #{duplicate_id}",

            similarity_percentage

        )


    return (

        "No Duplicate",

        similarity_percentage

    )


# =========================================================
# HOTSPOT DETECTION
# =========================================================

def detect_hotspots(df):

    if df.empty:

        return df


    valid_df = df.dropna(

        subset=[
            "latitude",
            "longitude"
        ]

    ).copy()


    if len(valid_df) < 2:

        valid_df["hotspot"] = "No Cluster"

        return valid_df


    coordinates = valid_df[

        [
            "latitude",
            "longitude"
        ]

    ].values


    clustering = DBSCAN(

        eps=0.01,

        min_samples=2

    ).fit(

        coordinates

    )


    valid_df["cluster_id"] = (

        clustering.labels_

    )


    valid_df["hotspot"] = (

        valid_df["cluster_id"]

        .apply(

            lambda x:

            "Hotspot"

            if x != -1

            else "No Cluster"

        )

    )


    return valid_df


# =========================================================
# AI INSIGHTS
# =========================================================

def generate_ai_insights(df):

    if df.empty:

        return """

### 🤖 AI Insights

No complaint data is currently available.

"""


    insights = []


    high_count = len(

        df[
            df["priority"] == "High"
        ]

    )


    if high_count > 0:

        insights.append(

            f"🔴 **{high_count} high-priority grievance(s) require immediate attention.**"

        )


    duplicate_count = len(

        df[

            df["duplicate_of"]

            != "No Duplicate"

        ]

    )


    if duplicate_count > 0:

        insights.append(

            f"🔁 **{duplicate_count} potential duplicate complaint(s) detected, reducing repeated processing.**"

        )


    top_category = (

        df["category"]

        .value_counts()

        .idxmax()

    )


    top_category_count = (

        df["category"]

        .value_counts()

        .max()

    )


    insights.append(

        f"📊 **Most reported issue: {top_category} ({top_category_count} complaints).**"

    )


    hotspot_df = detect_hotspots(
        df
    )


    if (

        not hotspot_df.empty

        and

        "hotspot"

        in hotspot_df.columns

    ):


        hotspots = hotspot_df[

            hotspot_df["hotspot"]

            == "Hotspot"

        ]


        if not hotspots.empty:


            hotspot_locations = (

                hotspots["location"]

                .value_counts()

                .head(3)

                .index

                .tolist()

            )


            insights.append(

                "🔥 **Potential geographic hotspot(s): "
                + ", ".join(
                    hotspot_locations
                )
                + ".**"

            )


    if not insights:

        insights.append(

            "🟢 **Complaint patterns currently appear manageable.**"

        )


    return """

# 🤖 AI GOVERNANCE INSIGHTS

""" + "\n\n".join(
        insights
    )


# =========================================================
# SUBMIT COMPLAINT
# =========================================================

def submit_complaint(

    name,

    email,

    complaint,

    location,

    latitude,

    longitude

):


    if (

        not name

        or not complaint

        or not location

    ):


        return """

## ❌ Required Information Missing

Please provide:

- Full Name
- Complaint Description
- Location

"""


    normalized_complaint = normalize_text(

        complaint

    )


    (

        category,

        department,

        confidence

    ) = classify_complaint_ai(

        normalized_complaint

    )


    priority = calculate_priority(

        normalized_complaint

    )


    (

        duplicate_of,

        similarity

    ) = detect_semantic_duplicate(

        normalized_complaint

    )


    created_at = datetime.now().strftime(

        "%d-%m-%Y %H:%M:%S"

    )


    conn = get_connection()

    cursor = conn.cursor()


    cursor.execute("""

        INSERT INTO complaints (

            name,

            email,

            complaint,

            location,

            latitude,

            longitude,

            category,

            department,

            priority,

            duplicate_of,

            similarity,

            status,

            created_at

        )

        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

    """, (

        name,

        email,

        complaint,

        location,

        latitude,

        longitude,

        category,

        department,

        priority,

        duplicate_of,

        similarity,

        "Submitted",

        created_at

    ))


    complaint_id = cursor.lastrowid


    conn.commit()

    conn.close()


    priority_icon = {

        "High": "🔴",

        "Medium": "🟠",

        "Low": "🟢"

    }.get(

        priority,

        "⚪"

    )


    return f"""

# 🎉 Complaint Submitted Successfully

## Complaint Reference: **#{complaint_id}**

---

# 🤖 ADVANCED AI ANALYSIS

### 🧠 AI Classification

**{category}**

### 🎯 AI Confidence Score

**{confidence}%**

### 🏛️ Smart Department Routing

**{department}**

### {priority_icon} Priority Intelligence

**{priority}**

### 🔁 Semantic Duplicate Detection

**{duplicate_of}**

### 📊 Semantic Similarity Score

**{similarity}%**

---

## 📍 Reported Location

**{location}**

### 🔄 Current Status

**Submitted**

> Your grievance was analyzed using
> **AI semantic embeddings, intelligent classification,
> priority scoring and duplicate detection.**

"""


# =========================================================
# TRACK COMPLAINT
# =========================================================

def track_complaint(complaint_id):


    if complaint_id is None:

        return (

            "❌ Please enter a Complaint ID."

        )


    try:

        complaint_id = int(
            complaint_id
        )


    except (

        ValueError,

        TypeError

    ):

        return (

            "❌ Invalid Complaint ID."

        )


    conn = get_connection()

    cursor = conn.cursor()


    cursor.execute("""

        SELECT

            id,

            category,

            department,

            priority,

            status,

            location,

            created_at

        FROM complaints

        WHERE id = ?

    """, (

        complaint_id,

    ))


    result = cursor.fetchone()


    conn.close()


    if not result:

        return (

            "❌ Complaint ID not found."

        )


    return f"""

# 🔎 Complaint Status

## Reference: **#{result[0]}**

| Information | Current Details |
|---|---|
| 🧠 AI Category | **{result[1]}** |
| 🏛️ Department | **{result[2]}** |
| ⚡ Priority | **{result[3]}** |
| 🔄 Status | **{result[4]}** |
| 📍 Location | **{result[5]}** |
| 🕒 Submitted | **{result[6]}** |

"""


# =========================================================
# DASHBOARD
# =========================================================

def generate_dashboard():

    df = get_all_complaints()


    if df.empty:


        empty_chart = px.bar(

            title="No Complaint Data Available"

        )


        return (

            "## 📊 No complaints submitted yet.",

            "### 🤖 No AI insights available yet.",

            pd.DataFrame(),

            empty_chart,

            empty_chart,

            empty_chart

        )


    total = len(df)


    high = len(

        df[
            df["priority"]
            == "High"
        ]

    )


    medium = len(

        df[
            df["priority"]
            == "Medium"
        ]

    )


    low = len(

        df[
            df["priority"]
            == "Low"
        ]

    )


    resolved = len(

        df[
            df["status"]
            == "Resolved"
        ]

    )


    duplicate_count = len(

        df[

            df["duplicate_of"]

            != "No Duplicate"

        ]

    )


    stats = f"""

# 🏛️ ROOTS & RISE AI INTELLIGENCE CENTER

## Real-Time Governance Analytics

| Intelligence Metric | Value |
|---|---:|
| 📢 Total Grievances | **{total}** |
| 🔴 High Priority | **{high}** |
| 🟠 Medium Priority | **{medium}** |
| 🟢 Low Priority | **{low}** |
| 🔁 Semantic Duplicates | **{duplicate_count}** |
| ✅ Resolved | **{resolved}** |

"""


    ai_insights = generate_ai_insights(
        df
    )


    category_counts = (

        df["category"]

        .value_counts()

        .reset_index()

    )


    category_counts.columns = [

        "Category",

        "Complaints"

    ]


    category_chart = px.bar(

        category_counts,

        x="Category",

        y="Complaints",

        title="🧠 AI Complaint Classification Analytics"

    )


    priority_counts = (

        df["priority"]

        .value_counts()

        .reset_index()

    )


    priority_counts.columns = [

        "Priority",

        "Complaints"

    ]


    priority_chart = px.pie(

        priority_counts,

        names="Priority",

        values="Complaints",

        title="⚡ Priority Intelligence Distribution"

    )


    hotspot_df = detect_hotspots(
        df
    )


    if hotspot_df.empty:


        map_chart = px.scatter_geo(

            title="📍 No Geographic Data Available"

        )


    else:


        map_chart = px.scatter_geo(

            hotspot_df,

            lat="latitude",

            lon="longitude",

            hover_name="location",

            hover_data=[

                "category",

                "priority",

                "department",

                "status",

                "hotspot"

            ],

            symbol="hotspot",

            title="🔥 AI-Detected Geographic Hotspots"

        )


        map_chart.update_geos(

            projection_type="natural earth",

            showcountries=True,

            showland=True

        )


    display_df = df.rename(columns={

        "id":
            "ID",

        "name":
            "Citizen",

        "complaint":
            "Complaint",

        "location":
            "Location",

        "category":
            "AI Category",

        "department":
            "Department",

        "priority":
            "Priority",

        "duplicate_of":
            "Duplicate Detection",

        "similarity":
            "Similarity %",

        "status":
            "Status",

        "created_at":
            "Submitted At"

    })


    columns_to_show = [

        "ID",

        "Citizen",

        "Complaint",

        "Location",

        "AI Category",

        "Department",

        "Priority",

        "Duplicate Detection",

        "Similarity %",

        "Status",

        "Submitted At"

    ]


    display_df = display_df[

        [

            col

            for col in columns_to_show

            if col in display_df.columns

        ]

    ]


    return (

        stats,

        ai_insights,

        display_df,

        category_chart,

        priority_chart,

        map_chart

    )


# =========================================================
# UPDATE COMPLAINT STATUS
# =========================================================

def update_status(

    complaint_id,

    new_status

):


    if complaint_id is None:

        return (

            "❌ Please enter a Complaint ID."

        )


    try:

        complaint_id = int(
            complaint_id
        )


    except (

        ValueError,

        TypeError

    ):

        return (

            "❌ Invalid Complaint ID."

        )


    conn = get_connection()

    cursor = conn.cursor()


    cursor.execute("""

        UPDATE complaints

        SET status = ?

        WHERE id = ?

    """, (

        new_status,

        complaint_id

    ))


    conn.commit()


    updated = cursor.rowcount


    conn.close()


    if updated == 0:

        return (

            "❌ Complaint ID not found."

        )


    return f"""

# ✅ Status Updated Successfully

### Complaint ID: **#{complaint_id}**

### New Status: **{new_status}**

Refresh the dashboard to view updated analytics.

"""


# =========================================================
# USER INTERFACE
# =========================================================

with gr.Blocks(

    title="ROOTS & RISE | Advanced AI | SIH 2026"

) as app:


    gr.Markdown("""

# 🌱 ROOTS & RISE

## AI-Powered Citizen Grievance Intelligence Platform

### *From Citizen Voices to Smarter Governance*

**Smart India Hackathon 2026 | Advanced AI Prototype**

---

### 🧠 Semantic AI Classification

### 🎯 Confidence Scoring

### 🔁 Embedding-Based Duplicate Detection

### 🔥 Geographic Hotspot Detection

### 🏛️ Intelligent Department Routing

""")


    with gr.Tabs():


        # =============================================
        # HOME
        # =============================================

        with gr.Tab("🏠 Home"):


            gr.Markdown("""

# 🚀 ROOTS & RISE

## Transforming Citizen Complaints into Governance Intelligence

### 👤 Citizen Submission

Citizens submit grievances in natural language.

⬇️

### 🌐 Language Normalization

Common abbreviations and Hinglish phrases are normalized.

⬇️

### 🧠 AI Semantic Classification

Sentence embeddings identify the meaning of the complaint.

⬇️

### 🎯 AI Confidence Scoring

The system calculates classification confidence.

⬇️

### ⚡ Priority Intelligence

Urgent and high-risk grievances are prioritized.

⬇️

### 🔁 Semantic Duplicate Detection

Embedding similarity identifies complaints describing the same issue.

⬇️

### 🔥 Geographic Hotspot Detection

AI clustering identifies areas with concentrated grievances.

⬇️

### 🏛️ Authority Intelligence Dashboard

Authorities receive structured and actionable insights.

""")


        # =============================================
        # CITIZEN PORTAL
        # =============================================

        with gr.Tab("👤 Citizen Portal"):


            gr.Markdown("""

# 👤 Submit a Citizen Grievance

Our AI system will automatically analyze,
classify and route your complaint.

""")


            with gr.Row():


                with gr.Column():


                    citizen_name = gr.Textbox(

                        label="👤 Full Name *",

                        placeholder=
                        "Enter your full name"

                    )


                    citizen_email = gr.Textbox(

                        label="📧 Email Address",

                        placeholder=
                        "example@email.com"

                    )


                    complaint_text = gr.Textbox(

                        label=
                        "📝 Describe Your Grievance *",

                        placeholder="""
Example:
There is a large pothole near the railway station.
The road is dangerous and accidents may happen.
Please repair it urgently.
""",

                        lines=8

                    )


                    citizen_location = gr.Textbox(

                        label="📍 Location *",

                        placeholder=
                        "Example: Thane Railway Station"

                    )


                with gr.Column():


                    gr.Markdown("""

# 🤖 Advanced AI Processing

### 🧠 Semantic Classification

Understands complaint meaning.

### 🎯 Confidence Score

Measures AI classification confidence.

### ⚡ Priority Intelligence

Identifies urgent complaints.

### 🔁 Duplicate Detection

Compares semantic embeddings.

### 🏛️ Smart Routing

Routes complaints to departments.

### 🔥 Hotspot Intelligence

Uses geographic clustering.

""")


                    latitude = gr.Number(

                        label=
                        "Latitude (Optional)",

                        value=None

                    )


                    longitude = gr.Number(

                        label=
                        "Longitude (Optional)",

                        value=None

                    )


                    submit_button = gr.Button(

                        "🚀 Analyze with Advanced AI",

                        variant="primary",

                        size="lg"

                    )


            submission_result = gr.Markdown()


            submit_button.click(

                fn=submit_complaint,

                inputs=[

                    citizen_name,

                    citizen_email,

                    complaint_text,

                    citizen_location,

                    latitude,

                    longitude

                ],

                outputs=
                submission_result

            )


        # =============================================
        # TRACK COMPLAINT
        # =============================================

        with gr.Tab("🔎 Track Complaint"):


            gr.Markdown("""

# 🔎 Track Your Grievance

Enter your Complaint Reference ID.

""")


            with gr.Row():


                tracking_id = gr.Number(

                    label=
                    "Complaint ID",

                    precision=0

                )


                track_button = gr.Button(

                    "🔎 Track Complaint",

                    variant="primary"

                )


            tracking_result = gr.Markdown()


            track_button.click(

                fn=
                track_complaint,

                inputs=
                tracking_id,

                outputs=
                tracking_result

            )


        # =============================================
        # AUTHORITY DASHBOARD
        # =============================================

        with gr.Tab("🏛️ Authority Dashboard"):


            gr.Markdown("""

# 🏛️ AI Governance Intelligence Center

Monitor citizen grievances using AI-driven analytics.

""")


            refresh_button = gr.Button(

                "🔄 Refresh AI Intelligence",

                variant="primary",

                size="lg"

            )


            dashboard_stats = gr.Markdown()


            ai_insights_output = gr.Markdown()


            gr.Markdown(
                "## 📋 Live Grievance Registry"
            )


            complaints_table = gr.Dataframe(

                interactive=False,

                wrap=True

            )


            gr.Markdown(
                "## 📊 AI Classification Analytics"
            )


            with gr.Row():


                category_output = gr.Plot()


                priority_output = gr.Plot()


            gr.Markdown(
                "## 🔥 Geographic Hotspot Intelligence"
            )


            map_output = gr.Plot()


            refresh_button.click(

                fn=
                generate_dashboard,

                outputs=[

                    dashboard_stats,

                    ai_insights_output,

                    complaints_table,

                    category_output,

                    priority_output,

                    map_output

                ]

            )


            gr.Markdown("---")


            gr.Markdown("""

# 🔄 Authority Action Center

Update grievance status.

""")


            with gr.Row():


                status_complaint_id = gr.Number(

                    label=
                    "Complaint ID",

                    precision=0

                )


                new_status = gr.Dropdown(

                    choices=[

                        "Submitted",

                        "Under Review",

                        "In Progress",

                        "Resolved"

                    ],

                    value=
                    "Under Review",

                    label=
                    "New Status"

                )


            status_button = gr.Button(

                "✅ Update Status",

                variant="primary"

            )


            status_result = gr.Markdown()


            status_button.click(

                fn=
                update_status,

                inputs=[

                    status_complaint_id,

                    new_status

                ],

                outputs=
                status_result

            )


        # =============================================
        # TECHNOLOGY
        # =============================================

        with gr.Tab("💡 AI Innovation"):


            gr.Markdown("""

# 💡 ROOTS & RISE — Advanced AI Innovation

## 🧠 Semantic Understanding

Traditional systems search for keywords.

ROOTS & RISE uses **Sentence Embeddings**
to understand the meaning of citizen grievances.

---

## 🔁 Semantic Duplicate Detection

Two complaints may use completely different words
while describing the same issue.

Our AI compares complaint embeddings to identify
potential duplicates.

---

## 🎯 Confidence Scoring

The system calculates how strongly a complaint
matches each grievance category.

---

## 🔥 Hotspot Detection

Geographic coordinates are processed using
**DBSCAN clustering** to identify concentrated
grievance regions.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Backend |
| Gradio | Web Interface |
| SQLite | Database |
| Sentence Transformers | Semantic AI |
| all-MiniLM-L6-v2 | Text Embeddings |
| Scikit-learn | AI & Clustering |
| DBSCAN | Hotspot Detection |
| Plotly | Data Visualization |
| Pandas | Analytics |

---

# 🌱 ROOTS & RISE

### From Citizen Voices to Smarter Governance

**Smart India Hackathon 2026 — Advanced Working Prototype**

""")


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    print("🌱 ROOTS & RISE READY!")

    port = int(os.environ.get("PORT", 7865))

    app.launch(
        server_name="0.0.0.0",
        server_port=port
    )
