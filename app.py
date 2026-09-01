import gradio as gr
import pandas as pd
import sqlite3
import os
import re
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px


# =========================================================
# ROOTS & RISE
# AI-POWERED CITIZEN GRIEVANCE PLATFORM
# =========================================================


DB_NAME = "roots_and_rise.db"


# =========================================================
# DATABASE
# =========================================================

def get_connection():
    return sqlite3.connect(DB_NAME, check_same_thread=False)


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

    text = text.lower()

    replacements = {
        "pls": "please",
        "plz": "please",
        "govt": "government",
        "rd": "road",
        "water supply": "water",
        "powercut": "power cut"
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


# =========================================================
# AI COMPLAINT CLASSIFICATION
# =========================================================

CATEGORY_KEYWORDS = {

    "Road & Infrastructure": [
        "pothole",
        "road",
        "bridge",
        "footpath",
        "street",
        "highway",
        "traffic",
        "construction",
        "drainage road"
    ],

    "Water Supply": [
        "water",
        "drinking water",
        "water supply",
        "pipeline",
        "tap",
        "no water",
        "water shortage"
    ],

    "Sanitation & Garbage": [
        "garbage",
        "waste",
        "trash",
        "dirty",
        "cleaning",
        "sewage",
        "drain",
        "overflow"
    ],

    "Electricity": [
        "electricity",
        "power",
        "power cut",
        "electric",
        "street light",
        "transformer",
        "voltage"
    ],

    "Public Safety": [
        "danger",
        "accident",
        "unsafe",
        "crime",
        "emergency",
        "fire",
        "risk"
    ],

    "Health & Public Services": [
        "hospital",
        "health",
        "doctor",
        "medicine",
        "clinic",
        "ambulance"
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
        "Public Health Department"
}


def classify_complaint(text):

    text = normalize_text(text)

    scores = {}

    for category, keywords in CATEGORY_KEYWORDS.items():

        score = 0

        for keyword in keywords:

            if keyword in text:
                score += 1

        scores[category] = score


    best_category = max(
        scores,
        key=scores.get
    )


    if scores[best_category] == 0:

        best_category = "General Public Grievance"


    department = DEPARTMENT_MAPPING.get(
        best_category,
        "Citizen Grievance Department"
    )


    return best_category, department


# =========================================================
# PRIORITY SCORING
# =========================================================

HIGH_PRIORITY_WORDS = [

    "accident",
    "emergency",
    "danger",
    "dangerous",
    "fire",
    "hospital",
    "injury",
    "life threatening",
    "serious",
    "urgent",
    "critical"
]


MEDIUM_PRIORITY_WORDS = [

    "problem",
    "not working",
    "overflow",
    "several days",
    "major",
    "blocked",
    "shortage"
]


def calculate_priority(text):

    text = normalize_text(text)

    high_score = sum(
        1 for word in HIGH_PRIORITY_WORDS
        if word in text
    )


    medium_score = sum(
        1 for word in MEDIUM_PRIORITY_WORDS
        if word in text
    )


    if high_score >= 1:
        return "High"

    elif medium_score >= 1:
        return "Medium"

    else:
        return "Low"


# =========================================================
# DUPLICATE DETECTION
# =========================================================

def get_all_complaints():

    conn = get_connection()

    df = pd.read_sql_query(
        "SELECT * FROM complaints",
        conn
    )

    conn.close()

    return df


def detect_duplicate(new_complaint):

    df = get_all_complaints()

    if df.empty:

        return "No Duplicate", 0.0


    existing_complaints = (
        df["complaint"]
        .fillna("")
        .tolist()
    )


    documents = existing_complaints + [
        new_complaint
    ]


    try:

        vectorizer = TfidfVectorizer(
            stop_words="english"
        )


        vectors = vectorizer.fit_transform(
            documents
        )


        similarity_matrix = cosine_similarity(
            vectors[-1],
            vectors[:-1]
        )[0]


        max_index = similarity_matrix.argmax()

        max_similarity = float(
            similarity_matrix[max_index]
        )


        if max_similarity >= 0.45:

            duplicate_id = int(
                df.iloc[max_index]["id"]
            )


            return (
                f"Possible Duplicate of Complaint #{duplicate_id}",
                round(max_similarity * 100, 2)
            )


        return "No Duplicate", round(
            max_similarity * 100,
            2
        )


    except Exception:

        return "No Duplicate", 0.0


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

    if not name or not complaint or not location:

        return """
❌ **Please fill all required fields:**

- Full Name
- Complaint Description
- Location
"""


    normalized_complaint = normalize_text(
        complaint
    )


    category, department = classify_complaint(
        normalized_complaint
    )


    priority = calculate_priority(
        normalized_complaint
    )


    duplicate_of, similarity = detect_duplicate(
        normalized_complaint
    )


    conn = get_connection()
    cursor = conn.cursor()


    created_at = datetime.now().strftime(
        "%d-%m-%Y %H:%M:%S"
    )


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

    }.get(priority, "⚪")


    return f"""

# ✅ Complaint Submitted Successfully!

## Complaint ID: **#{complaint_id}**

---

### 🤖 AI Analysis Result

**🏷️ Category:** {category}

**🏢 Routed Department:** {department}

**{priority_icon} Priority:** {priority}

**🔁 Duplicate Analysis:** {duplicate_of}

**📊 Similarity Score:** {similarity}%

---

### 📍 Location

**{location}**

---

### 🔄 Current Status

**Submitted**

Your grievance has been analyzed and routed using
the **ROOTS & RISE AI Intelligence Engine**.
"""


# =========================================================
# TRACK COMPLAINT
# =========================================================

def track_complaint(complaint_id):

    if complaint_id is None:

        return "❌ Please enter a Complaint ID."


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

    """, (int(complaint_id),))


    result = cursor.fetchone()

    conn.close()


    if not result:

        return "❌ Complaint ID not found."


    return f"""

# 🔎 Complaint Tracking

## Complaint ID: **#{result[0]}**

| Information | Details |
|---|---|
| 🏷️ Category | **{result[1]}** |
| 🏢 Department | **{result[2]}** |
| ⚡ Priority | **{result[3]}** |
| 🔄 Status | **{result[4]}** |
| 📍 Location | **{result[5]}** |
| 🕒 Submitted | **{result[6]}** |

"""


# =========================================================
# DASHBOARD DATA
# =========================================================

def generate_dashboard():

    df = get_all_complaints()


    if df.empty:

        empty_chart = px.bar(
            title="No Complaints Available"
        )


        return (
            """
# 🏛️ Authority Intelligence Dashboard

### No complaints have been submitted yet.
""",

            pd.DataFrame(),

            empty_chart,

            empty_chart,

            empty_chart
        )


    total = len(df)


    high = len(
        df[df["priority"] == "High"]
    )


    medium = len(
        df[df["priority"] == "Medium"]
    )


    low = len(
        df[df["priority"] == "Low"]
    )


    resolved = len(
        df[df["status"] == "Resolved"]
    )


    duplicates = len(
        df[
            df["duplicate_of"] !=
            "No Duplicate"
        ]
    )


    stats = f"""

# 🏛️ ROOTS & RISE AUTHORITY DASHBOARD

## 📊 Live Grievance Intelligence

| Metric | Count |
|---|---:|
| 📢 Total Complaints | **{total}** |
| 🔴 High Priority | **{high}** |
| 🟠 Medium Priority | **{medium}** |
| 🟢 Low Priority | **{low}** |
| 🔁 Possible Duplicates | **{duplicates}** |
| ✅ Resolved | **{resolved}** |

"""


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

        title="📊 Complaints by Category"

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

        title="⚡ Priority Distribution"

    )


    valid_locations = df.dropna(
        subset=["latitude", "longitude"]
    )


    if valid_locations.empty:

        map_chart = px.scatter_geo(
            title="📍 No Geographic Data Available"
        )

    else:

        map_chart = px.scatter_geo(

            valid_locations,

            lat="latitude",

            lon="longitude",

            hover_name="location",

            hover_data=[

                "category",
                "priority",
                "department",
                "status"

            ],

            title="📍 Geographic Complaint Hotspots"

        )


        map_chart.update_geos(

            projection_type="natural earth",

            showcountries=True,

            showland=True

        )


    display_df = df.rename(columns={

        "id": "Complaint ID",

        "name": "Citizen Name",

        "email": "Email",

        "complaint": "Complaint",

        "location": "Location",

        "category": "Category",

        "department": "Department",

        "priority": "Priority",

        "duplicate_of": "Duplicate Analysis",

        "similarity": "Similarity %",

        "status": "Status",

        "created_at": "Submitted At"

    })


    columns_to_show = [

        "Complaint ID",

        "Citizen Name",

        "Complaint",

        "Location",

        "Category",

        "Department",

        "Priority",

        "Duplicate Analysis",

        "Status",

        "Submitted At"

    ]


    display_df = display_df[
        [
            column for column in columns_to_show
            if column in display_df.columns
        ]
    ]


    return (

        stats,

        display_df,

        category_chart,

        priority_chart,

        map_chart

    )


# =========================================================
# UPDATE STATUS
# =========================================================

def update_status(
    complaint_id,
    new_status
):

    if complaint_id is None:

        return "❌ Please enter a Complaint ID."


    conn = get_connection()

    cursor = conn.cursor()


    cursor.execute(

        """

        UPDATE complaints

        SET status = ?

        WHERE id = ?

        """,

        (
            new_status,
            int(complaint_id)
        )

    )


    conn.commit()


    if cursor.rowcount == 0:

        conn.close()

        return "❌ Complaint ID not found."


    conn.close()


    return f"""

✅ **Complaint #{int(complaint_id)} updated successfully!**

New Status: **{new_status}**

"""


# =========================================================
# USER INTERFACE
# =========================================================

with gr.Blocks(
    title="Roots & Rise | AI Grievance Platform",
    theme=gr.themes.Soft()
) as app:


    gr.Markdown("""

# 🌱 ROOTS & RISE

## AI-Powered Citizen Grievance Intelligence Platform

### *From Citizen Voices to Smarter Governance*

ROOTS & RISE uses Artificial Intelligence to
automatically **classify grievances, assign priority,
route complaints to departments, detect duplicates,
and visualize geographic hotspots**.

""")


    with gr.Tabs():


        # =================================================
        # HOME
        # =================================================

        with gr.Tab("🏠 Home"):

            gr.Markdown("""

# Welcome to ROOTS & RISE 🌱

## A Complete AI-Powered Governance Solution

### 👤 For Citizens
Submit grievances and receive AI-powered analysis.

### 🤖 AI Intelligence
Automatic classification, routing and priority scoring.

### 🔁 Duplicate Detection
Identify semantically similar complaints.

### 🏛️ For Authorities
Monitor complaints through a live intelligence dashboard.

### 📍 Geographic Intelligence
Identify complaint clusters and potential hotspots.

""")


        # =================================================
        # CITIZEN PORTAL
        # =================================================

        with gr.Tab("👤 Citizen Portal"):


            gr.Markdown("""

# 👤 Submit Your Grievance

Fill in the details below.
The ROOTS & RISE AI Engine will automatically
analyze your complaint.

""")


            with gr.Row():

                with gr.Column():

                    citizen_name = gr.Textbox(
                        label="Full Name",
                        placeholder="Enter your full name"
                    )


                    citizen_email = gr.Textbox(
                        label="Email Address",
                        placeholder="example@email.com"
                    )


                    complaint_text = gr.Textbox(

                        label="Describe Your Complaint",

                        placeholder="""
Example:
There is a large pothole near my locality.
The road is dangerous and accidents may happen.
""",

                        lines=6
                    )


                    citizen_location = gr.Textbox(

                        label="Location",

                        placeholder="Example: Thane Railway Station"

                    )


                with gr.Column():

                    latitude = gr.Number(
                        label="Latitude (Optional)",
                        value=None
                    )


                    longitude = gr.Number(
                        label="Longitude (Optional)",
                        value=None
                    )


                    submit_button = gr.Button(

                        "🤖 Analyze & Submit with AI",

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

                outputs=submission_result

            )


        # =================================================
        # TRACK COMPLAINT
        # =================================================

        with gr.Tab("🔎 Track Complaint"):


            gr.Markdown("""

# 🔎 Track Your Complaint

Enter the Complaint ID received after submission.

""")


            tracking_id = gr.Number(

                label="Complaint ID",

                precision=0

            )


            track_button = gr.Button(
                "🔎 Track Complaint"
            )


            tracking_result = gr.Markdown()


            track_button.click(

                fn=track_complaint,

                inputs=tracking_id,

                outputs=tracking_result

            )


        # =================================================
        # AUTHORITY DASHBOARD
        # =================================================

        with gr.Tab("🏛️ Authority Dashboard"):


            gr.Markdown("""

# 🏛️ Authority Intelligence Dashboard

Monitor citizen grievances and make
data-driven governance decisions.

""")


            refresh_button = gr.Button(

                "🔄 Refresh Live Dashboard",

                variant="primary"

            )


            dashboard_stats = gr.Markdown()


            complaints_table = gr.Dataframe(

                label="📋 All Complaints",

                interactive=False,

                wrap=True

            )


            with gr.Row():

                category_output = gr.Plot(
                    label="Category Analytics"
                )


                priority_output = gr.Plot(
                    label="Priority Analytics"
                )


            map_output = gr.Plot(
                label="📍 Geographic Hotspots"
            )


            refresh_button.click(

                fn=generate_dashboard,

                outputs=[

                    dashboard_stats,

                    complaints_table,

                    category_output,

                    priority_output,

                    map_output

                ]

            )


            gr.Markdown("---")


            gr.Markdown(
                "## 🔄 Complaint Status Management"
            )


            with gr.Row():

                status_complaint_id = gr.Number(

                    label="Complaint ID",

                    precision=0

                )


                new_status = gr.Dropdown(

                    choices=[

                        "Submitted",

                        "Under Review",

                        "In Progress",

                        "Resolved"

                    ],

                    value="Under Review",

                    label="New Status"

                )


            status_button = gr.Button(

                "✅ Update Complaint Status",

                variant="primary"

            )


            status_result = gr.Markdown()


            status_button.click(

                fn=update_status,

                inputs=[

                    status_complaint_id,

                    new_status

                ],

                outputs=status_result

            )


print("🌱 ROOTS & RISE APPLICATION READY!")
