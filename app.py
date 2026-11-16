"""
HCAI Framework Classifier
Unit 5: Human Centered AI

Based on Ben Shneiderman's 2D HCAI framework:
  X-axis = Computer Automation (0-10)
  Y-axis = Human Control (0-10)
"""

import numpy as np
import streamlit as st
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="HCAI Framework Classifier", layout="wide")

# ---------------------------------------------------------------------------
# 1. DATA: questions, presets, quadrants
# ---------------------------------------------------------------------------

# Each question is (key, label). The key is used to store the slider value.
AUTOMATION_QUESTIONS = [
    ("a1", "Works without asking the user"),
    ("a2", "Makes decisions alone"),
    ("a3", "Learns by itself"),
    ("a4", "Does most of the task"),
    ("a5", "Acts faster than a human can react"),
]

CONTROL_QUESTIONS = [
    ("c1", "User can stop / override it anytime"),
    ("c2", "Explains its decisions"),
    ("c3", "User can change settings"),
    ("c4", "Asks confirmation before important actions"),
    ("c5", "User can undo actions"),
]

# Preset answers (1-5) in the order of the questions above.
# These are rough example values for the presentation, not official ratings.
# Each preset also has a "name" that fills the product name box.
PRESETS = {
    "Custom":          {"name": "My Product",      "auto": [3, 3, 3, 3, 3], "ctrl": [3, 3, 3, 3, 3]},
    "Tesla Autopilot": {"name": "Tesla Autopilot", "auto": [4, 4, 3, 4, 5], "ctrl": [4, 2, 4, 3, 3]},
    "Alexa":           {"name": "Alexa",           "auto": [4, 3, 3, 3, 3], "ctrl": [4, 2, 4, 4, 3]},
    "Roomba":          {"name": "Roomba",          "auto": [5, 4, 3, 5, 2], "ctrl": [4, 2, 4, 3, 3]},
    "Instagram Feed":  {"name": "Instagram Feed",  "auto": [5, 5, 5, 5, 5], "ctrl": [1, 1, 2, 1, 1]},
    "Google Maps":     {"name": "Google Maps",     "auto": [3, 4, 4, 4, 3], "ctrl": [5, 4, 5, 4, 4]},
}

# Quadrant info: name, vibrant colour, explanation
QUADRANTS = {
    "trustworthy": {
        "name": "Reliable, Safe & Trustworthy",
        "color": "#10B981", "label": "green",
        "text": "High automation AND high human control. This is the goal of "
                "HCAI: powerful AI that people can still understand and steer.",
    },
    "excessive_auto": {
        "name": "Excessive Automation",
        "color": "#F43F5E", "label": "red",
        "text": "The computer does a lot but the human has little control. "
                "This is dangerous when the AI makes mistakes and nobody can step in.",
    },
    "excessive_human": {
        "name": "Excessive Human Control",
        "color": "#F59E0B", "label": "orange",
        "text": "The human has lots of control but the computer helps very little. "
                "Users may be overloaded and miss out on useful automation.",
    },
    "simple": {
        "name": "Simple Tool",
        "color": "#6366F1", "label": "violet",
        "text": "Low automation and low control. A basic tool that does simple "
                "jobs without much AI involved.",
    },
}

# Design suggestions shown when a control answer is low (1 or 2).
SUGGESTIONS = {
    "c1": "Add an emergency stop button so the user can override at any time.",
    "c2": "Add explanations, e.g. a 'Why did I see this?' button.",
    "c3": "Give users a settings page to customise how the AI behaves.",
    "c4": "Ask for confirmation before important or risky actions.",
    "c5": "Add an undo button or history so mistakes can be reversed.",
}


# ---------------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def average_score(values):
    """Turn 1-5 answers into a 0-10 score: all 1s -> 0, all 5s -> 10."""
    average = sum(values) / len(values)
    return round((average - 1) / 4 * 10, 1)


def get_quadrant(auto_score, ctrl_score):
    """Split at 5 on both axes and return the quadrant key."""
    high_auto = auto_score >= 5
    high_ctrl = ctrl_score >= 5
    if high_auto and high_ctrl:
        return "trustworthy"
    if high_auto and not high_ctrl:
        return "excessive_auto"
    if not high_auto and high_ctrl:
        return "excessive_human"
    return "simple"


def load_preset():
    """Called when the dropdown changes: copy preset values into the sliders."""
    preset = PRESETS[st.session_state["preset"]]
    st.session_state["product_name"] = preset["name"]
    for (key, _), value in zip(AUTOMATION_QUESTIONS, preset["auto"]):
        st.session_state[key] = value
    for (key, _), value in zip(CONTROL_QUESTIONS, preset["ctrl"]):
        st.session_state[key] = value


def hex_to_rgb(hex_color):
    """'#10B981' -> np.array([16, 185, 129])"""
    h = hex_color.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=float)


def make_gradient_image(size=400):
    """Build a size x size picture for the 0-10 x 0-10 graph area.
    Each quadrant is the full vibrant colour at its outer corner and fades to a
    lighter, softer shade of the SAME colour (25% white) near the centre."""
    # Pixel centres converted to graph coordinates (row 0 = top = y of 10)
    coords = (np.arange(size) + 0.5) / size * 10
    x, y = np.meshgrid(coords, coords[::-1])

    # Pick the quadrant colour for every pixel
    right, top = x >= 5, y >= 5
    full = np.zeros((size, size, 3))
    image = full
    image[top & right] = hex_to_rgb(QUADRANTS["trustworthy"]["color"])
    image[~top & right] = hex_to_rgb(QUADRANTS["excessive_auto"]["color"])
    image[top & ~right] = hex_to_rgb(QUADRANTS["excessive_human"]["color"])
    image[~top & ~right] = hex_to_rgb(QUADRANTS["simple"]["color"])

    # t = distance from the centre (5,5) divided by the largest distance (a corner)
    distance = np.sqrt((x - 5) ** 2 + (y - 5) ** 2)
    t = (distance / np.sqrt(50))[..., None]

    # light shade = the quadrant colour mixed with 25% white
    light = image * 0.75 + 255 * 0.25

    # colour = light_shade*(1-t) + full_colour*t
    image = light * (1 - t) + image * t
    return Image.fromarray(image.astype("uint8"))


def draw_graph(products, compare=False):
    """Draw the HCAI graph with Plotly.
    products = list of (name, auto_score, ctrl_score).
    compare=False -> one product with glow, star and guide lines.
    compare=True  -> several products, each with its own colour and a legend."""
    fig = go.Figure()

    # 1. Gradient picture stretched over exactly 0-10 x 0-10, behind everything.
    fig.add_layout_image(
        source=make_gradient_image(), xref="x", yref="y",
        x=0, y=10, sizex=10, sizey=10, xanchor="left", yanchor="top",
        sizing="stretch", layer="below")

    # Quadrant labels: x, y, text, text position (outer CORNERS of each quadrant)
    labels = [
        (9.3, 9.0, "Reliable, Safe &<br>Trustworthy ✅", "bottom left"),
        (9.5, 0.5, "Excessive Automation ⚠️", "top left"),
        (0.7, 9.0, "Excessive Human<br>Control ✋", "bottom right"),
        (0.5, 0.5, "Simple Tool 🔧", "top right"),
    ]

    # 2. Quadrant labels in white bold (added first, so dots draw on top).
    fig.add_trace(go.Scatter(
        x=[l[0] for l in labels], y=[l[1] for l in labels],
        mode="text", text=[l[2] for l in labels],
        textposition=[l[3] for l in labels],
        textfont=dict(size=15, color="white",
                      family="Arial Black, Arial, sans-serif"),
        hoverinfo="skip", showlegend=False))

    # 3. Solid white divider lines at x=5 and y=5.
    divider = dict(color="rgba(255,255,255,0.8)", width=2)
    fig.add_shape(type="line", x0=5, x1=5, y0=0, y1=10, line=divider)
    fig.add_shape(type="line", x0=0, x1=10, y0=5, y1=5, line=divider)

    # Solid white border around the full 0-10 plot area
    fig.add_shape(type="rect", x0=0, x1=10, y0=0, y1=10,
                  line=dict(color="white", width=2))

    palette = px.colors.qualitative.D3  # different colour for each product in compare mode

    for i, (name, x, y) in enumerate(products):
        quad = QUADRANTS[get_quadrant(x, y)]
        color = palette[i % len(palette)] if compare else quad["color"]

        # Text shown when hovering over the dot
        hover = (f"<b>{name}</b><br>Automation: {x}<br>"
                 f"Control: {y}<br>Quadrant: {quad['name']}<extra></extra>")

        if not compare:
            # Dotted guide lines from the dot to the X axis and the Y axis
            # (dark colour so they stay visible on the bright quadrants)
            guide = dict(color="#111827", width=2, dash="dot")
            fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=y, line=guide, opacity=0.8)
            fig.add_shape(type="line", x0=0, x1=x, y0=y, y1=y, line=guide, opacity=0.8)

            # Big semi-transparent circle behind the dot = "glow"
            fig.add_trace(go.Scatter(
                x=[x], y=[y], mode="markers", hoverinfo="skip", showlegend=False,
                marker=dict(size=45, color="white", opacity=0.35)))

        # The solid marker (star for the single product, circle when comparing)
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers", name=name, showlegend=compare,
            hovertemplate=hover,
            marker=(
                # Compare mode: circle in the product's own colour
                dict(size=16, symbol="circle", color=color,
                     line=dict(color="white", width=2)) if compare else
                # Single product: white star with a border in the quadrant colour
                dict(size=24, symbol="star", color="white",
                     line=dict(color=color, width=3)))))

        # Label in a rounded box with an arrow pointing at the dot.
        # ax/ay are pixel offsets: negative ax = left, positive ay = below.
        # The label always points TOWARDS THE CENTRE, away from the corner labels.
        # In compare mode every second product gets a slightly longer arrow,
        # shifted sideways, so labels do not sit on top of each other.
        odd = compare and i % 2 == 1
        distance = 60 if odd else 50
        ay = distance if y >= 5 else -distance   # top half -> below, bottom half -> above
        if x > 7:
            ax = -distance                       # near the right edge -> to the left
        elif x < 3:
            ax = distance                        # near the left edge -> to the right
        else:
            ax = 0
        if odd:
            ax += -20 if x > 5 else 20
        fig.add_annotation(
            x=x, y=y, text=f"<b>{name}</b>", showarrow=True, arrowhead=2,
            arrowcolor="white", ax=ax, ay=ay,
            bgcolor="white", bordercolor=color, borderwidth=2,
            borderpad=4, font=dict(color="#111827", size=15))

    # 4. Axes: range -0.5 to 10.5 so dots at 0 or 10 are fully visible.
    tick_font = dict(size=14, color="white")
    fig.update_xaxes(title="Computer Automation →", range=[-0.5, 10.5],
                     tickvals=[0, 2, 4, 6, 8, 10], showgrid=False, zeroline=False,
                     tickfont=tick_font, automargin=True)
    fig.update_yaxes(title="Human Control →", range=[-0.5, 10.5],
                     tickvals=[0, 2, 4, 6, 8, 10], showgrid=False, zeroline=False,
                     tickfont=tick_font, automargin=True)

    # 5. Dark theme with a transparent background.
    fig.update_layout(
        template="plotly_dark", title="HCAI Framework (Shneiderman)",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0F172A",
        hoverlabel=dict(font_size=14),
        height=600, margin=dict(l=80, r=20, t=60, b=80))
    return fig


# ---------------------------------------------------------------------------
# 3. SESSION STATE (Streamlit's memory between clicks)
# ---------------------------------------------------------------------------

# Start every slider at 3 the first time the app runs.
for key, _ in AUTOMATION_QUESTIONS + CONTROL_QUESTIONS:
    if key not in st.session_state:
        st.session_state[key] = 3

if "product_name" not in st.session_state:
    st.session_state["product_name"] = "My Product"

# Saved products for compare mode: {product name: (auto score, control score)}
if "saved" not in st.session_state:
    st.session_state["saved"] = {}


# ---------------------------------------------------------------------------
# 4. PAGE LAYOUT
# ---------------------------------------------------------------------------

st.title("HCAI Framework Classifier")
st.caption("Unit 5: Human Centered AI - based on Ben Shneiderman's 2D framework")

with st.expander("About HCAI"):
    st.write(
        "Human-Centered AI (HCAI) is an idea from Ben Shneiderman. "
        "He argues that automation and human control are NOT opposites: "
        "a system can have both at the same time. "
        "His 2D framework plots Computer Automation against Human Control, "
        "and the goal is the top-right corner: Reliable, Safe & Trustworthy AI "
        "that is powerful but still keeps people in charge."
    )

left, right = st.columns(2)

# ----- LEFT COLUMN: inputs -----
with left:
    st.selectbox("Choose a preset", list(PRESETS.keys()), key="preset",
                 on_change=load_preset)
    product_name = st.text_input("Product name", key="product_name")

    st.subheader("Computer Automation (1 = never, 5 = always)")
    for key, label in AUTOMATION_QUESTIONS:
        st.slider(label, 1, 5, key=key)

    st.subheader("Human Control (1 = never, 5 = always)")
    for key, label in CONTROL_QUESTIONS:
        st.slider(label, 1, 5, key=key)

# Calculate scores from the current slider values.
auto_values = [st.session_state[k] for k, _ in AUTOMATION_QUESTIONS]
ctrl_values = [st.session_state[k] for k, _ in CONTROL_QUESTIONS]
auto_score = average_score(auto_values)
ctrl_score = average_score(ctrl_values)
quadrant = QUADRANTS[get_quadrant(auto_score, ctrl_score)]

# ----- RIGHT COLUMN: results -----
with right:
    st.subheader("Result")
    col1, col2 = st.columns(2)
    col1.metric("Automation score", f"{auto_score:.1f} / 10")
    col2.metric("Control score", f"{ctrl_score:.1f} / 10")

    # Graph of the current product only
    fig = draw_graph([(product_name, auto_score, ctrl_score)])
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"### {product_name} is: :{quadrant['label']}[{quadrant['name']}]")
    st.write(quadrant["text"])

    # Design suggestions: control questions rated 1 or 2, lowest rating first
    st.subheader("Design suggestions")
    low_keys = [k for k, _ in CONTROL_QUESTIONS if st.session_state[k] <= 2]
    low_keys.sort(key=lambda k: st.session_state[k])
    if low_keys:
        for k in low_keys:
            st.warning(SUGGESTIONS[k])
    else:
        st.success("Great! This product already gives users good control.")


# ---------------------------------------------------------------------------
# 5. COMPARE MODE
# ---------------------------------------------------------------------------

st.divider()
st.header("Compare Mode")

btn1, btn2, _ = st.columns([1, 1, 4])
if btn1.button("Add to comparison"):
    # Using the name as a dictionary key means the same name is updated, not duplicated.
    st.session_state["saved"][product_name] = (auto_score, ctrl_score)
if btn2.button("Clear comparison"):
    st.session_state["saved"] = {}

saved = st.session_state["saved"]
if saved:
    # Graph with every saved product
    products = [(name, a, c) for name, (a, c) in saved.items()]
    fig2 = draw_graph(products, compare=True)
    st.plotly_chart(fig2, use_container_width=True)

    # Small table under the graph
    table = []
    for name, a, c in products:
        table.append({
            "Product": name,
            "Automation score": a,
            "Control score": c,
            "Quadrant": QUADRANTS[get_quadrant(a, c)]["name"],
        })
    st.table(table)
else:
    st.info("Click 'Add to comparison' to save products and compare them here.")
