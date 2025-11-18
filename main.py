import cv2
import base64
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

from ultralytics import YOLO

import dash
from dash import Dash, html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc

import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


# CSS personalizado para tema oscuro
app.index_string = '''
<!DOCTYPE html>
<html data-bs-theme="dark">
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                background-color: #1a1a1a;
                color: #ffffff;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            .kpi-card {
                background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
                border-radius: 12px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
                padding: 20px;
                text-align: center;
                margin: 10px;
                border: 1px solid #4a5568;
                border-left: 4px solid #00d4ff;
            }
            .kpi-value {
                font-size: 2.5rem;
                font-weight: bold;
                color: #00d4ff;
                margin: 10px 0;
                text-shadow: 0 0 10px rgba(0, 212, 255, 0.3);
            }
            .kpi-label {
                font-size: 0.9rem;
                color: #a0aec0;
                text-transform: uppercase;
                font-weight: 500;
            }
            .dashboard-header {
                background: linear-gradient(135deg, #1a202c 0%, #2d3748 50%, #4a5568 100%);
                color: white;
                border-radius: 12px;
                margin-bottom: 20px;
                border: 1px solid #4a5568;
            }
            .card {
                background-color: #2d3748 !important;
                border: 1px solid #4a5568 !important;
                color: #ffffff !important;
            }
            .card-header {
                background-color: #1a202c !important;
                border-bottom: 1px solid #4a5568 !important;
                color: #00d4ff !important;
                font-weight: bold;
            }
            .card-body {
                background-color: #2d3748 !important;
            }
            /* Estilos para componentes Plotly en tema oscuro */
            .js-plotly-plot .plotly .modebar {
                background-color: #2d3748;
            }
            .js-plotly-plot .plotly .modebar-btn {
                color: #ffffff;
            }
        </style>
        <script>
            // Forzar tema oscuro globalmente
            document.documentElement.setAttribute('data-bs-theme', 'dark');
        </script>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''
traffic_history = []
latest_counts = defaultdict(int)

# ------------------ YOLO ---------------------
model = YOLO("yolov8n.pt")
stream_url = "https://live.smartechlatam.online/claro/javierprado/index.m3u8"

def predict_future_traffic(history, minutes_ahead=5):
    if len(history) < 5:
        return None, None  # No hay suficientes datos

    df = pd.DataFrame(history)

    # Crear variable temporal (0,1,2,...)
    df["t"] = np.arange(len(df))

    # Entrenar modelo simple
    model = LinearRegression()
    model.fit(df[["t"]], df["total"])

    # Tiempo futuro
    future_df = pd.DataFrame({"t": [len(df) + minutes_ahead]})
    prediction = model.predict(future_df)[0]
    # Error estimado (desviación estándar de residuos)
    preds = model.predict(df[["t"]])
    error = np.std(df["total"] - preds)

    return prediction, error
def get_frame_from_stream(url):
    cap = cv2.VideoCapture(url)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None, {}

    results = model(frame, imgsz=640, conf=0.5)

    vehicle_counts = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0}

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label in vehicle_counts:
                vehicle_counts[label] += 1

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    _, buffer = cv2.imencode('.jpg', frame)
    encoded = base64.b64encode(buffer).decode('utf-8')
    frame_b64 = f"data:image/jpeg;base64,{encoded}"

    return frame_b64, vehicle_counts


# ------------------ AI SEMÁFORO ---------------------
def ai_traffic_controller(history):
    if len(history) < 5:
        return {
            "status": "verde",
            "countdown": 20,
            "reason": "Aún sin suficientes datos"
        }

    df = pd.DataFrame(history)

    avg = df["total"].tail(10).mean()
    current = df["total"].iloc[-1]
    cars = df["cars"].iloc[-1]
    buses = df["buses"].iloc[-1]

    congestion_factor = current / max(1, avg)

    if congestion_factor > 1.7 or buses > 5:
        status = "rojo"
        countdown = 35
        reason = "Alta congestión, despejando vía"
    elif congestion_factor > 1.2:
        status = "amarillo"
        countdown = 12
        reason = "Congestión moderada"
    else:
        status = "verde"
        countdown = 20
        reason = "Flujo normal"

    return {
        "status": status,
        "countdown": countdown,
        "reason": reason,
        "current_flow": current,
        "cars": cars,
        "buses": buses,
        "factor": round(congestion_factor, 2)
    }


# ------------------ DASHBOARD ---------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])


app.layout = dbc.Container([

    # HEADER
    html.Div([
        html.Img(src='/assets/logo_traffiq.png',
                 style={'height': '60px'}),
        html.H1("TraffiQ - AI Traffic Control System",
                style={'margin-left': '20px', 'color': '#00d4ff'})
    ], style={'display': 'flex', 'align-items': 'center', 'padding': '20px'}),

    dbc.Row([

        # ---------------- COL 1 ----------------
        dbc.Col([
            html.Div([
                html.H4("Vehículos Detectados"),
                html.H2(id="total-vehicles", children="--")
            ], className="kpi-card"),

            dbc.Card([
                dbc.CardHeader("🚗 Clasificación de Vehículos"),
                dbc.CardBody([dcc.Graph(id="vehicle-classification")])
            ]),

            dbc.Card([
                dbc.CardHeader("📊 Análisis de Correlación"),
                dbc.CardBody([dcc.Graph(id="scatter-plot")])
            ])
        ], width=4),

        # ---------------- COL 2 ----------------
        dbc.Col([
            html.Div([
                html.H4("Tiempo Espera Promedio"),
                html.H2(id="avg-wait-time", children="--")
            ], className="kpi-card"),

            dbc.Card([
                dbc.CardHeader("📈 Tendencia del Tráfico"),
                dbc.CardBody([dcc.Graph(id="traffic-trend")])
            ]),

            dbc.Card([
                dbc.CardHeader("🚦 Control Semafórico IA"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([dcc.Graph(id="traffic-light-status")], width=6),
                        dbc.Col([
                            html.Div(id="recommendations",
                                     style={'padding': '15px',
                                            'border': '1px solid #4a5568',
                                            'border-radius': '8px',
                                            'height': '250px'})
                        ], width=6)
                    ])
                ])
            ])
        ], width=4),

        # ---------------- COL 3 ----------------
        dbc.Col([
            html.Div([
                html.H4("Reducción Lograda"),
                html.H2(id="reduction-percent", children="--")
            ]),

            dbc.Card([
                dbc.CardHeader("🎥 Cámara en Vivo"),
                dbc.CardBody([
                    html.Img(id="video-feed",
                             src="./assets/camara1.png",
                             style={'width': '100%', 'border-radius': '8px', 'height': '350px'})
                ])
            ]),

            dbc.Card([
                dbc.CardHeader("🗺️ Heatmap de Congestión"),
                dbc.CardBody([dcc.Graph(id="congestion-heatmap")])
            ])
        ], width=4)

    ]),

    dcc.Interval(id='interval-component', interval=5000, n_intervals=0)

], fluid=True)


# ---------------- CALLBACK PRINCIPAL ----------------
@callback(
    [Output('avg-wait-time', 'children'),
     Output('reduction-percent', 'children'),
     Output('traffic-trend', 'figure'),
     Output('congestion-heatmap', 'figure'),
     Output('traffic-light-status', 'figure'),
     Output('recommendations', 'children'),
     Output('scatter-plot', 'figure')],
    Input('interval-component', 'n_intervals')
)
def update_dashboard(n):

    wait_time = np.random.uniform(45, 120)
    reduction = np.random.uniform(15, 35)

    df = pd.DataFrame(traffic_history) if len(traffic_history) > 2 else \
        pd.DataFrame({"time": [], "cars": [], "buses": [], "total": []})

    ai_decision = ai_traffic_controller(traffic_history)
    light_status = ai_decision["status"]
    countdown = ai_decision["countdown"]
    ai_reason = ai_decision["reason"]

    # --------- LINE TREND ----------
    if not df.empty:
        trend_fig = px.line(
            df,
            x="time",
            y=["cars", "buses"],
            title="Flujo Vehicular"
        )
        trend_fig.update_layout(template="plotly_dark")
    else:
        trend_fig = go.Figure()

    # --------- HEATMAP ----------
    pred, err = predict_future_traffic(traffic_history, minutes_ahead=5)
    forecast_fig = go.Figure()

    if pred is None:
        # No data yet
        forecast_fig.add_annotation(text="Aún no hay suficientes datos para predecir",
                                    x=0.5, y=0.5, showarrow=False,
                                    font=dict(color="white", size=18))
    else:
        if len(traffic_history) < 5:
            df_hist = pd.DataFrame({
                "t": [0, 1, 2, 3, 4],
                "total": [0, 0, 0, 0, 0]
            })
        else:
            df_hist = pd.DataFrame({
                "t": range(len(traffic_history)),
                "total": [h["total"] for h in traffic_history]
            })
        # Crear DataFrame extendido
        df_future = pd.DataFrame({
            "t": [len(df_hist) + 5],
            "pred": [pred],
            "hi": [pred + err],
            "lo": [pred - err]
        })
        # Línea real
        forecast_fig.add_trace(go.Scatter(
            x=df_hist["t"],
            y=df_hist["total"],
            mode="lines+markers",
            name="Tráfico Real",
            line=dict(color="#00d4ff")
        ))

        # Predicción
        forecast_fig.add_trace(go.Scatter(
            x=df_future["t"],
            y=df_future["pred"],
            mode="lines+markers",
            name="Predicción IA",
            line=dict(color="#ffa500", dash="dash")
        ))

        # Área sombreada (intervalo de error)
        forecast_fig.add_trace(go.Scatter(
            x=[df_future["t"][0], df_future["t"][0]],
            y=[df_future["lo"][0], df_future["hi"][0]],
            fill="toself",
            mode="lines",
            name="Intervalo de error",
            line=dict(color="rgba(255,255,255,0.3)")
        ))

    forecast_fig.update_layout(
        title="🤖 Predicción IA de Tráfico (5 min al futuro)",
        template="plotly_dark",
        paper_bgcolor='rgba(45,55,72,0.8)',
        plot_bgcolor='rgba(26,32,44,0.8)',
        font_color='#ffffff',
        title_font_color='#00d4ff'
    )

    # --------- SEMÁFORO IA ----------
    light_colors = {'verde': '#4ade80', 'amarillo': '#f7931e', 'rojo': '#ef4444'}

    traffic_light_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=countdown,
        gauge={'axis': {'range': [0, 40]},
               'bar': {'color': light_colors[light_status]}},
        title={'text': f"Estado: {light_status.upper()}"}
    ))
    traffic_light_fig.update_layout(template="plotly_dark")

    # --------- SCATTER ----------
    if len(traffic_history) > 5:
        scatter_df = pd.DataFrame(traffic_history)
    else:
        scatter_df = pd.DataFrame({"total": [0], "wait": [0], "type": ["--"]})

    if "wait" not in scatter_df.columns:
        scatter_df["wait"] = scatter_df["total"] * 0.8

    if "type" not in scatter_df.columns:
        scatter_df["type"] = "Principal"

    scatter_fig = px.scatter(
        scatter_df,
        x="total",
        y="wait",
        color="type",
        title="Flujo vs Tiempo de Espera"
    )
    scatter_fig.update_layout(template="plotly_dark")

    # --------- RECOMENDACIONES IA ----------
    recommendations_text = html.Div([
        html.H6("🤖 Recomendaciones IA"),
        html.P(f"• Estado sugerido: {light_status.upper()}"),
        html.P(f"• Razón: {ai_reason}"),
        html.P(f"• Flujo actual: {ai_decision.get('current_flow', 'N/A')} vehículos"),
        html.P(f"• Autos: {ai_decision.get('cars', 0)} | Buses: {ai_decision.get('buses', 0)}"),
        html.P(f"• Factor congestión: {ai_decision.get('factor', 'N/A')}"),
    ])

    return (f"{wait_time:.1f}s", f"{reduction:.1f}%",
            trend_fig, forecast_fig, traffic_light_fig,
            recommendations_text, scatter_fig)


# ---------------- VIDEO FEED ----------------
@callback(
    Output('video-feed', 'src'),
    Output('total-vehicles', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_camera(n):
    global latest_counts

    frame, counts = get_frame_from_stream(stream_url)

    if frame is None:
        return "./assets/camara1.png", "--"

    latest_counts = counts
    total = sum(counts.values())
    wait_estimate = total * 0.8

    traffic_history.append({
        "time": datetime.now(),
        "total": total,
        "cars": counts.get("car", 0),
        "buses": counts.get("bus", 0),
        "wait": wait_estimate,
        "type": "Principal"
    })

    traffic_history[:] = traffic_history[-2000:]

    return frame, str(total)


# ---------------- PIE CHART ----------------
@callback(
    Output('vehicle-classification', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_vehicle_pie(n):
    counts = latest_counts

    df = pd.DataFrame({
        'tipo': ['Autos', 'Motos', 'Buses', 'Camiones'],
        'cantidad': [
            counts.get('car', 0),
            counts.get('motorcycle', 0),
            counts.get('bus', 0),
            counts.get('truck', 0)
        ]
    })

    fig = px.pie(df, values='cantidad', names='tipo',
                 title="Clasificación (IA)",
                 color_discrete_sequence=['#00d4ff', '#ff6b35', '#f7931e', '#4ade80'])
    fig.update_layout(template="plotly_dark")

    return fig


# ---------------- RUN ----------------
if __name__ == '__main__':
    app.run(debug=True)
