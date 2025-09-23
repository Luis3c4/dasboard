
from dash import Dash, html, dcc, callback, Input, Output
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go

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

# Layout modificado - reemplazar desde "# KPIs con tema oscuro" hasta el final del Container
app.layout = dbc.Container([
    # Header (sin cambios)
    html.Div([
        html.Img(src='/assets/logo_traffiq.png', style={'height': '60px', 'filter': 'brightness(1.2)'}),
        html.H1("TraffIQ - Operational Control System", 
                style={'display': 'inline-block', 'margin-left': '20px', 'color': '#00d4ff'})
    ], className="dashboard-header", style={'display': 'flex', 'align-items': 'center', 'padding': '20px'}),
    
    # Layout de 3 columnas principal
    dbc.Row([
        # COLUMNA 1 - Izquierda
        dbc.Col([
            # KPI 1
            html.Div([
                html.H4("Vehículos Detectados", className="kpi-label"),
                html.H2(id="total-vehicles", children="--", className="kpi-value")
            ], className="kpi-card", style={'margin-bottom': '20px'}),
            
            # Gráfico de clasificación de vehículos
            dbc.Card([
                dbc.CardHeader("🚗 Clasificación de Vehículos"),
                dbc.CardBody([
                    dcc.Graph(id="vehicle-classification")
                ])
            ], style={'margin-bottom': '20px'}),
            
            # Gráfico scatter adicional (placeholder)
            dbc.Card([
                dbc.CardHeader("📊 Análisis de Correlación"),
                dbc.CardBody([
                    dcc.Graph(id="scatter-plot")
                ])
            ])
        ], width=4),
        
        # COLUMNA 2 - Centro
        dbc.Col([
            # KPI 2
            html.Div([
                html.H4("Tiempo Espera Promedio", className="kpi-label"),
                html.H2(id="avg-wait-time", children="--", className="kpi-value")
            ], className="kpi-card", style={'margin-bottom': '20px'}),
            
            # Gráfico de tendencia principal (más grande)
            dbc.Card([
                dbc.CardHeader("📈 Tendencia de Tráfico"),
                dbc.CardBody([
                    dcc.Graph(id="traffic-trend")
                ])
            ], style={'margin-bottom': '20px'}),
            # Control de semáforo y recomendaciones DENTRO de la columna 2
            dbc.Card([
                dbc.CardHeader("🚦 Control de Semáforo y Recomendaciones"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id="traffic-light-status")
                        ], width=6),
                        dbc.Col([
                            html.Div(id="recommendations", className="mt-3", 
                                style={'background-color': '#1a202c', 'padding': '15px', 
                                       'border-radius': '8px', 'border': '1px solid #4a5568',
                                       'height': '250px'})
                        ], width=6)
                    ])
                ])
            ])
        ], width=4),
        
        # COLUMNA 3 - Derecha
        dbc.Col([
            # KPI 3
            html.Div([
                html.H4("Reducción Lograda", className="kpi-label"),
                html.H2(id="reduction-percent", children="--", className="kpi-value")
            ], className="kpi-card", style={'margin-bottom': '20px'}),
            
            # Video feed
            dbc.Card([
                dbc.CardHeader("🎥 Cámara 1 - En Vivo"),
                dbc.CardBody([
                    html.Img(id="video-feed", src="./assets/camara1.png", 
                            style={'width': '100%', 'border-radius': '8px', 'border': '2px solid #4a5568', 'height': '350px'}),
                    # Botón para cambiar cámara
                    dbc.Button("Cambiar Cámara", id="btn-switch-camera", color="primary", 
                    style={'margin-top': '10px', 'width': '100%'})
                ])
            ], style={'margin-bottom': '20px'}),
            
            # Heatmap de congestión
            dbc.Card([
                dbc.CardHeader("🗺️ Mapa de Congestión"),
                dbc.CardBody([
                    dcc.Graph(id="congestion-heatmap")
                ])
            ])
        ], width=4)
    ], style={'margin-bottom': '20px'}),
    
    
    dcc.Interval(
        id='interval-component',
        interval=5*1000,
        n_intervals=0
    )
], fluid=True, style={'background-color': '#1a1a1a', 'min-height': '100vh', 'padding': '20px'})

# Callback mejorado con tema oscuro para gráficos
@callback(
    [Output('total-vehicles', 'children'),
     Output('avg-wait-time', 'children'),
     Output('reduction-percent', 'children'),
     Output('traffic-trend', 'figure'),
     Output('vehicle-classification', 'figure'),
     Output('congestion-heatmap', 'figure'),
     Output('traffic-light-status', 'figure'),
     Output('recommendations', 'children'),
     Output('scatter-plot', 'figure')],  
    [Input('interval-component', 'n_intervals')]
)

def update_dashboard(n):
    # Simular datos en tiempo real
    current_vehicles = np.random.randint(50, 200)
    wait_time = np.random.uniform(45, 120)
    reduction = np.random.uniform(15, 35)
    
    # Gráfico de tendencia con tema oscuro
    hours = pd.date_range(start='2025-09-22 00:00', periods=24, freq='h')
    traffic_data = pd.DataFrame({
        'hora': hours,
        'autos': np.random.poisson(60, 24),
        'motos': np.random.poisson(25, 24),
        'buses': np.random.poisson(15, 24)
    })
    
    trend_fig = px.line(traffic_data, x='hora', y=['autos', 'motos', 'buses'],
                        title="Flujo Vehicular por Hora",
                        color_discrete_sequence=['#00d4ff', '#ff6b35', '#f7931e'])
    trend_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(45,55,72,0.8)',
        plot_bgcolor='rgba(26,32,44,0.8)',
        font_color='#ffffff',
        title_font_color='#00d4ff'
    )
    
    # Gráfico de clasificación con tema oscuro
    classification_data = pd.DataFrame({
        'tipo': ['Autos', 'Motos', 'Buses', 'Camiones'],
        'cantidad': [65, 20, 10, 5]
    })
    classification_fig = px.pie(classification_data, values='cantidad', names='tipo',
                                title="Clasificación de Vehículos",
                                color_discrete_sequence=['#00d4ff', '#ff6b35', '#f7931e', '#4ade80'])
    classification_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(45,55,72,0.8)',
        font_color='#ffffff',
        title_font_color='#00d4ff'
    )
    
    # Heatmap de congestión con tema oscuro
    days = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
    hours_heat = list(range(24))
    congestion_matrix = np.random.randint(0, 100, (7, 24))
    
    heatmap_fig = px.imshow(congestion_matrix,
                            x=hours_heat, y=days,
                            title="Mapa de Congestión Semanal",
                            color_continuous_scale="plasma",
                            aspect="auto")
    heatmap_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(45,55,72,0.8)',
        font_color='#ffffff',
        title_font_color='#00d4ff'
    )
    
    # Indicador de semáforo con tema oscuro
    light_status = np.random.choice(['verde', 'amarillo', 'rojo'])
    light_colors = {'verde': '#4ade80', 'amarillo': '#f7931e', 'rojo': '#ef4444'}
    countdown = np.random.randint(1, 30)
    
    traffic_light_fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = countdown,
        gauge = {
            'axis': {'range': [None, 30], 'tickcolor': '#ffffff'},
            'bar': {'color': light_colors[light_status]},
            'bgcolor': 'rgba(26,32,44,0.8)',
            'borderwidth': 2,
            'bordercolor': '#4a5568',
            'steps': [
                {'range': [0, 10], 'color': "rgba(74,85,104,0.3)"},
                {'range': [10, 25], 'color': "rgba(74,85,104,0.5)"}],
            'threshold': {
                'line': {'color': "#ef4444", 'width': 4},
                'thickness': 0.75, 'value': 25}
        },
        title = {'text': f"Semáforo: {light_status.upper()}<br>Tiempo restante", 'font': {'color': '#ffffff'}}
    ))
    traffic_light_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(45,55,72,0.8)',
        font_color='#ffffff',
        height=300
    )
    scatter_data = pd.DataFrame({
        'tiempo_espera': np.random.uniform(30, 150, 50),
        'num_vehiculos': np.random.uniform(20, 200, 50),
        'tipo_via': np.random.choice(['Principal', 'Secundaria'], 50)
    })
    
    scatter_fig = px.scatter(scatter_data, 
                           x='num_vehiculos', 
                           y='tiempo_espera',
                           color='tipo_via',
                           title="Vehículos vs Tiempo de Espera",
                           color_discrete_sequence=['#00d4ff', '#ff6b35'])
    scatter_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(45,55,72,0.8)',
        plot_bgcolor='rgba(26,32,44,0.8)',
        font_color='#ffffff',
        title_font_color='#00d4ff'
    )
    
    # Recomendaciones con tema oscuro
    recommendations_text = html.Div([
        html.H6("🤖 Recomendaciones Automáticas:", 
                style={'color': '#00d4ff', 'margin-bottom': '10px', 'font-weight': 'bold'}),
        html.P(f"• Ajustar ciclo en {np.random.randint(5, 15)} segundos", 
               style={'color': '#a0aec0', 'margin': '5px 0'}),
        html.P(f"• Congestión {'🔴 alta' if current_vehicles > 150 else '🟡 moderada'}", 
               style={'color': '#a0aec0', 'margin': '5px 0'}),
        html.P(f"• Priorizar vía {'🛣️ principal' if np.random.random() > 0.5 else '🛤️ secundaria'}", 
               style={'color': '#a0aec0', 'margin': '5px 0'})
    ])
    
    return (f"{current_vehicles}", f"{wait_time:.1f}s", f"{reduction:.1f}%",
        trend_fig, classification_fig, heatmap_fig, traffic_light_fig, 
        recommendations_text, scatter_fig)
# Agregar callback para alternar imagen entre dos cámaras
@callback(
    Output('video-feed', 'src'),
    Input('btn-switch-camera', 'n_clicks'),
    prevent_initial_call=True
)
def switch_camera(n_clicks):
    if n_clicks is None or n_clicks % 2 == 0:
        return "./assets/camara1.png"
    else:
        return "./assets/camara2.jpg"

if __name__ == '__main__':
    app.run(debug=True)
