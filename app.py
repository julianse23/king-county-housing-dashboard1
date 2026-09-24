
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np

# ---- Cargar datos ----
df_dash = pd.read_csv("kc_house_data_clean.csv")

# ---- Resultados de los modelos (obtenidos en el análisis previo) ----
resultados_modelos = {
    "Regresión lineal (base)": {"R2": 0.5163, "RMSE": 262532.66},
    "Regresión lineal (mejorado)": {"R2": 0.6801, "RMSE": 213511.28},
    "Regresión logística (base)": {"Accuracy": 0.7337},
    "Regresión logística (mejorado)": {"Accuracy": 0.8015},
}

resultado_hipotesis = {
    "t_stat": -16.19,
    "p_value": 1.48e-58,
    "media_antes_1980": 504422.73,
    "media_desde_1980": 587481.35
}

# ---- Inicializar la app ----
import os

# ---- Inicializar la app ----
JUPYTERHUB_SERVICE_PREFIX = os.environ.get("JUPYTERHUB_SERVICE_PREFIX", "/")
requests_pathname_prefix = JUPYTERHUB_SERVICE_PREFIX + "proxy/8050/"

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    requests_pathname_prefix=requests_pathname_prefix
)
server = app.server  # necesario para despliegue (Binder/Render/Heroku)

app.title = "King County Housing Dashboard"

# ---- Layout ----
app.layout = dbc.Container([

    dbc.Row([
        dbc.Col(html.H1("Análisis del Mercado Inmobiliario — King County, USA"), width=12),
        dbc.Col(html.P("Dashboard de resultados: contraste de hipótesis, regresión lineal y regresión logística."), width=12),
    ], style={"marginBottom": "20px", "marginTop": "20px"}),

    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("R² Modelo Lineal (mejorado)"),
            html.H2(f"{resultados_modelos['Regresión lineal (mejorado)']['R2']:.4f}")
        ])), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("RMSE Modelo Lineal"),
            html.H2(f"${resultados_modelos['Regresión lineal (mejorado)']['RMSE']:,.0f}")
        ])), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Accuracy Regresión Logística"),
            html.H2(f"{resultados_modelos['Regresión logística (mejorado)']['Accuracy']*100:.2f}%")
        ])), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("p-valor Contraste de Hipótesis"),
            html.H2("< 0.001", style={"color": "green"})
        ])), width=3),
    ], style={"marginBottom": "30px"}),

    dbc.Row([
        dbc.Col([
            html.Label("Filtrar por año de construcción:"),
            dcc.RangeSlider(
                id="filtro-anio",
                min=int(df_dash["yr_built"].min()),
                max=int(df_dash["yr_built"].max()),
                value=[int(df_dash["yr_built"].min()), int(df_dash["yr_built"].max())],
                marks={y: str(y) for y in range(1900, 2020, 20)},
                tooltip={"placement": "bottom"}
            ),
        ], width=12)
    ], style={"marginBottom": "30px"}),

    dbc.Row([
        dbc.Col(dcc.Graph(id="grafico-dispersion"), width=6),
        dbc.Col(dcc.Graph(id="grafico-histograma"), width=6),
    ]),

    dbc.Row([
        dbc.Col(dcc.Graph(id="grafico-boxplot"), width=6),
        dbc.Col(dcc.Graph(id="grafico-comparacion-r2"), width=6),
    ]),

    dbc.Row([
        dbc.Col([
            html.H3("Conclusiones principales"),
            html.Ul([
                html.Li("Existe una diferencia estadísticamente significativa en el precio entre viviendas antes y después de 1980 (p < 0.001)."),
                html.Li("El tamaño habitable (sqft_living) es el predictor individual más fuerte del precio."),
                html.Li("Incorporar ubicación (lat, long, waterfront) y calidad (grade, view) mejoró el R² de 0.52 a 0.68."),
                html.Li("El mismo enfoque mejoró el accuracy de clasificación de 73.4% a 80.2%."),
                html.Li("La regularización no fue necesaria en el modelo lineal, pero sí ayudó levemente en el modelo logístico."),
            ])
        ], width=12)
    ], style={"marginTop": "30px", "marginBottom": "40px"}),

], fluid=True)


# ---- Callbacks ----
@app.callback(
    Output("grafico-dispersion", "figure"),
    Output("grafico-histograma", "figure"),
    Output("grafico-boxplot", "figure"),
    Input("filtro-anio", "value")
)
def actualizar_graficos(rango_anio):
    dff = df_dash[(df_dash["yr_built"] >= rango_anio[0]) & (df_dash["yr_built"] <= rango_anio[1])]

    fig_disp = px.scatter(dff, x="sqft_living", y="price", opacity=0.3,
                           title="Tamaño habitable vs. Precio",
                           labels={"sqft_living": "Sqft Living", "price": "Precio (USD)"})

    fig_hist = px.histogram(dff, x="price", nbins=50,
                             title="Distribución de precios",
                             labels={"price": "Precio (USD)"})

    dff = dff.copy()
    dff["grupo"] = np.where(dff["yr_built"] < 1980, "Antes de 1980", "Desde 1980")
    fig_box = px.box(dff, x="grupo", y="price",
                      title="Precio por periodo de construcción",
                      labels={"price": "Precio (USD)", "grupo": ""})

    return fig_disp, fig_hist, fig_box


fig_r2 = px.bar(
    x=["Base (3 var.)", "Mejorado (8 var.)"],
    y=[0.5163, 0.6801],
    title="Mejora del R² — Regresión Lineal",
    labels={"x": "Modelo", "y": "R²"},
    text=[0.5163, 0.6801]
)
fig_r2.update_traces(texttemplate='%{text:.4f}', textposition='outside')

@app.callback(
    Output("grafico-comparacion-r2", "figure"),
    Input("filtro-anio", "value")
)
def mostrar_comparacion(_):
    return fig_r2


# ---- Ejecutar servidor ----
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
