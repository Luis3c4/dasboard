# Manual de Usuario - TraffiQ Dashboard

## Descripción General

TraffiQ es un sistema de monitoreo de tráfico vehicular en tiempo real que utiliza inteligencia artificial (YOLO) para detectar y clasificar vehículos desde una transmisión de video en vivo. El dashboard muestra métricas clave, gráficos de análisis y recomendaciones automáticas para el control semafórico.

## Requisitos Previos

- Python 3.8 o superior
- Conexión a internet (para la transmisión de video)
- Dependencias instaladas (ver archivo requirements.txt si existe)

## Instalación

1. Instalar las dependencias necesarias:
```bash
pip install dash dash-bootstrap-components plotly pandas numpy opencv-python ultralytics scikit-learn
```

2. Asegúrate de tener el archivo `yolov8n.pt` en el directorio del proyecto.

## Uso

1. Ejecutar la aplicación:
```bash
python main.py
```

2. Abrir el navegador en la dirección que aparece en la consola (generalmente `http://127.0.0.1:8050`)

3. El dashboard se actualiza automáticamente cada 5 segundos con nuevos datos del tráfico.

## Componentes del Dashboard

### Métricas Principales (KPI)

- **Vehículos Detectados**: Total de vehículos detectados en el momento actual
- **Tiempo Espera Promedio**: Tiempo estimado de espera en segundos
- **Reducción Lograda**: Porcentaje de mejora en el flujo de tráfico

### Gráficos y Visualizaciones

#### 1. Clasificación de Vehículos
**¿Para qué sirve?**  
Muestra la distribución porcentual de los tipos de vehículos detectados (autos, motos, buses, camiones) en el momento actual. Útil para entender la composición del tráfico.

#### 2. Análisis de Correlación (Flujo vs Tiempo de Espera)
**¿Para qué sirve?**  
Gráfico de dispersión que relaciona el volumen total de vehículos con el tiempo de espera estimado. Permite identificar patrones de congestión y correlaciones entre flujo vehicular y tiempos de espera.

#### 3. Tendencia del Tráfico
**¿Para qué sirve?**  
Gráfico de líneas que muestra la evolución del flujo de autos y buses a lo largo del tiempo. Ayuda a identificar horas pico, tendencias crecientes o decrecientes, y patrones de comportamiento del tráfico.

#### 4. Predicción IA de Tráfico
**¿Para qué sirve?**  
Predice el volumen de tráfico esperado en los próximos 5 minutos basándose en el historial reciente. La línea punteada muestra la predicción y el área sombreada indica el intervalo de error estimado. Útil para anticipar congestiones.

#### 5. Control Semafórico IA
**¿Para qué sirve?**  
Indicador visual (gauge) que muestra el estado recomendado del semáforo (verde, amarillo, rojo) y el tiempo de ciclo sugerido. El sistema ajusta automáticamente las recomendaciones según la congestión detectada.

#### 6. Cámara en Vivo
**¿Para qué sirve?**  
Muestra el video en tiempo real de la cámara con las detecciones de vehículos superpuestas (cajas delimitadoras y etiquetas). Permite verificar visualmente las detecciones del sistema.

### Recomendaciones IA

El panel de recomendaciones muestra:
- Estado sugerido del semáforo
- Razón de la decisión
- Flujo actual de vehículos
- Desglose de autos y buses
- Factor de congestión calculado

## Funcionamiento del Sistema

1. **Detección**: El sistema captura frames del video en vivo cada 5 segundos y utiliza YOLO para detectar vehículos.
2. **Clasificación**: Los vehículos se clasifican en categorías (car, motorcycle, bus, truck).
3. **Análisis**: Los datos se almacenan en un historial y se analizan para calcular métricas y tendencias.
4. **Predicción**: Se utiliza regresión lineal para predecir el tráfico futuro.
5. **Recomendaciones**: El algoritmo de IA del semáforo ajusta las recomendaciones según la congestión.

## Notas Importantes

- El dashboard requiere conexión a internet para acceder al stream de video
- Las predicciones mejoran con más datos históricos (se necesitan al menos 5 registros)
- El sistema mantiene un historial de los últimos 2000 registros para optimizar el rendimiento
- Si no se puede acceder al video, se mostrará una imagen predeterminada

## Solución de Problemas

- **No se detectan vehículos**: Verificar que la URL del stream esté funcionando
- **Gráficos vacíos**: Esperar unos momentos para que se acumulen datos suficientes
- **Error al iniciar**: Verificar que todas las dependencias estén instaladas correctamente

