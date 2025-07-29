# Monitor de performance y validación continua para TuneMetrics

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class TuneMetricsPerformanceMonitor:
    """
    Monitor de performance para modelos de TuneMetrics en producción
    Detecta drift en datos, degradación de performance y anomalías
    """
    
    def __init__(self, models_dir: str, monitoring_config: Dict = None):
        """
        Inicializa el monitor de performance
        
        Args:
            models_dir (str): Directorio de modelos
            monitoring_config (Dict): Configuración de monitoreo
        """
        self.models_dir = Path(models_dir)
        self.config = monitoring_config or self._default_config()
        
        # Almacenamiento de métricas históricas
        self.performance_history = []
        self.prediction_history = []
        self.data_drift_history = []
        
        # Umbrales de alerta
        self.alert_thresholds = self.config.get('alert_thresholds', {
            'accuracy_drop': 0.05,      # 5% drop in accuracy
            'confidence_drop': 0.10,    # 10% drop in confidence
            'data_drift_score': 0.15,   # 15% drift score
            'prediction_volume_change': 0.30  # 30% change in volume
        })
        
        # Cargar baseline metrics
        self.baseline_metrics = self._load_baseline_metrics()
        
    def _default_config(self) -> Dict:
        """Configuración por defecto del monitor"""
        return {
            'monitoring_window_days': 7,
            'alert_thresholds': {
                'accuracy_drop': 0.05,
                'confidence_drop': 0.10,
                'data_drift_score': 0.15,
                'prediction_volume_change': 0.30
            },
            'feature_drift_methods': ['statistical', 'distribution'],
            'save_monitoring_data': True,
            'monitoring_data_path': 'monitoring/performance_logs'
        }
    
    def _load_baseline_metrics(self) -> Dict:
        """Carga métricas baseline del entrenamiento"""
        metrics_path = self.models_dir / "metrics" / "model_metrics.json"
        
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                return json.load(f)
        else:
            print("⚠️  No se encontraron métricas baseline")
            return {}
    
    def log_predictions(self, predictions: pd.DataFrame, 
                       ground_truth: pd.DataFrame = None,
                       model_name: str = 'default') -> None:
        """
        Registra predicciones para monitoreo
        
        Args:
            predictions (pd.DataFrame): Predicciones del modelo
            ground_truth (pd.DataFrame): Valores reales (opcional)
            model_name (str): Nombre del modelo usado
        """
        timestamp = datetime.now()
        
        # Estadísticas de predicciones
        pred_stats = {
            'timestamp': timestamp,
            'model_name': model_name,
            'total_predictions': len(predictions),
            'avg_confidence': predictions['confidence'].mean() if 'confidence' in predictions.columns else None,
            'engagement_distribution': predictions['predicted_engagement'].value_counts().to_dict() if 'predicted_engagement' in predictions.columns else {},
            'low_confidence_count': (predictions['confidence'] < 0.6).sum() if 'confidence' in predictions.columns else None
        }
        
        # Si hay ground truth, calcular métricas de performance
        if ground_truth is not None:
            performance_metrics = self._calculate_performance_metrics(predictions, ground_truth)
            pred_stats.update(performance_metrics)
        
        # Guardar en historial
        self.prediction_history.append(pred_stats)
        
        # Guardar en archivo si está configurado
        if self.config.get('save_monitoring_data'):
            self._save_monitoring_log(pred_stats, 'predictions')
        
        print(f"📊 Predicciones registradas: {len(predictions)} canciones")
    
    def _calculate_performance_metrics(self, predictions: pd.DataFrame, 
                                     ground_truth: pd.DataFrame) -> Dict:
        """Calcula métricas de performance actuales"""
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        
        # Alinear datos
        merged = predictions.merge(ground_truth, on='spotify_track_uri', how='inner')
        
        if len(merged) == 0:
            return {'error': 'No matching records between predictions and ground truth'}
        
        y_true = merged['actual_engagement']
        y_pred = merged['predicted_engagement']
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted')
        }
        
        return metrics
    
    def monitor_data_drift(self, current_features: pd.DataFrame,
                          reference_features: pd.DataFrame = None) -> Dict:
        """
        Detecta drift en las features de entrada
        
        Args:
            current_features (pd.DataFrame): Features actuales
            reference_features (pd.DataFrame): Features de referencia (baseline)
        
        Returns:
            Dict: Resultados del análisis de drift
        """
        print("🔍 Analizando drift en datos...")
        
        if reference_features is None:
            print("⚠️  No hay datos de referencia para comparar drift")
            return {'error': 'No reference data available'}
        
        drift_results = {
            'timestamp': datetime.now(),
            'total_features': len(current_features.columns),
            'drift_detected': False,
            'features_with_drift': [],
            'drift_scores': {},
            'statistical_tests': {}
        }
        
        for feature in current_features.columns:
            if feature in reference_features.columns:
                # Test estadístico (Kolmogorov-Smirnov)
                drift_score = self._calculate_feature_drift(
                    current_features[feature], 
                    reference_features[feature]
                )
                
                drift_results['drift_scores'][feature] = drift_score
                
                # Determinar si hay drift significativo
                if drift_score > self.alert_thresholds['data_drift_score']:
                    drift_results['features_with_drift'].append(feature)
                    drift_results['drift_detected'] = True
        
        # Calcular drift score general
        if drift_results['drift_scores']:
            drift_results['overall_drift_score'] = np.mean(list(drift_results['drift_scores'].values()))
        
        # Guardar en historial
        self.data_drift_history.append(drift_results)
        
        if drift_results['drift_detected']:
            print(f"⚠️  Drift detectado en {len(drift_results['features_with_drift'])} features")
        else:
            print("✅ No se detectó drift significativo")
        
        return drift_results
    
    def _calculate_feature_drift(self, current_data: pd.Series, 
                               reference_data: pd.Series) -> float:
        """Calcula score de drift para una feature específica"""
        from scipy import stats
        
        # Remover valores nulos
        current_clean = current_data.dropna()
        reference_clean = reference_data.dropna()
        
        if len(current_clean) == 0 or len(reference_clean) == 0:
            return 0.0
        
        # Test Kolmogorov-Smirnov
        try:
            ks_statistic, ks_p_value = stats.ks_2samp(current_clean, reference_clean)
            return ks_statistic
        except:
            # Fallback: diferencia en medias normalizadas
            current_mean = current_clean.mean()
            reference_mean = reference_clean.mean()
            reference_std = reference_clean.std()
            
            if reference_std == 0:
                return 0.0
            
            return abs(current_mean - reference_mean) / reference_std
    
    def check_performance_degradation(self, model_name: str = 'default') -> Dict:
        """
        Verifica degradación en el performance del modelo
        
        Args:
            model_name (str): Nombre del modelo a verificar
        
        Returns:
            Dict: Análisis de degradación
        """
        print(f"📉 Verificando degradación de performance para {model_name}...")
        
        # Filtrar historial por modelo
        model_history = [
            record for record in self.prediction_history 
            if record.get('model_name') == model_name and 'accuracy' in record
        ]
        
        if len(model_history) < 2:
            return {'error': 'Insufficient historical data for degradation analysis'}
        
        # Obtener métricas baseline
        baseline_accuracy = self.baseline_metrics.get(model_name, {}).get('accuracy', 0)
        baseline_f1 = self.baseline_metrics.get(model_name, {}).get('f1_macro', 0)
        
        # Métricas recientes (última semana)
        recent_window = datetime.now() - timedelta(days=self.config['monitoring_window_days'])
        recent_records = [
            record for record in model_history 
            if record['timestamp'] > recent_window
        ]
        
        if not recent_records:
            return {'error': 'No recent performance data available'}
        
        # Calcular métricas promedio recientes
        recent_accuracy = np.mean([r['accuracy'] for r in recent_records])
        recent_f1 = np.mean([r['f1_macro'] for r in recent_records])
        recent_confidence = np.mean([r['avg_confidence'] for r in recent_records if r['avg_confidence']])
        
        # Calcular degradación
        accuracy_drop = baseline_accuracy - recent_accuracy
        f1_drop = baseline_f1 - recent_f1
        
        degradation_analysis = {
            'model_name': model_name,
            'timestamp': datetime.now(),
            'baseline_accuracy': baseline_accuracy,
            'recent_accuracy': recent_accuracy,
            'accuracy_drop': accuracy_drop,
            'baseline_f1': baseline_f1,
            'recent_f1': recent_f1,
            'f1_drop': f1_drop,
            'recent_avg_confidence': recent_confidence,
            'degradation_detected': False,
            'alert_level': 'normal'
        }
        
        # Determinar nivel de alerta
        if accuracy_drop > self.alert_thresholds['accuracy_drop']:
            degradation_analysis['degradation_detected'] = True
            if accuracy_drop > self.alert_thresholds['accuracy_drop'] * 2:
                degradation_analysis['alert_level'] = 'critical'
            else:
                degradation_analysis['alert_level'] = 'warning'
        
        # Guardar en historial
        self.performance_history.append(degradation_analysis)
        
        if degradation_analysis['degradation_detected']:
            print(f"⚠️  Degradación detectada: Accuracy bajó {accuracy_drop:.3f}")
        else:
            print("✅ Performance estable")
        
        return degradation_analysis
    
    def generate_monitoring_alerts(self) -> List[Dict]:
        """Genera alertas basadas en el monitoreo"""
        alerts = []
        current_time = datetime.now()
        
        # Alertas de degradación de performance
        if self.performance_history:
            latest_performance = self.performance_history[-1]
            if latest_performance.get('degradation_detected'):
                alerts.append({
                    'type': 'performance_degradation',
                    'severity': latest_performance.get('alert_level', 'warning'),
                    'message': f"Performance degradation detected for {latest_performance['model_name']}",
                    'details': latest_performance,
                    'timestamp': current_time
                })
        
        # Alertas de drift de datos
        if self.data_drift_history:
            latest_drift = self.data_drift_history[-1]
            if latest_drift.get('drift_detected'):
                alerts.append({
                    'type': 'data_drift',
                    'severity': 'warning',
                    'message': f"Data drift detected in {len(latest_drift['features_with_drift'])} features",
                    'details': latest_drift,
                    'timestamp': current_time
                })
        
        # Alertas de confianza baja
        if self.prediction_history:
            latest_predictions = self.prediction_history[-1]
            if latest_predictions.get('avg_confidence') and latest_predictions['avg_confidence'] < 0.6:
                alerts.append({
                    'type': 'low_confidence',
                    'severity': 'warning',
                    'message': f"Low average prediction confidence: {latest_predictions['avg_confidence']:.3f}",
                    'details': latest_predictions,
                    'timestamp': current_time
                })
        
        return alerts
    
    def create_monitoring_dashboard_data(self) -> Dict:
        """Crea datos para dashboard de monitoreo"""
        dashboard_data = {
            'summary': {
                'total_predictions_logged': len(self.prediction_history),
                'data_drift_checks': len(self.data_drift_history),
                'performance_checks': len(self.performance_history),
                'active_alerts': len(self.generate_monitoring_alerts())
            },
            'recent_performance': self.performance_history[-5:] if self.performance_history else [],
            'recent_predictions': self.prediction_history[-10:] if self.prediction_history else [],
            'drift_summary': self.data_drift_history[-3:] if self.data_drift_history else [],
            'alerts': self.generate_monitoring_alerts()
        }
        
        return dashboard_data
    
    def _save_monitoring_log(self, data: Dict, log_type: str) -> None:
        """Guarda logs de monitoreo en archivos"""
        logs_dir = Path(self.config.get('monitoring_data_path', 'monitoring/performance_logs'))
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Crear nombre de archivo con timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{log_type}_{timestamp}.json"
        
        # Convertir datetime objects a strings para JSON
        data_serializable = self._make_json_serializable(data)
        
        # Guardar archivo
        with open(logs_dir / filename, 'w') as f:
            json.dump(data_serializable, f, indent=2)
    
    def _make_json_serializable(self, obj):
        """Convierte objetos a formato serializable para JSON"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj
    
    def run_full_monitoring_cycle(self, current_data: pd.DataFrame,
                                 ground_truth: pd.DataFrame = None,
                                 reference_features: pd.DataFrame = None,
                                 model_name: str = 'default') -> Dict:
        """
        Ejecuta un ciclo completo de monitoreo
        
        Args:
            current_data (pd.DataFrame): Datos actuales con predicciones
            ground_truth (pd.DataFrame): Datos reales para validación
            reference_features (pd.DataFrame): Features de referencia para drift
            model_name (str): Nombre del modelo
        
        Returns:
            Dict: Resumen completo del monitoreo
        """
        print("🔄 Ejecutando ciclo completo de monitoreo...")
        
        monitoring_results = {
            'timestamp': datetime.now(),
            'model_name': model_name,
            'data_processed': len(current_data),
            'monitoring_status': 'completed'
        }
        
        try:
            # 1. Registrar predicciones
            self.log_predictions(current_data, ground_truth, model_name)
            
            # 2. Verificar drift de datos
            if reference_features is not None:
                # Extraer features de current_data
                feature_columns = [col for col in reference_features.columns if col in current_data.columns]
                current_features = current_data[feature_columns]
                
                drift_results = self.monitor_data_drift(current_features, reference_features)
                monitoring_results['data_drift'] = drift_results
            
            # 3. Verificar degradación de performance
            performance_results = self.check_performance_degradation(model_name)
            monitoring_results['performance_check'] = performance_results
            
            # 4. Generar alertas
            alerts = self.generate_monitoring_alerts()
            monitoring_results['alerts'] = alerts
            
            # 5. Crear datos de dashboard
            dashboard_data = self.create_monitoring_dashboard_data()
            monitoring_results['dashboard_data'] = dashboard_data
            
            print(f"✅ Monitoreo completado - {len(alerts)} alertas generadas")
            
        except Exception as e:
            monitoring_results['monitoring_status'] = 'error'
            monitoring_results['error'] = str(e)
            print(f"❌ Error en monitoreo: {e}")
        
        return monitoring_results
    
    def generate_monitoring_report(self) -> str:
        """Genera reporte de monitoreo en formato texto"""
        current_time = datetime.now()
        alerts = self.generate_monitoring_alerts()
        
        report = f"""
🎵 TUNEMETRICS - REPORTE DE MONITOREO DE PERFORMANCE 🎵
{'='*70}

📊 RESUMEN GENERAL:
├── Timestamp: {current_time.strftime('%Y-%m-%d %H:%M:%S')}
├── Predicciones registradas: {len(self.prediction_history)}
├── Verificaciones de drift: {len(self.data_drift_history)}
├── Verificaciones de performance: {len(self.performance_history)}
└── Alertas activas: {len(alerts)}

🚨 ALERTAS ACTIVAS:
"""
        
        if alerts:
            for alert in alerts:
                severity_icon = "🔴" if alert['severity'] == 'critical' else "🟡"
                report += f"""
{severity_icon} {alert['type'].upper()}:
  ├── Severidad: {alert['severity']}
  ├── Mensaje: {alert['message']}
  └── Timestamp: {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"""
        else:
            report += "\n✅ No hay alertas activas"
        
        # Performance reciente
        if self.performance_history:
            latest_perf = self.performance_history[-1]
            report += f"""

📈 PERFORMANCE RECIENTE:
├── Modelo: {latest_perf.get('model_name', 'N/A')}
├── Accuracy baseline: {latest_perf.get('baseline_accuracy', 'N/A'):.4f}
├── Accuracy reciente: {latest_perf.get('recent_accuracy', 'N/A'):.4f}
├── Drop en accuracy: {latest_perf.get('accuracy_drop', 'N/A'):.4f}
└── Estado: {'⚠️ Degradación detectada' if latest_perf.get('degradation_detected') else '✅ Estable'}"""
        
        # Drift de datos
        if self.data_drift_history:
            latest_drift = self.data_drift_history[-1]
            report += f"""

🔍 ANÁLISIS DE DRIFT:
├── Features analizadas: {latest_drift.get('total_features', 'N/A')}
├── Features con drift: {len(latest_drift.get('features_with_drift', []))}
├── Score de drift general: {latest_drift.get('overall_drift_score', 'N/A'):.4f}
└── Estado: {'⚠️ Drift detectado' if latest_drift.get('drift_detected') else '✅ Sin drift significativo'}"""
        
        report += f"""

📝 RECOMENDACIONES:
"""
        
        # Generar recomendaciones basadas en alertas
        recommendations = []
        
        for alert in alerts:
            if alert['type'] == 'performance_degradation':
                recommendations.append("• Considerar reentrenamiento del modelo")
                recommendations.append("• Verificar calidad de datos de entrada")
            elif alert['type'] == 'data_drift':
                recommendations.append("• Investigar cambios en fuente de datos")
                recommendations.append("• Evaluar necesidad de actualización de modelo")
            elif alert['type'] == 'low_confidence':
                recommendations.append("• Revisar thresholds de confianza")
                recommendations.append("• Analizar casos de baja confianza")
        
        if not recommendations:
            recommendations.append("• Continuar monitoreo regular")
            recommendations.append("• Mantener pipeline de datos actualizado")
        
        for rec in set(recommendations):  # Eliminar duplicados
            report += f"\n{rec}"
        
        return report

def main():
    """Función principal para demostrar el monitor de performance"""
    
    try:
        # Inicializar monitor
        monitor = TuneMetricsPerformanceMonitor("models")
        
        print("📊 Monitor de Performance TuneMetrics inicializado")
        
        # Simular datos de predicciones
        print("\n🎵 Simulando monitoreo de predicciones...")
        
        sample_predictions = pd.DataFrame({
            'spotify_track_uri': [f'track_{i}' for i in range(100)],
            'predicted_engagement': np.random.choice(['High', 'Medium', 'Low'], 100),
            'confidence': np.random.uniform(0.3, 0.95, 100)
        })
        
        sample_ground_truth = pd.DataFrame({
            'spotify_track_uri': [f'track_{i}' for i in range(100)],
            'actual_engagement': np.random.choice(['High', 'Medium', 'Low'], 100)
        })
        
        # Registrar predicciones
        monitor.log_predictions(sample_predictions, sample_ground_truth, 'random_forest')
        
        # Simular features para análisis de drift
        print("\n🔍 Simulando análisis de drift...")
        
        reference_features = pd.DataFrame({
            'completion_rate_score': np.random.normal(0.6, 0.2, 1000),
            'skip_resistance_score': np.random.normal(0.8, 0.15, 1000),
            'context_preference_score': np.random.normal(0.4, 0.3, 1000)
        })
        
        # Simular drift leve en current features
        current_features = pd.DataFrame({
            'completion_rate_score': np.random.normal(0.55, 0.2, 200),  # Drift leve
            'skip_resistance_score': np.random.normal(0.8, 0.15, 200),  # Sin drift
            'context_preference_score': np.random.normal(0.35, 0.3, 200)  # Drift leve
        })
        
        drift_results = monitor.monitor_data_drift(current_features, reference_features)
        
        # Verificar degradación de performance
        print("\n📉 Verificando degradación de performance...")
        degradation_results = monitor.check_performance_degradation('random_forest')
        
        # Generar alertas
        print("\n🚨 Generando alertas...")
        alerts = monitor.generate_monitoring_alerts()
        print(f"Alertas generadas: {len(alerts)}")
        
        # Crear datos de dashboard
        dashboard_data = monitor.create_monitoring_dashboard_data()
        print(f"Datos de dashboard creados con {len(dashboard_data)} secciones")
        
        # Generar reporte
        report = monitor.generate_monitoring_report()
        print("\n" + "="*50)
        print("REPORTE DE MONITOREO:")
        print("="*50)
        print(report)
        
        print("\n✅ Monitor de performance funcionando correctamente!")
        
    except Exception as e:
        print(f"❌ Error en monitor de performance: {e}")
        raise

if __name__ == "__main__":
    main()