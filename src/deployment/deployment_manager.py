# Manager principal para deployment y monitoreo de TuneMetrics

import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Importar módulos propios
try:
    from src.models.model_evaluator import TuneMetricsModelEvaluator
    from src.deployment.inference_pipeline import TuneMetricsInferencePipeline
    from src.deployment.performance_monitor import TuneMetricsPerformanceMonitor
except ImportError:
    # Para ejecución standalone
    import sys
    sys.path.append('.')
    print("Ejecutando en modo standalone")

class TuneMetricsDeploymentManager:
    """
    Manager principal para deployment de TuneMetrics
    Integra evaluación, inferencia y monitoreo
    """
    
    def __init__(self, config_path: str = None):
        """
        Inicializa el deployment manager
        
        Args:
            config_path (str): Ruta al archivo de configuración
        """
        self.config = self._load_config(config_path)
        self.models_dir = self.config.get('models_dir', 'models')
        self.data_dir = self.config.get('data_dir', 'data/processed')
        self.output_dir = self.config.get('output_dir', 'reports')
        
        # Inicializar componentes
        self.evaluator = None
        self.inference_pipeline = None
        self.performance_monitor = None
        
        # Estado del deployment
        self.deployment_status = {
            'initialized': False,
            'models_loaded': False,
            'monitoring_active': False,
            'last_health_check': None
        }
        
    def _load_config(self, config_path: str) -> Dict:
        """Carga configuración desde archivo o usa defaults"""
        default_config = {
            'models_dir': 'models',
            'data_dir': 'data/processed',
            'output_dir': 'reports',
            'monitoring': {
                'enabled': True,
                'monitoring_window_days': 7,
                'alert_thresholds': {
                    'accuracy_drop': 0.05,
                    'confidence_drop': 0.10,
                    'data_drift_score': 0.15
                }
            },
            'inference': {
                'default_model': 'best',
                'batch_size': 1000,
                'confidence_threshold': 0.6
            },
            'evaluation': {
                'test_data_split': 'test',
                'create_visualizations': True,
                'save_detailed_results': True
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            # Merge configs
            default_config.update(user_config)
        
        return default_config
    
    def initialize_deployment(self) -> bool:
        """Inicializa todos los componentes del deployment"""
        print("Iniciando deployment de TuneMetrics...")
        
        try:
            # 1. Inicializar evaluador
            print("  Inicializando evaluador de modelos...")
            self.evaluator = TuneMetricsModelEvaluator(self.models_dir)
            self.evaluator.load_trained_models()
            
            # 2. Inicializar pipeline de inferencia
            print("  Inicializando pipeline de inferencia...")
            default_model = self.config.get('inference', {}).get('default_model', 'best')
            self.inference_pipeline = TuneMetricsInferencePipeline(
                self.models_dir, 
                default_model=default_model
            )
            
            # 3. Inicializar monitor de performance
            if self.config.get('monitoring', {}).get('enabled', True):
                print("  Inicializando monitor de performance...")
                self.performance_monitor = TuneMetricsPerformanceMonitor(
                    self.models_dir, 
                    self.config.get('monitoring', {})
                )
                self.deployment_status['monitoring_active'] = True
            
            # 4. Verificar estado de modelos
            model_info = self.inference_pipeline.get_model_info()
            print(f"  Modelos disponibles: {model_info['available_models']}")
            print(f"  Modelo por defecto: {model_info['default_model']}")
            
            self.deployment_status['initialized'] = True
            self.deployment_status['models_loaded'] = True
            self.deployment_status['last_health_check'] = datetime.now()
            
            print("Deployment inicializado exitosamente")
            return True
            
        except Exception as e:
            print(f"Error inicializando deployment: {e}")
            return False
    
    def run_comprehensive_evaluation(self, test_data_path: str = None) -> Dict:
        """
        Ejecuta evaluación comprehensiva de modelos
        
        Args:
            test_data_path (str): Ruta a datos de test
        
        Returns:
            Dict: Resultados de evaluación
        """
        print("\nEjecutando evaluación comprehensiva...")
        
        if not self.evaluator:
            raise RuntimeError("Evaluador no inicializado")
        
        # Cargar datos de test
        if test_data_path is None:
            test_data_path = Path(self.data_dir) / "gold_data.parquet"
        
        print(f"  Cargando datos de test: {test_data_path}")
        
        if str(test_data_path).endswith('.parquet'):
            test_data = pd.read_parquet(test_data_path)
        else:
            test_data = pd.read_csv(test_data_path)
        
        # Filtrar datos de test
        test_split = self.config.get('evaluation', {}).get('test_data_split', 'test')
        if 'data_split' in test_data.columns:
            test_data = test_data[test_data['data_split'] == test_split]
        
        print(f"  Evaluando con {len(test_data)} canciones")
        
        # Ejecutar evaluación
        evaluation_results = self.evaluator.evaluate_on_new_data(test_data)
        
        # Análisis adicionales
        feature_importance = self.evaluator.analyze_feature_importance()
        confidence_analysis = self.evaluator.analyze_prediction_confidence()
        model_comparison = self.evaluator.compare_models_performance()
        
        # Crear visualizaciones si está habilitado
        if self.config.get('evaluation', {}).get('create_visualizations', True):
            viz_dir = Path(self.output_dir) / "evaluation_visualizations"
            self.evaluator.create_evaluation_visualizations(str(viz_dir))
        
        # Generar reporte
        report = self.evaluator.generate_evaluation_report()
        
        # Guardar resultados
        results_dir = Path(self.output_dir) / "evaluation_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Guardar reporte
        with open(results_dir / f"evaluation_report_{timestamp}.txt", 'w') as f:
            f.write(report)
        
        # Guardar resultados detallados
        if self.config.get('evaluation', {}).get('save_detailed_results', True):
            detailed_results = {
                'evaluation_results': evaluation_results,
                'feature_importance': feature_importance,
                'confidence_analysis': confidence_analysis,
                'model_comparison': model_comparison.to_dict() if model_comparison is not None else None,
                'timestamp': timestamp
            }
            
            with open(results_dir / f"detailed_results_{timestamp}.json", 'w') as f:
                json.dump(self._make_json_serializable(detailed_results), f, indent=2)
        
        print(f"Evaluación completada y guardada en {results_dir}")
        
        return {
            'evaluation_results': evaluation_results,
            'feature_importance': feature_importance,
            'confidence_analysis': confidence_analysis,
            'model_comparison': model_comparison,
            'report': report
        }
    
    def predict_engagement_batch(self, input_data: pd.DataFrame,
                                data_type: str = 'processed') -> pd.DataFrame:
        """
        Realiza predicciones de engagement en lote
        
        Args:
            input_data (pd.DataFrame): Datos de entrada
            data_type (str): 'processed' (features ya calculadas) o 'raw' (datos de Spotify)
        
        Returns:
            pd.DataFrame: Predicciones con insights de negocio
        """
        print(f"\nRealizando predicciones de engagement para {len(input_data)} elementos...")
        
        if not self.inference_pipeline:
            raise RuntimeError("Pipeline de inferencia no inicializado")
        
        # Validar datos de entrada
        is_valid, errors = self.inference_pipeline.validate_input_data(input_data)
        if not is_valid:
            raise ValueError(f"Datos de entrada inválidos: {errors}")
        
        # Realizar predicciones según el tipo de datos
        if data_type == 'raw':
            print("  Procesando datos raw de Spotify...")
            predictions = self.inference_pipeline.predict_from_spotify_data(input_data)
        else:
            print("  Usando features procesadas...")
            predictions = self.inference_pipeline.predict_batch(
                input_data, 
                include_features=True
            )
        
        # Registrar predicciones en monitor si está activo
        if self.performance_monitor:
            self.performance_monitor.log_predictions(predictions)
        
        print(f"Predicciones completadas para {len(predictions)} canciones")
        
        return predictions
    
    def run_monitoring_cycle(self, current_data: pd.DataFrame,
                           ground_truth: pd.DataFrame = None) -> Dict:
        """
        Ejecuta ciclo completo de monitoreo
        
        Args:
            current_data (pd.DataFrame): Datos actuales con predicciones
            ground_truth (pd.DataFrame): Datos reales para validación
        
        Returns:
            Dict: Resultados del monitoreo
        """
        print("\nEjecutando ciclo de monitoreo...")
        
        if not self.performance_monitor:
            raise RuntimeError("Monitor de performance no inicializado")
        
        # Cargar datos de referencia para drift analysis
        reference_path = Path(self.data_dir) / "gold_data.parquet"
        if reference_path.exists():
            if str(reference_path).endswith('.parquet'):
                reference_data = pd.read_parquet(reference_path)
            else:
                reference_data = pd.read_csv(reference_path)
            
            # Filtrar datos de entrenamiento como referencia
            if 'data_split' in reference_data.columns:
                reference_features = reference_data[reference_data['data_split'] == 'train']
            else:
                reference_features = reference_data.sample(frac=0.8)  # 80% como referencia
        else:
            print("  No se encontraron datos de referencia para análisis de drift")
            reference_features = None
        
        # Ejecutar monitoreo completo
        monitoring_results = self.performance_monitor.run_full_monitoring_cycle(
            current_data=current_data,
            ground_truth=ground_truth,
            reference_features=reference_features
        )
        
        # Generar reporte de monitoreo
        monitoring_report = self.performance_monitor.generate_monitoring_report()
        
        # Guardar resultados
        monitoring_dir = Path(self.output_dir) / "monitoring"
        monitoring_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        with open(monitoring_dir / f"monitoring_report_{timestamp}.txt", 'w') as f:
            f.write(monitoring_report)
        
        with open(monitoring_dir / f"monitoring_results_{timestamp}.json", 'w') as f:
            json.dump(self._make_json_serializable(monitoring_results), f, indent=2)
        
        print(f"Monitoreo completado y guardado en {monitoring_dir}")
        
        return monitoring_results
    
    def health_check(self) -> Dict:
        """Verifica el estado de salud del deployment"""
        print("\nEjecutando health check...")
        
        health_status = {
            'timestamp': datetime.now(),
            'overall_status': 'healthy',
            'components': {},
            'alerts': [],
            'recommendations': []
        }
        
        # Verificar evaluador
        if self.evaluator:
            try:
                model_count = len(self.evaluator.models)
                health_status['components']['evaluator'] = {
                    'status': 'healthy',
                    'models_loaded': model_count,
                    'details': f"{model_count} modelos cargados"
                }
            except Exception as e:
                health_status['components']['evaluator'] = {
                    'status': 'error',
                    'error': str(e)
                }
                health_status['overall_status'] = 'unhealthy'
        
        # Verificar pipeline de inferencia
        if self.inference_pipeline:
            try:
                model_info = self.inference_pipeline.get_model_info()
                health_status['components']['inference_pipeline'] = {
                    'status': 'healthy',
                    'default_model': model_info['default_model'],
                    'available_models': model_info['available_models'],
                    'details': f"Pipeline listo con {len(model_info['available_models'])} modelos"
                }
            except Exception as e:
                health_status['components']['inference_pipeline'] = {
                    'status': 'error',
                    'error': str(e)
                }
                health_status['overall_status'] = 'unhealthy'
        
        # Verificar monitor de performance
        if self.performance_monitor:
            try:
                alerts = self.performance_monitor.generate_monitoring_alerts()
                health_status['components']['performance_monitor'] = {
                    'status': 'healthy' if len(alerts) == 0 else 'warning',
                    'active_alerts': len(alerts),
                    'details': f"Monitor activo con {len(alerts)} alertas"
                }
                health_status['alerts'].extend(alerts)
            except Exception as e:
                health_status['components']['performance_monitor'] = {
                    'status': 'error',
                    'error': str(e)
                }
                health_status['overall_status'] = 'unhealthy'
        
        # Generar recomendaciones
        if health_status['overall_status'] == 'unhealthy':
            health_status['recommendations'].append("Revisar componentes con errores")
        if len(health_status['alerts']) > 0:
            health_status['recommendations'].append("Revisar alertas de monitoreo")
        if not health_status['recommendations']:
            health_status['recommendations'].append("Sistema funcionando correctamente")
        
        # Actualizar timestamp de health check
        self.deployment_status['last_health_check'] = datetime.now()
        
        print(f"Health check completado - Estado: {health_status['overall_status']}")
        
        return health_status
    
    def _make_json_serializable(self, obj):
        """Convierte objetos a formato serializable para JSON"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj
    
    def get_deployment_status(self) -> Dict:
        """Retorna estado actual del deployment"""
        return {
            'deployment_status': self.deployment_status,
            'config': self.config,
            'models_dir': str(self.models_dir),
            'data_dir': str(self.data_dir),
            'output_dir': str(self.output_dir)
        }

def main():
    """Función principal con CLI para deployment manager"""
    parser = argparse.ArgumentParser(description='TuneMetrics Deployment Manager')
    parser.add_argument('--config', type=str, help='Ruta al archivo de configuración')
    parser.add_argument('--action', type=str, required=True,
                       choices=['init', 'evaluate', 'predict', 'monitor', 'health'],
                       help='Acción a ejecutar')
    parser.add_argument('--input-data', type=str, help='Ruta a datos de entrada')
    parser.add_argument('--data-type', type=str, choices=['raw', 'processed'], 
                       default='processed', help='Tipo de datos de entrada')
    parser.add_argument('--output-file', type=str, help='Archivo de salida para predicciones')
    
    args = parser.parse_args()
    
    try:
        # Crear deployment manager
        manager = TuneMetricsDeploymentManager(args.config)
        
        if args.action == 'init':
            print("Inicializando deployment...")
            success = manager.initialize_deployment()
            if success:
                print("Deployment inicializado exitosamente")
            else:
                print("Error inicializando deployment")
                return 1
        
        elif args.action == 'evaluate':
            manager.initialize_deployment()
            print("Ejecutando evaluación comprehensiva...")
            results = manager.run_comprehensive_evaluation(args.input_data)
            print("Evaluación completada")
        
        elif args.action == 'predict':
            if not args.input_data:
                print("Se requiere --input-data para predicciones")
                return 1
            
            manager.initialize_deployment()
            
            # Cargar datos
            if args.input_data.endswith('.parquet'):
                input_data = pd.read_parquet(args.input_data)
            else:
                input_data = pd.read_csv(args.input_data)
            
            # Realizar predicciones
            predictions = manager.predict_engagement_batch(input_data, args.data_type)
            
            # Guardar resultados
            output_file = args.output_file or f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            predictions.to_csv(output_file, index=False)
            print(f"Predicciones guardadas en {output_file}")
        
        elif args.action == 'monitor':
            if not args.input_data:
                print("Se requiere --input-data para monitoreo")
                return 1
            
            manager.initialize_deployment()
            
            # Cargar datos
            if args.input_data.endswith('.parquet'):
                current_data = pd.read_parquet(args.input_data)
            else:
                current_data = pd.read_csv(args.input_data)
            
            # Ejecutar monitoreo
            monitoring_results = manager.run_monitoring_cycle(current_data)
            print("Monitoreo completado")
        
        elif args.action == 'health':
            manager.initialize_deployment()
            health_status = manager.health_check()
            
            print(f"\nESTADO DE SALUD DEL DEPLOYMENT:")
            print(f"Estado general: {health_status['overall_status']}")
            print(f"Alertas activas: {len(health_status['alerts'])}")
            
            for component, status in health_status['components'].items():
                print(f"  {component}: {status['status']}")
        
        return 0
        
    except Exception as e:
        print(f"Error ejecutando deployment manager: {e}")
        return 1

if __name__ == "__main__":
    exit(main())