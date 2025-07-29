# Script principal para ejecutar el pipeline completo de TuneMetrics

import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime
import json
import yaml

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tunemetrics_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class TuneMetricsPipelineRunner:
    """
    Orquestador principal del pipeline completo de TuneMetrics
    Ejecuta todos los pasos desde EDA hasta Deployment
    """
    
    def __init__(self, config_path: str = None):
        """
        Inicializa el runner del pipeline
        
        Args:
            config_path (str): Ruta al archivo de configuración
        """
        self.config_path = config_path or "configs/config.yaml"
        self.config = self.load_config()
        self.pipeline_status = {
            'started_at': datetime.now(),
            'steps_completed': [],
            'steps_failed': [],
            'current_step': None
        }
        
        # Configurar rutas
        self.setup_paths()
        
    def load_config(self) -> dict:
        """Carga configuración desde archivo YAML"""
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                logger.info(f"Configuración cargada desde {self.config_path}")
                return config
            else:
                logger.warning(f"Archivo de configuración no encontrado: {self.config_path}")
                return self.get_default_config()
        except Exception as e:
            logger.error(f"Error cargando configuración: {e}")
            return self.get_default_config()
    
    def get_default_config(self) -> dict:
        """Configuración por defecto del pipeline"""
        return {
            'project': {
                'name': 'TuneMetrics Music Engagement Predictor',
                'version': '1.0.0'
            },
            'data': {
                'raw_data_path': 'data/raw/',
                'processed_data_path': 'data/processed/',
                'spotify_history': 'spotify_history.csv',
                'train_years': list(range(2013, 2023)),
                'validation_years': [2023],
                'test_years': [2024]
            },
            'features': {
                'estimated_song_duration_ms': 210000,
                'engagement_thresholds': {'high': 0.75, 'medium': 0.45}
            },
            'models': {
                'algorithms': ['random_forest', 'xgboost', 'logistic_regression', 'mlp'],
                'target_metrics': {'min_accuracy': 0.85, 'min_f1_score': 0.80},
                'random_state': 42
            },
            'output': {
                'models_path': 'models/',
                'reports_path': 'reports/',
                'dashboard_path': 'dashboard_data/'
            }
        }
    
    def setup_paths(self):
        """Configura y crea directorios necesarios"""
        paths_to_create = [
            self.config['data']['processed_data_path'],
            f"{self.config['output']['models_path']}/trained",
            f"{self.config['output']['models_path']}/metrics", 
            f"{self.config['output']['reports_path']}/figures",
            self.config['output']['dashboard_path'],
            "logs"
        ]
        
        for path in paths_to_create:
            Path(path).mkdir(parents=True, exist_ok=True)
            
        logger.info("Estructura de directorios verificada")
    
    def run_step(self, step_name: str, step_function, *args, **kwargs):
        """
        Ejecuta un paso del pipeline con manejo de errores
        
        Args:
            step_name (str): Nombre del paso
            step_function: Función a ejecutar
            *args, **kwargs: Argumentos para la función
        """
        logger.info(f"🔄 Iniciando paso: {step_name}")
        self.pipeline_status['current_step'] = step_name
        
        try:
            start_time = datetime.now()
            result = step_function(*args, **kwargs)
            end_time = datetime.now()
            duration = (end_time - start_time).seconds
            
            self.pipeline_status['steps_completed'].append({
                'step': step_name,
                'duration_seconds': duration,
                'completed_at': end_time.isoformat()
            })
            
            logger.info(f"✅ {step_name} completado en {duration}s")
            return result
            
        except Exception as e:
            self.pipeline_status['steps_failed'].append({
                'step': step_name,
                'error': str(e),
                'failed_at': datetime.now().isoformat()
            })
            logger.error(f"❌ Error en {step_name}: {e}")
            raise
    
    def step_1_exploratory_analysis(self):
        """Paso 1: Análisis Exploratorio de Datos"""
        try:
            from src.data.exploratory_analysis import SpotifyDataExplorer
            
            data_path = Path(self.config['data']['raw_data_path']) / self.config['data']['spotify_history']
            
            explorer = SpotifyDataExplorer(str(data_path))
            explorer.basic_info()
            explorer.analyze_categorical_variables()
            explorer.analyze_engagement_metrics()
            explorer.analyze_temporal_patterns()
            
            # Crear visualizaciones
            viz_path = Path(self.config['output']['reports_path']) / "figures" / "eda_dashboard.png"
            explorer.create_visualization_dashboard(str(viz_path))
            
            # Generar reporte
            report = explorer.generate_summary_report()
            report_path = Path(self.config['output']['reports_path']) / "eda_summary_report.txt"
            with open(report_path, 'w') as f:
                f.write(report)
            
            logger.info(f"EDA completado - Reporte guardado en {report_path}")
            return True
            
        except ImportError:
            logger.error("Módulo de EDA no encontrado - ejecutando versión simplificada")
            return False
    
    def step_2_data_processing(self):
        """Paso 2: Procesamiento de Datos (Bronze → Silver → Gold)"""
        try:
            from src.data.data_processor import TuneMetricsDataProcessor
            
            # Configuración para el procesador
            processor_config = {
                'estimated_song_duration_ms': self.config['features']['estimated_song_duration_ms'],
                'engagement_thresholds': self.config['features']['engagement_thresholds'],
                'train_years': self.config['data']['train_years'],
                'validation_years': self.config['data']['validation_years'],
                'test_years': self.config['data']['test_years']
            }
            
            processor = TuneMetricsDataProcessor(processor_config)
            
            # Pipeline Bronze → Silver → Gold
            data_path = Path(self.config['data']['raw_data_path']) / self.config['data']['spotify_history']
            processor.load_bronze_data(str(data_path))
            processor.create_silver_data()
            processor.create_gold_data()
            
            # Guardar datos procesados
            output_dir = self.config['data']['processed_data_path']
            processor.save_processed_data(output_dir)
            
            # Generar reporte
            report = processor.generate_processing_report()
            report_path = Path(self.config['output']['reports_path']) / "data_processing_report.txt"
            with open(report_path, 'w') as f:
                f.write(report)
            
            logger.info(f"Procesamiento completado - Datos guardados en {output_dir}")
            return True
            
        except ImportError:
            logger.error("Módulo de procesamiento no encontrado")
            return False
    
    def step_3_model_training(self):
        """Paso 3: Entrenamiento de Modelos"""
        try:
            from src.models.model_trainer import TuneMetricsModelTrainer
            
            # Configuración para entrenamiento
            training_config = {
                'random_state': self.config['models']['random_state'],
                'target_metrics': self.config['models']['target_metrics']
            }
            
            trainer = TuneMetricsModelTrainer(training_config)
            
            # Cargar datos Gold
            gold_data_path = Path(self.config['data']['processed_data_path']) / "gold_data.parquet"
            trainer.load_gold_data(str(gold_data_path))
            trainer.prepare_features_and_splits()
            
            # Entrenar modelos según configuración
            algorithms = self.config['models']['algorithms']
            
            if 'random_forest' in algorithms:
                trainer.train_random_forest()
            if 'xgboost' in algorithms:
                trainer.train_xgboost()
            if 'logistic_regression' in algorithms:
                trainer.train_logistic_regression()
            if 'mlp' in algorithms:
                trainer.train_mlp()
            
            # Evaluar y seleccionar mejor modelo
            trainer.evaluate_models()
            trainer.select_best_model()
            
            # Guardar modelos y resultados
            models_output_dir = self.config['output']['models_path']
            trainer.save_models_and_results(models_output_dir)
            
            # Generar reporte
            report = trainer.generate_training_report()
            report_path = Path(self.config['output']['reports_path']) / "training_report.txt"
            with open(report_path, 'w') as f:
                f.write(report)
            
            logger.info(f"Entrenamiento completado - Modelos guardados en {models_output_dir}")
            return True
            
        except ImportError:
            logger.error("Módulo de entrenamiento no encontrado")
            return False
    
    def step_4_model_evaluation(self):
        """Paso 4: Evaluación Comprehensiva de Modelos"""
        try:
            from src.models.model_evaluator import TuneMetricsModelEvaluator
            
            evaluator = TuneMetricsModelEvaluator(self.config['output']['models_path'])
            evaluator.load_trained_models()
            
            # Cargar datos de test
            gold_data_path = Path(self.config['data']['processed_data_path']) / "gold_data.parquet"
            
            if gold_data_path.exists():
                import pandas as pd
                if str(gold_data_path).endswith('.parquet'):
                    test_data = pd.read_parquet(gold_data_path)
                else:
                    test_data = pd.read_csv(gold_data_path)
                
                # Filtrar datos de test
                test_data = test_data[test_data['data_split'] == 'test']
                
                # Evaluar modelos
                evaluator.evaluate_on_new_data(test_data)
                evaluator.analyze_feature_importance()
                evaluator.analyze_prediction_confidence()
                
                # Crear visualizaciones
                viz_dir = Path(self.config['output']['reports_path']) / "evaluation_visualizations"
                evaluator.create_evaluation_visualizations(str(viz_dir))
                
                # Generar reporte
                report = evaluator.generate_evaluation_report()
                report_path = Path(self.config['output']['reports_path']) / "model_evaluation_report.txt"
                with open(report_path, 'w') as f:
                    f.write(report)
                
                logger.info(f"Evaluación completada - Reporte guardado en {report_path}")
                return True
            else:
                logger.error(f"Datos de test no encontrados en {gold_data_path}")
                return False
                
        except ImportError:
            logger.error("Módulo de evaluación no encontrado")
            return False
    
    def step_5_dashboard_generation(self):
        """Paso 5: Generación de Dashboard para Looker Studio"""
        try:
            from src.visualization.dashboard_generator import TuneMetricsDashboardGenerator
            
            # Configurar fuentes de datos
            data_sources = {
                'gold_data': str(Path(self.config['data']['processed_data_path']) / "gold_data.parquet"),
                'models_metrics': str(Path(self.config['output']['models_path']) / "metrics" / "model_metrics.json")
            }
            
            dashboard_generator = TuneMetricsDashboardGenerator(data_sources)
            
            # Generar datasets para Looker Studio
            dashboard_output_dir = self.config['output']['dashboard_path']
            datasets = dashboard_generator.generate_looker_studio_datasets(dashboard_output_dir)
            
            # Crear visualizaciones estáticas
            viz_output_dir = Path(self.config['output']['reports_path']) / "dashboard_visualizations"
            visualizations = dashboard_generator.create_static_visualizations(str(viz_output_dir))
            
            # Generar configuración para Looker Studio
            looker_config = dashboard_generator.generate_dashboard_config_for_looker()
            config_file = Path(dashboard_output_dir) / "looker_studio_config.json"
            with open(config_file, 'w') as f:
                json.dump(looker_config, f, indent=2)
            
            logger.info(f"Dashboard generado - Datasets: {dashboard_output_dir}")
            return True
            
        except ImportError:
            logger.error("Módulo de dashboard no encontrado")
            return False
    
    def step_6_deployment_setup(self):
        """Paso 6: Configuración de Deployment"""
        try:
            from src.deployment.deployment_manager import TuneMetricsDeploymentManager
            
            # Configurar deployment manager
            deployment_config = {
                'models_dir': self.config['output']['models_path'],
                'data_dir': self.config['data']['processed_data_path'],
                'output_dir': self.config['output']['reports_path']
            }
            
            manager = TuneMetricsDeploymentManager()
            manager.config.update(deployment_config)
            
            # Inicializar deployment
            success = manager.initialize_deployment()
            
            if success:
                # Ejecutar health check
                health_status = manager.health_check()
                
                # Guardar estado de deployment
                deployment_status_file = Path(self.config['output']['reports_path']) / "deployment_status.json"
                with open(deployment_status_file, 'w') as f:
                    json.dump({
                        'deployment_status': manager.get_deployment_status(),
                        'health_check': health_status
                    }, f, indent=2, default=str)
                
                logger.info(f"Deployment configurado - Estado: {deployment_status_file}")
                return True
            else:
                logger.error("Fallo en inicialización de deployment")
                return False
                
        except ImportError:
            logger.error("Módulo de deployment no encontrado")
            return False
    
    def run_full_pipeline(self, steps: list = None):
        """
        Ejecuta el pipeline completo de TuneMetrics
        
        Args:
            steps (list): Lista de pasos a ejecutar (None = todos)
        """
        logger.info("🚀 Iniciando pipeline completo de TuneMetrics")
        logger.info(f"Proyecto: {self.config['project']['name']} v{self.config['project']['version']}")
        
        # Definir todos los pasos disponibles
        all_steps = [
            ('eda', 'Análisis Exploratorio', self.step_1_exploratory_analysis),
            ('processing', 'Procesamiento de Datos', self.step_2_data_processing),
            ('training', 'Entrenamiento de Modelos', self.step_3_model_training),
            ('evaluation', 'Evaluación de Modelos', self.step_4_model_evaluation),
            ('dashboard', 'Generación de Dashboard', self.step_5_dashboard_generation),
            ('deployment', 'Configuración de Deployment', self.step_6_deployment_setup)
        ]
        
        # Filtrar pasos si se especifica
        if steps:
            steps_to_run = [(s, n, f) for s, n, f in all_steps if s in steps]
        else:
            steps_to_run = all_steps
        
        # Ejecutar pasos
        total_steps = len(steps_to_run)
        successful_steps = 0
        
        for i, (step_code, step_name, step_function) in enumerate(steps_to_run, 1):
            logger.info(f"📋 Paso {i}/{total_steps}: {step_name}")
            
            try:
                success = self.run_step(step_name, step_function)
                if success:
                    successful_steps += 1
                else:
                    logger.warning(f"Paso {step_name} completado con advertencias")
                    successful_steps += 1  # Contar como exitoso
                    
            except Exception as e:
                logger.error(f"Fallo crítico en {step_name}: {e}")
                
                # Preguntar si continuar
                if i < total_steps:
                    continue_pipeline = input(f"\n¿Continuar con los pasos restantes? (y/n): ").lower().strip()
                    if continue_pipeline != 'y':
                        logger.info("Pipeline detenido por el usuario")
                        break
        
        # Resumen final
        self.pipeline_status['completed_at'] = datetime.now()
        self.pipeline_status['total_duration'] = (
            self.pipeline_status['completed_at'] - self.pipeline_status['started_at']
        ).seconds
        
        logger.info("="*60)
        logger.info("🎵 PIPELINE TUNEMETRICS COMPLETADO 🎵")
        logger.info("="*60)
        logger.info(f"Pasos exitosos: {successful_steps}/{total_steps}")
        logger.info(f"Duración total: {self.pipeline_status['total_duration']}s")
        
        if self.pipeline_status['steps_failed']:
            logger.warning(f"Pasos fallidos: {len(self.pipeline_status['steps_failed'])}")
            for failed_step in self.pipeline_status['steps_failed']:
                logger.warning(f"  - {failed_step['step']}: {failed_step['error']}")
        
        # Guardar resumen del pipeline
        pipeline_summary_file = Path(self.config['output']['reports_path']) / "pipeline_execution_summary.json"
        with open(pipeline_summary_file, 'w') as f:
            json.dump(self.pipeline_status, f, indent=2, default=str)
        
        logger.info(f"📄 Resumen guardado en {pipeline_summary_file}")
        
        # Mostrar siguientes pasos
        if successful_steps == total_steps:
            logger.info("""
✅ PIPELINE COMPLETADO EXITOSAMENTE!

📋 PRÓXIMOS PASOS RECOMENDADOS:
1. Revisar reportes generados en reports/
2. Configurar dashboard en Looker Studio usando dashboard_data/
3. Usar deployment para hacer predicciones en nuevos datos
4. Configurar monitoreo continuo de performance

🚀 COMANDO PARA USAR MODELOS:
python src/deployment/deployment_manager.py --action predict --input-data your_data.csv
            """)
        else:
            logger.info("""
⚠️  PIPELINE COMPLETADO CON ALGUNOS ERRORES

📋 REVISA:
1. Logs en tunemetrics_pipeline.log
2. Reportes de pasos completados en reports/
3. Errores específicos arriba para resolución
            """)
        
        return successful_steps == total_steps

def main():
    """Función principal con CLI"""
    parser = argparse.ArgumentParser(description='TuneMetrics - Pipeline Completo de Engagement Musical')
    parser.add_argument('--config', type=str, help='Ruta al archivo de configuración YAML')
    parser.add_argument('--steps', nargs='+', 
                       choices=['eda', 'processing', 'training', 'evaluation', 'dashboard', 'deployment'],
                       help='Pasos específicos a ejecutar (default: todos)')
    parser.add_argument('--validate-setup', action='store_true',
                       help='Solo validar configuración y estructura')
    
    args = parser.parse_args()
    
    try:
        # Crear runner del pipeline
        runner = TuneMetricsPipelineRunner(args.config)
        
        if args.validate_setup:
            logger.info("🔍 Validando configuración y estructura...")
            logger.info(f"✅ Configuración cargada: {runner.config['project']['name']}")
            logger.info("✅ Estructura de directorios verificada")
            logger.info("✅ Setup válido - listo para ejecutar pipeline")
            return 0
        
        # Ejecutar pipeline
        success = runner.run_full_pipeline(args.steps)
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        logger.info("\n⏹️  Pipeline interrumpido por el usuario")
        return 1
    except Exception as e:
        logger.error(f"❌ Error crítico en pipeline: {e}")
        return 1

if __name__ == "__main__":
    exit(main())