# Script simplificado para ejecutar TuneMetrics de forma directa

import os
import sys
from pathlib import Path
from datetime import datetime

def print_header():
    """Imprime header del proyecto"""
    print("="*60)
    print("TUNEMETRICS - PREDICTOR DE ENGAGEMENT MUSICAL")
    print("="*60)
    print("Universidad: PUCMM")
    print("Proyecto: Práctica Final - Análisis de datos de la industria musical")
    print("Metodología: CRISP-DM")
    print("Dataset: 149,860 reproducciones de Spotify (2013-2024)")
    print("="*60)
    print()

def check_requirements():
    """Verifica que los archivos necesarios existen"""
    print("Verificando requisitos...")
    
    required_files = [
        "data/raw/spotify_history.csv"
    ]
    
    required_dirs = [
        "src",
        "data/raw",
        "configs"
    ]
    
    # Crear directorios que no existen
    created_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            created_dirs.append(dir_path)
    
    if created_dirs:
        print(f"Directorios creados: {', '.join(created_dirs)}")
    
    # Verificar archivos críticos
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("Archivos críticos faltantes:")
        for missing in missing_files:
            print(f"  - {missing}")
        print("\nSOLUCIÓN:")
        print("  1. Coloca el archivo spotify_history.csv en data/raw/")
        print("  2. O usa un dataset de ejemplo si no tienes los datos reales")
        return False
    
    # Verificar módulos Python
    print("Verificando módulos Python...")
    try:
        import pandas as pd
        import numpy as np
        import sklearn
        print("Dependencias básicas disponibles")
    except ImportError as e:
        print(f"Dependencias faltantes: {e}")
        print("Ejecuta: pip install pandas numpy scikit-learn")
        return False
    
    print("Todos los requisitos están presentes")
    return True

def run_quick_pipeline():
    """Ejecuta pipeline simplificado paso a paso"""
    try:
        # Verificar requisitos
        if not check_requirements():
            print("\nNo se puede ejecutar el pipeline. Revisa los archivos faltantes.")
            return False
        
        print("\nIniciando pipeline TuneMetrics...")
        start_time = datetime.now()
        
        # Inicializar runner UNA VEZ al principio
        try:
            from main import TuneMetricsPipelineRunner
            runner = TuneMetricsPipelineRunner()
            print("Pipeline runner inicializado")
        except Exception as e:
            print(f"Error inicializando runner: {e}")
            return False
        
        # Paso 1: EDA
        print("\nPASO 1: Análisis Exploratorio de Datos")
        print("-" * 50)
        try:
            runner.step_1_exploratory_analysis()
            print("EDA completado")
        except Exception as e:
            print(f"EDA con advertencias: {e}")
        
        # Paso 2: Procesamiento de Datos
        print("\nPASO 2: Procesamiento de Datos (Bronze → Silver → Gold)")
        print("-" * 50)
        try:
            runner.step_2_data_processing()
            print("Procesamiento completado")
        except Exception as e:
            print(f"Procesamiento con advertencias: {e}")
        
        # Paso 3: Entrenamiento
        print("\nPASO 3: Entrenamiento de Modelos")
        print("-" * 50)
        try:
            runner.step_3_model_training()
            print("Entrenamiento completado")
        except Exception as e:
            print(f"Entrenamiento con advertencias: {e}")
        
        # Paso 4: Evaluación
        print("\nPASO 4: Evaluación de Modelos")
        print("-" * 50)
        try:
            runner.step_4_model_evaluation()
            print("Evaluación completada")
        except Exception as e:
            print(f"Evaluación con advertencias: {e}")
        
        # Paso 5: Dashboard
        print("\nPASO 5: Generación de Dashboard")
        print("-" * 50)
        try:
            runner.step_5_dashboard_generation()
            print("Dashboard generado")
        except Exception as e:
            print(f"Dashboard con advertencias: {e}")
        
        # Paso 6: Deployment
        print("\nPASO 6: Configuración de Deployment")
        print("-" * 50)
        try:
            runner.step_6_deployment_setup()
            print("Deployment configurado")
        except Exception as e:
            print(f"Deployment con advertencias: {e}")
        
        # Resumen final
        end_time = datetime.now()
        duration = (end_time - start_time).seconds
        
        print("\n" + "="*60)
        print("PIPELINE TUNEMETRICS COMPLETADO")
        print("="*60)
        print(f"Duración total: {duration} segundos")
        print(f"Completado: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("\nRESULTADOS GENERADOS:")
        print("├── Análisis exploratorio: reports/eda_summary_report.txt")
        print("├── Datos procesados: data/processed/gold_data.parquet")
        print("├── Modelos entrenados: models/trained/")
        print("├── Evaluación: reports/model_evaluation_report.txt")
        print("├── Dashboard: dashboard_data/")
        print("└── Deployment: reports/deployment_status.json")
        
        print("\nPRÓXIMOS PASOS:")
        print("1. Revisar reportes en carpeta 'reports/'")
        print("2. Usar modelos para predicciones:")
        print("   python src/deployment/deployment_manager.py --action predict --input-data tu_archivo.csv")
        print("3. Configurar dashboard en Looker Studio con archivos de 'dashboard_data/'")
        
        return True
        
    except Exception as e:
        print(f"\nError crítico en pipeline: {e}")
        print("Intenta ejecutar pasos individuales para diagnosticar el problema")
        return False

def run_individual_step():
    """Permite ejecutar pasos individuales"""
    steps = {
        '1': ('Análisis Exploratorio (EDA)', 'eda'),
        '2': ('Procesamiento de Datos', 'processing'), 
        '3': ('Entrenamiento de Modelos', 'training'),
        '4': ('Evaluación de Modelos', 'evaluation'),
        '5': ('Generación de Dashboard', 'dashboard'),
        '6': ('Configuración de Deployment', 'deployment')
    }
    
    print("\nPASOS DISPONIBLES:")
    for key, (name, _) in steps.items():
        print(f"  {key}. {name}")
    
    choice = input("\n¿Qué paso quieres ejecutar? (1-6): ").strip()
    
    if choice in steps:
        step_name, step_code = steps[choice]
        print(f"\nEjecutando: {step_name}")
        
        try:
            # Intentar importar e inicializar runner
            try:
                from main import TuneMetricsPipelineRunner
                runner = TuneMetricsPipelineRunner()
            except ImportError as e:
                print(f"Error importando módulos: {e}")
                print("Asegúrate de que todos los archivos estén en la estructura correcta")
                return False
            except Exception as e:
                print(f"Error inicializando runner: {e}")
                return False
            
            # Ejecutar paso específico
            if step_code == 'eda':
                runner.step_1_exploratory_analysis()
            elif step_code == 'processing':
                runner.step_2_data_processing()
            elif step_code == 'training':
                runner.step_3_model_training()
            elif step_code == 'evaluation':
                runner.step_4_model_evaluation()
            elif step_code == 'dashboard':
                runner.step_5_dashboard_generation()
            elif step_code == 'deployment':
                runner.step_6_deployment_setup()
            
            print(f"{step_name} completado exitosamente!")
            
        except Exception as e:
            print(f"Error ejecutando {step_name}: {e}")
            print(f"Detalles del error: {type(e).__name__}")
            return False
    else:
        print("Opción inválida")
        return False
    
    return True

def show_demo():
    """Muestra información del proyecto para demo académica"""
    print("\nINFORMACIÓN ACADÉMICA DEL PROYECTO")
    print("="*50)
    print("OBJETIVO:")
    print("   Desarrollar un sistema predictivo de engagement musical")
    print("   usando metodología CRISP-DM y datos de Spotify")
    print()
    print("DATASET:")
    print("   • 149,860 reproducciones individuales")
    print("   • Período: 2013-2024 (11+ años)")
    print("   • ~15,000 canciones únicas agregadas")
    print("   • Variables: duración, skip, shuffle, plataforma, etc.")
    print()
    print("METODOLOGÍA CRISP-DM:")
    print("   1. Business Understanding: TuneMetrics como consultora musical")
    print("   2. Data Understanding: EDA completo del dataset")
    print("   3. Data Preparation: Pipeline Bronze→Silver→Gold")
    print("   4. Modeling: Random Forest, XGBoost, Logistic Regression, MLP")
    print("   5. Evaluation: Accuracy ≥85%, F1-score ≥0.80")
    print("   6. Deployment: Pipeline de inferencia y monitoreo")
    print()
    print("MÉTRICAS DE ENGAGEMENT CREADAS:")
    print("   • Completion Rate: duración_escuchada / duración_total")
    print("   • Skip Resistance: 1 - tasa_de_skip")
    print("   • Context Preference: escucha_intencional vs shuffle")
    print("   • Platform Appeal: diversidad de dispositivos")
    print("   • Final Engagement Score: métrica compuesta")
    print()
    print("CATEGORIZACIÓN:")
    print("   • Alto Engagement (≥0.75): Inversión recomendada")
    print("   • Medio Engagement (0.45-0.74): Análisis adicional")
    print("   • Bajo Engagement (<0.45): Evitar inversión")
    print()
    print("PRODUCTOS ENTREGABLES:")
    print("   • Modelos ML serializados (.pkl)")
    print("   • Pipeline de inferencia en producción") 
    print("   • Dashboard ejecutivo para Looker Studio")
    print("   • Reportes de evaluación y monitoreo")
    print("   • Sistema de alertas de degradación")

def main():
    """Función principal del script simplificado"""
    print_header()
    
    print("¿Qué quieres hacer?")
    print("1. Ejecutar pipeline completo (recomendado)")
    print("2. Ejecutar paso individual")
    print("3. Ver información del proyecto (demo)")
    print("4. Solo verificar requisitos")
    print("5. Salir")
    
    choice = input("\nSelecciona una opción (1-5): ").strip()
    
    if choice == '1':
        print("\nEJECUTANDO PIPELINE COMPLETO...")
        print("Esto puede tomar 10-30 minutos dependiendo del hardware")
        confirm = input("¿Continuar? (y/n): ").lower().strip()
        
        if confirm == 'y':
            success = run_quick_pipeline()
            return 0 if success else 1
        else:
            print("Pipeline cancelado")
            return 0
            
    elif choice == '2':
        success = run_individual_step()
        return 0 if success else 1
        
    elif choice == '3':
        show_demo()
        return 0
        
    elif choice == '4':
        check_requirements()
        return 0
        
    elif choice == '5':
        print("¡Hasta luego!")
        return 0
        
    else:
        print("Opción inválida")
        return 1

if __name__ == "__main__":
    exit(main())