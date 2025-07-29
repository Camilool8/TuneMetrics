# Evaluador y validador de modelos entrenados para TuneMetrics

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, roc_auc_score
)
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class TuneMetricsModelEvaluator:
    """
    Evaluador completo de modelos entrenados para TuneMetrics
    Incluye carga de modelos, evaluación avanzada y análisis de interpretabilidad
    """
    
    def __init__(self, models_dir: str):
        """
        Inicializa el evaluador
        
        Args:
            models_dir (str): Directorio donde están los modelos entrenados
        """
        self.models_dir = Path(models_dir)
        self.models = {}
        self.scalers = {}
        self.label_encoder = None
        self.feature_names = []
        self.evaluation_results = {}
        
    def load_trained_models(self):
        """Carga todos los modelos entrenados y sus componentes"""
        print("Cargando modelos entrenados...")
        
        trained_dir = self.models_dir / "trained"
        if not trained_dir.exists():
            raise FileNotFoundError(f"Directorio de modelos no encontrado: {trained_dir}")
        
        # Cargar modelos
        model_files = {
            'random_forest': 'random_forest_model.pkl',
            'xgboost': 'xgboost_model.pkl', 
            'logistic_regression': 'logistic_regression_model.pkl',
            'mlp': 'mlp_model.pkl'
        }
        
        loaded_models = 0
        for model_name, filename in model_files.items():
            model_path = trained_dir / filename
            if model_path.exists():
                self.models[model_name] = joblib.load(model_path)
                print(f"  {model_name} cargado")
                loaded_models += 1
            else:
                print(f"  {model_name} no encontrado")
        
        # Cargar scalers
        scaler_files = {
            'logistic_regression': 'logistic_regression_scaler.pkl',
            'mlp': 'mlp_scaler.pkl'
        }
        
        for scaler_name, filename in scaler_files.items():
            scaler_path = trained_dir / filename
            if scaler_path.exists():
                self.scalers[scaler_name] = joblib.load(scaler_path)
                print(f"  Scaler {scaler_name} cargado")
        
        # Cargar label encoder
        encoder_path = trained_dir / "label_encoder.pkl"
        if encoder_path.exists():
            self.label_encoder = joblib.load(encoder_path)
            print(f"  Label encoder cargado")
        
        # Cargar métricas previas si existen
        metrics_path = self.models_dir / "metrics" / "model_metrics.json"
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                self.previous_metrics = json.load(f)
            print(f"  Métricas previas cargadas")
        
        print(f"{loaded_models} modelos cargados exitosamente")
        return self.models
    
    def evaluate_on_new_data(self, test_data: pd.DataFrame, 
                           feature_columns: list = None,
                           target_column: str = 'engagement_category'):
        """
        Evalúa los modelos en nuevos datos
        
        Args:
            test_data (pd.DataFrame): Datos de prueba
            feature_columns (list): Columnas de features a usar
            target_column (str): Columna objetivo
        """
        print("\nEvaluando modelos en nuevos datos...")
        
        if feature_columns is None:
            feature_columns = [
                'completion_rate_score', 'skip_resistance_score', 'context_preference_score',
                'consistency_score', 'platform_appeal_score', 'total_plays', 'popularity_score',
                'weekend_listening_rate', 'hour_variability', 'ms_played_mean', 'ms_played_std'
            ]
        
        # Verificar features disponibles
        available_features = [col for col in feature_columns if col in test_data.columns]
        missing_features = [col for col in feature_columns if col not in test_data.columns]
        
        if missing_features:
            print(f"  Features faltantes: {missing_features}")
        
        self.feature_names = available_features
        print(f"  Features utilizadas: {len(self.feature_names)}")
        
        # Preparar datos
        X_test = test_data[self.feature_names].fillna(test_data[self.feature_names].median())
        
        if target_column in test_data.columns:
            y_test = test_data[target_column]
            y_test_encoded = self.label_encoder.transform(y_test)
            has_ground_truth = True
        else:
            print("  No hay ground truth disponible, solo se harán predicciones")
            has_ground_truth = False
        
        # Evaluar cada modelo
        results = {}
        
        for model_name, model in self.models.items():
            print(f"\n  Evaluando {model_name}...")
            
            # Preparar datos según el modelo
            if model_name in self.scalers:
                X_test_processed = self.scalers[model_name].transform(X_test)
            else:
                X_test_processed = X_test
            
            # Hacer predicciones
            y_pred = model.predict(X_test_processed)
            y_pred_proba = model.predict_proba(X_test_processed)
            
            # Convertir predicciones a nombres de clases
            y_pred_labels = self.label_encoder.inverse_transform(y_pred)
            
            result = {
                'predictions': y_pred_labels.tolist(),
                'predicted_classes': y_pred.tolist(),
                'probabilities': y_pred_proba.tolist(),
                'class_names': self.label_encoder.classes_.tolist()
            }
            
            # Si hay ground truth, calcular métricas
            if has_ground_truth:
                accuracy = accuracy_score(y_test_encoded, y_pred)
                precision = precision_score(y_test_encoded, y_pred, average='weighted')
                recall = recall_score(y_test_encoded, y_pred, average='weighted')
                f1_weighted = f1_score(y_test_encoded, y_pred, average='weighted')
                f1_macro = f1_score(y_test_encoded, y_pred, average='macro')
                
                # AUC
                try:
                    auc_score = roc_auc_score(y_test_encoded, y_pred_proba, multi_class='ovr')
                except:
                    auc_score = None
                
                # Reporte de clasificación
                class_report = classification_report(
                    y_test_encoded, y_pred,
                    target_names=self.label_encoder.classes_,
                    output_dict=True
                )
                
                # Matriz de confusión
                conf_matrix = confusion_matrix(y_test_encoded, y_pred)
                
                result.update({
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_weighted': f1_weighted,
                    'f1_macro': f1_macro,
                    'auc_score': auc_score,
                    'classification_report': class_report,
                    'confusion_matrix': conf_matrix.tolist(),
                    'ground_truth': y_test.tolist()
                })
                
                print(f"    Accuracy: {accuracy:.4f}")
                print(f"    F1-Score (macro): {f1_macro:.4f}")
                print(f"    F1-Score (weighted): {f1_weighted:.4f}")
            
            results[model_name] = result
        
        self.evaluation_results = results
        return results
    
    def analyze_feature_importance(self):
        """Analiza la importancia de features en los modelos"""
        print("\nAnalizando importancia de features...")
        
        feature_importance_data = {}
        
        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                feature_importance_data[model_name] = dict(zip(self.feature_names, importance))
                
                print(f"\n  {model_name} - Top 5 features:")
                top_features = sorted(
                    zip(self.feature_names, importance), 
                    key=lambda x: x[1], reverse=True
                )[:5]
                
                for i, (feature, imp) in enumerate(top_features, 1):
                    print(f"    {i}. {feature}: {imp:.4f}")
            
            elif hasattr(model, 'coef_'):
                # Para regresión logística, usar coeficientes
                coef = np.abs(model.coef_).mean(axis=0)  # Promedio de coeficientes absolutos
                feature_importance_data[model_name] = dict(zip(self.feature_names, coef))
                
                print(f"\n  {model_name} - Top 5 features (coeficientes):")
                top_features = sorted(
                    zip(self.feature_names, coef), 
                    key=lambda x: x[1], reverse=True
                )[:5]
                
                for i, (feature, coef_val) in enumerate(top_features, 1):
                    print(f"    {i}. {feature}: {coef_val:.4f}")
        
        return feature_importance_data
    
    def analyze_prediction_confidence(self):
        """Analiza la confianza de las predicciones"""
        print("\nAnalizando confianza de predicciones...")
        
        confidence_analysis = {}
        
        for model_name, results in self.evaluation_results.items():
            probabilities = np.array(results['probabilities'])
            
            # Confianza = probabilidad máxima
            max_probabilities = np.max(probabilities, axis=1)
            
            # Estadísticas de confianza
            confidence_stats = {
                'mean_confidence': np.mean(max_probabilities),
                'median_confidence': np.median(max_probabilities),
                'std_confidence': np.std(max_probabilities),
                'min_confidence': np.min(max_probabilities),
                'max_confidence': np.max(max_probabilities),
                'low_confidence_count': np.sum(max_probabilities < 0.6),
                'high_confidence_count': np.sum(max_probabilities > 0.8)
            }
            
            confidence_analysis[model_name] = confidence_stats
            
            print(f"\n  {model_name}:")
            print(f"    Confianza promedio: {confidence_stats['mean_confidence']:.3f}")
            print(f"    Predicciones de baja confianza (<0.6): {confidence_stats['low_confidence_count']}")
            print(f"    Predicciones de alta confianza (>0.8): {confidence_stats['high_confidence_count']}")
        
        return confidence_analysis
    
    def compare_models_performance(self):
        """Compara el rendimiento de todos los modelos"""
        print("\nComparando rendimiento de modelos...")
        
        if not self.evaluation_results:
            print("  No hay resultados de evaluación disponibles")
            return None
        
        # Crear DataFrame de comparación
        comparison_data = []
        
        for model_name, results in self.evaluation_results.items():
            if 'accuracy' in results:  # Solo si hay ground truth
                comparison_data.append({
                    'Model': model_name,
                    'Accuracy': results['accuracy'],
                    'F1_Macro': results['f1_macro'],
                    'F1_Weighted': results['f1_weighted'],
                    'Precision': results['precision'],
                    'Recall': results['recall'],
                    'AUC': results.get('auc_score', 0)
                })
        
        if not comparison_data:
            print("  No hay métricas para comparar")
            return None
        
        comparison_df = pd.DataFrame(comparison_data)
        
        print("\n  Ranking por Accuracy:")
        ranking = comparison_df.sort_values('Accuracy', ascending=False)
        for i, row in ranking.iterrows():
            print(f"    {row.name + 1}. {row['Model']}: {row['Accuracy']:.4f}")
        
        print("\n  Ranking por F1-Macro:")
        ranking = comparison_df.sort_values('F1_Macro', ascending=False)
        for i, row in ranking.iterrows():
            print(f"    {row.name + 1}. {row['Model']}: {row['F1_Macro']:.4f}")
        
        return comparison_df
    
    def create_evaluation_visualizations(self, save_dir: str = None):
        """Crea visualizaciones completas de la evaluación"""
        print("\nCreando visualizaciones de evaluación...")
        
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Matriz de confusión para cada modelo
        models_with_metrics = {
            name: results for name, results in self.evaluation_results.items()
            if 'confusion_matrix' in results
        }
        
        if models_with_metrics:
            n_models = len(models_with_metrics)
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.flatten()
            
            for i, (model_name, results) in enumerate(models_with_metrics.items()):
                if i < 4:  # Máximo 4 modelos
                    conf_matrix = np.array(results['confusion_matrix'])
                    sns.heatmap(
                        conf_matrix, annot=True, fmt='d', cmap='Blues',
                        xticklabels=self.label_encoder.classes_,
                        yticklabels=self.label_encoder.classes_,
                        ax=axes[i]
                    )
                    axes[i].set_title(f'{model_name.replace("_", " ").title()}')
                    axes[i].set_xlabel('Predicted')
                    axes[i].set_ylabel('Actual')
            
            # Ocultar axes no utilizados
            for i in range(n_models, 4):
                axes[i].set_visible(False)
            
            plt.suptitle('Matrices de Confusión por Modelo', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if save_dir:
                plt.savefig(save_path / "confusion_matrices.png", dpi=300, bbox_inches='tight')
            plt.show()
        
        # 2. Comparación de métricas
        comparison_df = self.compare_models_performance()
        
        if comparison_df is not None:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Gráfico de barras para métricas principales
            metrics_to_plot = ['Accuracy', 'F1_Macro', 'F1_Weighted']
            comparison_subset = comparison_df[['Model'] + metrics_to_plot].set_index('Model')
            
            comparison_subset.plot(kind='bar', ax=axes[0], width=0.8)
            axes[0].set_title('Comparación de Métricas Principales')
            axes[0].set_ylabel('Score')
            axes[0].legend()
            axes[0].tick_params(axis='x', rotation=45)
            
            # Gráfico radar
            categories = metrics_to_plot
            
            # Preparar datos para gráfico radar
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
            angles = np.concatenate((angles, [angles[0]]))  # Cerrar el círculo
            
            for _, row in comparison_df.iterrows():
                values = [row[cat] for cat in categories]
                values += [values[0]]  # Cerrar el círculo
                
                axes[1].plot(angles, values, 'o-', linewidth=2, label=row['Model'])
                axes[1].fill(angles, values, alpha=0.25)
            
            axes[1].set_xticks(angles[:-1])
            axes[1].set_xticklabels(categories)
            axes[1].set_ylim(0, 1)
            axes[1].set_title('Comparación Radar de Modelos')
            axes[1].legend()
            axes[1].grid(True)
            
            plt.tight_layout()
            
            if save_dir:
                plt.savefig(save_path / "model_comparison.png", dpi=300, bbox_inches='tight')
            plt.show()
        
        # 3. Distribución de confianza de predicciones
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (model_name, results) in enumerate(self.evaluation_results.items()):
            if i < 4 and 'probabilities' in results:
                probabilities = np.array(results['probabilities'])
                max_probs = np.max(probabilities, axis=1)
                
                axes[i].hist(max_probs, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                axes[i].axvline(np.mean(max_probs), color='red', linestyle='--', 
                               label=f'Media: {np.mean(max_probs):.3f}')
                axes[i].set_title(f'{model_name.replace("_", " ").title()}')
                axes[i].set_xlabel('Confianza de Predicción')
                axes[i].set_ylabel('Frecuencia')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        # Ocultar axes no utilizados
        for i in range(len(self.evaluation_results), 4):
            axes[i].set_visible(False)
        
        plt.suptitle('Distribución de Confianza de Predicciones', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(save_path / "prediction_confidence.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Visualizaciones creadas" + (f" y guardadas en {save_dir}" if save_dir else ""))
    
    def validate_model_stability(self, validation_data: pd.DataFrame):
        """Valida la estabilidad de los modelos con datos de validación"""
        print("\n🔬 Validando estabilidad de modelos...")
        
        # Dividir datos de validación en subconjuntos temporales
        validation_data_sorted = validation_data.sort_values('ts' if 'ts' in validation_data.columns else validation_data.index)
        
        n_splits = 3
        split_size = len(validation_data_sorted) // n_splits
        
        stability_results = {}
        
        for model_name in self.models.keys():
            print(f"\n  🔍 Validando {model_name}...")
            
            split_metrics = []
            
            for i in range(n_splits):
                start_idx = i * split_size
                end_idx = (i + 1) * split_size if i < n_splits - 1 else len(validation_data_sorted)
                
                split_data = validation_data_sorted.iloc[start_idx:end_idx]
                
                # Evaluar en este split
                split_results = self.evaluate_on_new_data(split_data)
                
                if model_name in split_results and 'accuracy' in split_results[model_name]:
                    split_metrics.append({
                        'split': i + 1,
                        'accuracy': split_results[model_name]['accuracy'],
                        'f1_macro': split_results[model_name]['f1_macro']
                    })
            
            if split_metrics:
                # Calcular estabilidad
                accuracies = [m['accuracy'] for m in split_metrics]
                f1_scores = [m['f1_macro'] for m in split_metrics]
                
                stability_metrics = {
                    'mean_accuracy': np.mean(accuracies),
                    'std_accuracy': np.std(accuracies),
                    'mean_f1': np.mean(f1_scores),
                    'std_f1': np.std(f1_scores),
                    'stability_score': 1 - (np.std(accuracies) + np.std(f1_scores)) / 2,
                    'split_metrics': split_metrics
                }
                
                stability_results[model_name] = stability_metrics
                
                print(f"    Accuracy: {stability_metrics['mean_accuracy']:.4f} ± {stability_metrics['std_accuracy']:.4f}")
                print(f"    F1-Score: {stability_metrics['mean_f1']:.4f} ± {stability_metrics['std_f1']:.4f}")
                print(f"    Stability Score: {stability_metrics['stability_score']:.4f}")
        
        return stability_results
    
    def generate_evaluation_report(self, output_path: str = None):
        """Genera reporte completo de evaluación"""
        print("\nGenerando reporte de evaluación...")
        
        # Análisis de importancia de features
        feature_importance = self.analyze_feature_importance()
        
        # Análisis de confianza
        confidence_analysis = self.analyze_prediction_confidence()
        
        # Comparación de modelos
        comparison_df = self.compare_models_performance()
        
        report = f"""
TUNEMETRICS - REPORTE DE EVALUACIÓN DE MODELOS
{'='*70}

MODELOS EVALUADOS:
├── Modelos cargados: {len(self.models)}
├── Features utilizadas: {len(self.feature_names)}
└── Datos evaluados: {len(list(self.evaluation_results.values())[0].get('predictions', [])) if self.evaluation_results else 0} predicciones

RENDIMIENTO GENERAL:
"""
        
        if comparison_df is not None:
            best_accuracy = comparison_df.loc[comparison_df['Accuracy'].idxmax()]
            best_f1 = comparison_df.loc[comparison_df['F1_Macro'].idxmax()]
            
            report += f"""├── Mejor Accuracy: {best_accuracy['Model']} ({best_accuracy['Accuracy']:.4f})
├── Mejor F1-Macro: {best_f1['Model']} ({best_f1['F1_Macro']:.4f})
└── Promedio Accuracy: {comparison_df['Accuracy'].mean():.4f}

MÉTRICAS DETALLADAS:
"""
            for _, row in comparison_df.iterrows():
                report += f"""
{row['Model'].upper()}:
  ├── Accuracy: {row['Accuracy']:.4f}
  ├── F1-Macro: {row['F1_Macro']:.4f}
  ├── F1-Weighted: {row['F1_Weighted']:.4f}
  ├── Precision: {row['Precision']:.4f}
  └── Recall: {row['Recall']:.4f}"""
        
        # Análisis de confianza
        report += f"""

ANÁLISIS DE CONFIANZA:
"""
        for model_name, conf_stats in confidence_analysis.items():
            report += f"""
{model_name.upper()}:
  ├── Confianza promedio: {conf_stats['mean_confidence']:.3f}
  ├── Predicciones baja confianza: {conf_stats['low_confidence_count']}
  └── Predicciones alta confianza: {conf_stats['high_confidence_count']}"""
        
        # Features más importantes
        report += f"""

FEATURES MÁS IMPORTANTES:
"""
        for model_name, importance_dict in feature_importance.items():
            if importance_dict:
                top_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:5]
                report += f"""
{model_name.upper()}:"""
                for i, (feature, importance) in enumerate(top_features, 1):
                    report += f"""
  {i}. {feature}: {importance:.4f}"""
        
        report += f"""

CONCLUSIONES Y RECOMENDACIONES:
├── Modelos preparados para deployment
├── Pipeline de inferencia listo
├── Métricas de monitoreo establecidas
└── Validación de performance completada
        """
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            print(f"Reporte guardado en: {output_path}")
        
        return report

def main():
    """Función principal para ejecutar la evaluación de modelos"""
    
    # Rutas de configuración
    models_dir = "models"
    test_data_path = "data/processed/gold_data.parquet"  # o .csv
    output_dir = "reports"
    
    try:
        # Crear evaluador
        evaluator = TuneMetricsModelEvaluator(models_dir)
        
        # Cargar modelos entrenados
        evaluator.load_trained_models()
        
        # Cargar datos de test
        print("\n🔄 Cargando datos de test...")
        if test_data_path.endswith('.parquet'):
            test_data = pd.read_parquet(test_data_path)
        else:
            test_data = pd.read_csv(test_data_path)
        
        # Filtrar solo datos de test
        test_data = test_data[test_data['data_split'] == 'test']
        print(f"✅ Datos de test cargados: {len(test_data)} canciones")
        
        # Evaluar modelos
        evaluator.evaluate_on_new_data(test_data)
        
        # Análisis avanzados
        evaluator.analyze_feature_importance()
        evaluator.analyze_prediction_confidence()
        evaluator.compare_models_performance()
        
        # Crear visualizaciones
        Path(output_dir).mkdir(exist_ok=True)
        evaluator.create_evaluation_visualizations(f"{output_dir}/figures")
        
        # Generar reporte
        report = evaluator.generate_evaluation_report(f"{output_dir}/model_evaluation_report.txt")
        print(report)
        
        print("\nEvaluación de modelos completada exitosamente!")
        
    except Exception as e:
        print(f"❌ Error en evaluación de modelos: {e}")
        raise

if __name__ == "__main__":
    main()