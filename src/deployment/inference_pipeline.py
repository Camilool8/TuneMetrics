# Pipeline de inferencia para deployment de TuneMetrics

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from typing import Union, Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class TuneMetricsInferencePipeline:
    """
    Pipeline de inferencia para predicción de engagement musical en producción
    Maneja la carga de modelos, procesamiento de datos y predicciones
    """
    
    def __init__(self, models_dir: str, default_model: str = 'best'):
        """
        Inicializa el pipeline de inferencia
        
        Args:
            models_dir (str): Directorio donde están los modelos entrenados
            default_model (str): Modelo por defecto a usar ('best', 'random_forest', etc.)
        """
        self.models_dir = Path(models_dir)
        self.default_model = default_model
        self.models = {}
        self.scalers = {}
        self.label_encoder = None
        self.feature_names = []
        self.model_metadata = {}
        
        # Configuración por defecto
        self.estimated_duration_ms = 210000  # 3.5 minutos
        self.engagement_thresholds = {'high': 0.75, 'medium': 0.45}
        
        # Cargar componentes automáticamente
        self._load_pipeline_components()
    
    def _load_pipeline_components(self):
        """Carga todos los componentes necesarios para la inferencia"""
        print("Inicializando pipeline de inferencia...")
        
        trained_dir = self.models_dir / "trained"
        if not trained_dir.exists():
            raise FileNotFoundError(f"Directorio de modelos no encontrado: {trained_dir}")
        
        # 1. Cargar modelos
        model_files = {
            'random_forest': 'random_forest_model.pkl',
            'xgboost': 'xgboost_model.pkl',
            'logistic_regression': 'logistic_regression_model.pkl',
            'mlp': 'mlp_model.pkl'
        }
        
        for model_name, filename in model_files.items():
            model_path = trained_dir / filename
            if model_path.exists():
                self.models[model_name] = joblib.load(model_path)
                print(f"  {model_name} cargado")
        
        # 2. Cargar scalers
        scaler_files = {
            'logistic_regression': 'logistic_regression_scaler.pkl',
            'mlp': 'mlp_scaler.pkl'
        }
        
        for scaler_name, filename in scaler_files.items():
            scaler_path = trained_dir / filename
            if scaler_path.exists():
                self.scalers[scaler_name] = joblib.load(scaler_path)
        
        # 3. Cargar label encoder
        encoder_path = trained_dir / "label_encoder.pkl"
        if encoder_path.exists():
            self.label_encoder = joblib.load(encoder_path)
        
        # 4. Cargar metadatos de modelos
        metadata_path = self.models_dir / "metrics" / "model_metrics.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.model_metadata = json.load(f)
        
        # 5. Determinar mejor modelo si se requiere
        if self.default_model == 'best':
            self.default_model = self._select_best_model()
        
        # 6. Definir features esperadas
        self.feature_names = [
            'completion_rate_score', 'skip_resistance_score', 'context_preference_score',
            'consistency_score', 'platform_appeal_score', 'total_plays', 'popularity_score',
            'weekend_listening_rate', 'hour_variability', 'ms_played_mean', 'ms_played_std'
        ]
        
        print(f"Pipeline inicializado con modelo por defecto: {self.default_model}")
    
    def _select_best_model(self) -> str:
        """Selecciona automáticamente el mejor modelo basado en métricas"""
        if not self.model_metadata:
            return 'random_forest'  # Fallback
        
        best_model = None
        best_score = 0
        
        for model_name, metrics in self.model_metadata.items():
            # Score compuesto: 40% accuracy + 40% f1_macro + 20% f1_weighted
            score = (0.4 * metrics.get('accuracy', 0) + 
                    0.4 * metrics.get('f1_macro', 0) + 
                    0.2 * metrics.get('f1_weighted', 0))
            
            if score > best_score:
                best_score = score
                best_model = model_name
        
        return best_model or 'random_forest'
    
    def process_raw_listening_data(self, listening_data: pd.DataFrame) -> pd.DataFrame:
        """
        Procesa datos de escucha raw para crear features de engagement
        
        Args:
            listening_data (pd.DataFrame): Datos raw de reproducciones por canción
            Columnas esperadas: ['spotify_track_uri', 'ms_played', 'skipped', 'shuffle', 
                               'platform', 'reason_start', 'reason_end', 'ts']
        
        Returns:
            pd.DataFrame: Features procesadas listas para predicción
        """
        print("Procesando datos de escucha...")
        
        # Validar columnas mínimas requeridas
        required_cols = ['spotify_track_uri', 'ms_played']
        missing_cols = [col for col in required_cols if col not in listening_data.columns]
        if missing_cols:
            raise ValueError(f"Columnas faltantes: {missing_cols}")
        
        # Agregar por canción
        agg_dict = {
            'spotify_track_uri': 'count',  # Total reproducciones
            'ms_played': ['mean', 'median', 'std', 'min', 'max'],
        }
        
        # Agregar métricas de comportamiento si están disponibles
        if 'skipped' in listening_data.columns:
            listening_data['skipped'] = listening_data['skipped'].fillna(False)
            agg_dict['skipped'] = 'mean'  # Tasa de skip
        
        if 'shuffle' in listening_data.columns:
            listening_data['shuffle'] = listening_data['shuffle'].fillna(True)
            agg_dict['shuffle'] = 'mean'  # Tasa de shuffle
        
        if 'platform' in listening_data.columns:
            agg_dict['platform'] = 'nunique'  # Diversidad de plataformas
        
        if 'ts' in listening_data.columns:
            listening_data['ts'] = pd.to_datetime(listening_data['ts'])
            listening_data['is_weekend'] = listening_data['ts'].dt.dayofweek.isin([5, 6])
            listening_data['hour'] = listening_data['ts'].dt.hour
            agg_dict['is_weekend'] = 'mean'
            agg_dict['hour'] = lambda x: x.std() if len(x) > 1 else 0
        
        # Agregar datos
        aggregated = listening_data.groupby('spotify_track_uri').agg(agg_dict).reset_index()
        
        # Aplanar columnas multinivel
        aggregated.columns = ['_'.join(col).strip() if col[1] else col[0] 
                             for col in aggregated.columns.values]
        aggregated = aggregated.rename(columns={'spotify_track_uri_': 'spotify_track_uri'})
        
        # Crear métricas de engagement
        processed_data = self._create_engagement_features(aggregated)
        
        print(f"{len(processed_data)} canciones procesadas")
        return processed_data
    
    def _create_engagement_features(self, aggregated_data: pd.DataFrame) -> pd.DataFrame:
        """Crea features de engagement a partir de datos agregados"""
        
        # Crear copia para trabajar
        df = aggregated_data.copy()
        
        # 1. Completion Rate Score
        df['completion_rate_score'] = np.clip(
            df.get('ms_played_mean', 0) / self.estimated_duration_ms, 0, 1
        )
        
        # 2. Skip Resistance Score
        if 'skipped_mean' in df.columns:
            df['skip_resistance_score'] = 1 - df['skipped_mean']
        else:
            df['skip_resistance_score'] = 0.8  # Valor por defecto conservador
        
        # 3. Context Preference Score (intencional vs shuffle)
        if 'shuffle_mean' in df.columns:
            df['context_preference_score'] = 1 - df['shuffle_mean']
        else:
            df['context_preference_score'] = 0.3  # Valor por defecto (mayoría shuffle)
        
        # 4. Consistency Score (basado en baja variabilidad)
        if 'ms_played_std' in df.columns:
            df['consistency_score'] = 1 / (1 + df['ms_played_std'].fillna(0) / df.get('ms_played_mean', 1))
        else:
            df['consistency_score'] = 0.5  # Valor neutral
        
        # 5. Platform Appeal Score
        if 'platform_nunique' in df.columns:
            max_platforms = 6  # Máximo observado: web, windows, android, iOS, cast, mac
            df['platform_appeal_score'] = np.clip(df['platform_nunique'] / max_platforms, 0, 1)
        else:
            df['platform_appeal_score'] = 0.3  # Valor por defecto
        
        # 6. Total Plays (renombrar)
        if 'spotify_track_uri_count' in df.columns:
            df['total_plays'] = df['spotify_track_uri_count']
        else:
            df['total_plays'] = 1  # Mínimo una reproducción
        
        # 7. Popularity Score (log-normalizado)
        df['popularity_score'] = np.log1p(df['total_plays']) / np.log1p(df['total_plays'].max())
        
        # 8. Weekend Listening Rate
        if 'is_weekend_mean' in df.columns:
            df['weekend_listening_rate'] = df['is_weekend_mean']
        else:
            df['weekend_listening_rate'] = 0.3  # Valor por defecto (30% weekend)
        
        # 9. Hour Variability
        if 'hour_<lambda>' in df.columns:
            df['hour_variability'] = df['hour_<lambda>']
        elif 'hour_mean' in df.columns:
            df['hour_variability'] = 0.5  # Valor por defecto
        else:
            df['hour_variability'] = 0.5
        
        # 10. Mantener columnas originales necesarias
        feature_columns = self.feature_names + ['spotify_track_uri']
        available_features = [col for col in feature_columns if col in df.columns]
        
        return df[available_features].fillna(0)
    
    def predict_single_song(self, song_data: Dict, model_name: Optional[str] = None) -> Dict:
        """
        Predice engagement para una sola canción
        
        Args:
            song_data (Dict): Datos de la canción con features o datos raw
            model_name (Optional[str]): Modelo específico a usar
        
        Returns:
            Dict: Resultado de predicción con engagement category y confidence
        """
        model_name = model_name or self.default_model
        
        if model_name not in self.models:
            raise ValueError(f"Modelo {model_name} no disponible. Modelos: {list(self.models.keys())}")
        
        # Convertir a DataFrame
        df = pd.DataFrame([song_data])
        
        # Si no tiene features procesadas, crear a partir de datos raw
        if not all(feature in df.columns for feature in self.feature_names):
            # Simular datos agregados para una canción
            df = self._create_engagement_features(df)
        
        # Preparar features
        X = df[self.feature_names].fillna(0)
        
        # Aplicar scaler si es necesario
        if model_name in self.scalers:
            X = self.scalers[model_name].transform(X)
        
        # Hacer predicción
        model = self.models[model_name]
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        
        # Convertir a nombres de clases
        predicted_class = self.label_encoder.inverse_transform([prediction])[0]
        confidence = np.max(probabilities)
        
        # Crear resultado detallado
        result = {
            'predicted_engagement': predicted_class,
            'confidence': float(confidence),
            'probabilities': {
                class_name: float(prob) 
                for class_name, prob in zip(self.label_encoder.classes_, probabilities)
            },
            'model_used': model_name,
            'features_used': dict(zip(self.feature_names, X.flatten() if hasattr(X, 'flatten') else X.iloc[0]))
        }
        
        return result
    
    def predict_batch(self, songs_data: pd.DataFrame, 
                     model_name: Optional[str] = None,
                     include_features: bool = False) -> pd.DataFrame:
        """
        Predice engagement para múltiples canciones
        
        Args:
            songs_data (pd.DataFrame): Datos de múltiples canciones
            model_name (Optional[str]): Modelo específico a usar
            include_features (bool): Si incluir features en el resultado
        
        Returns:
            pd.DataFrame: Resultados de predicción para todas las canciones
        """
        print(f"Prediciendo engagement para {len(songs_data)} canciones...")
        
        model_name = model_name or self.default_model
        
        if model_name not in self.models:
            raise ValueError(f"Modelo {model_name} no disponible")
        
        # Procesar datos si es necesario
        if not all(feature in songs_data.columns for feature in self.feature_names):
            processed_data = self._create_engagement_features(songs_data)
        else:
            processed_data = songs_data.copy()
        
        # Preparar features
        X = processed_data[self.feature_names].fillna(0)
        
        # Aplicar scaler si es necesario
        if model_name in self.scalers:
            X = self.scalers[model_name].transform(X)
        
        # Hacer predicciones
        model = self.models[model_name]
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        # Convertir predicciones
        predicted_classes = self.label_encoder.inverse_transform(predictions)
        confidences = np.max(probabilities, axis=1)
        
        # Crear DataFrame de resultados
        results = pd.DataFrame({
            'spotify_track_uri': processed_data.get('spotify_track_uri', range(len(predictions))),
            'predicted_engagement': predicted_classes,
            'confidence': confidences,
            'model_used': model_name
        })
        
        # Agregar probabilidades por clase
        for i, class_name in enumerate(self.label_encoder.classes_):
            results[f'prob_{class_name.lower()}'] = probabilities[:, i]
        
        # Agregar features si se solicita
        if include_features:
            feature_df = pd.DataFrame(X, columns=self.feature_names)
            results = pd.concat([results, feature_df], axis=1)
        
        print(f"Predicciones completadas para {len(results)} canciones")
        return results
    
    def predict_from_spotify_data(self, raw_listening_data: pd.DataFrame,
                                 model_name: Optional[str] = None) -> pd.DataFrame:
        """
        Pipeline completo: datos raw de Spotify → predicciones de engagement
        
        Args:
            raw_listening_data (pd.DataFrame): Datos raw de reproducciones
            model_name (Optional[str]): Modelo específico a usar
        
        Returns:
            pd.DataFrame: Predicciones de engagement por canción
        """
        # 1. Procesar datos raw
        processed_features = self.process_raw_listening_data(raw_listening_data)
        
        # 2. Hacer predicciones
        predictions = self.predict_batch(processed_features, model_name, include_features=True)
        
        # 3. Agregar información adicional de negocio
        predictions = self._add_business_insights(predictions)
        
        return predictions
    
    def _add_business_insights(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Agrega insights de negocio a las predicciones"""
        
        # Crear recomendaciones de inversión
        def investment_recommendation(row):
            engagement = row['predicted_engagement']
            confidence = row['confidence']
            
            if engagement == 'High' and confidence > 0.8:
                return 'High Investment Recommended'
            elif engagement == 'High' and confidence > 0.6:
                return 'Moderate Investment Recommended'
            elif engagement == 'Medium' and confidence > 0.7:
                return 'Moderate Investment Recommended'
            elif engagement == 'Low' and confidence > 0.8:
                return 'Avoid Investment'
            else:
                return 'Additional Analysis Required'
        
        predictions['investment_recommendation'] = predictions.apply(investment_recommendation, axis=1)
        
        # Calcular score de riesgo
        predictions['risk_score'] = 1 - predictions['confidence']
        
        # Categorizar por confianza
        def confidence_category(confidence):
            if confidence > 0.8:
                return 'High Confidence'
            elif confidence > 0.6:
                return 'Medium Confidence'
            else:
                return 'Low Confidence'
        
        predictions['confidence_category'] = predictions['confidence'].apply(confidence_category)
        
        return predictions
    
    def get_model_info(self) -> Dict:
        """Retorna información sobre los modelos disponibles"""
        info = {
            'available_models': list(self.models.keys()),
            'default_model': self.default_model,
            'feature_names': self.feature_names,
            'engagement_classes': self.label_encoder.classes_.tolist() if self.label_encoder else [],
            'model_metadata': self.model_metadata
        }
        return info
    
    def validate_input_data(self, data: Union[pd.DataFrame, Dict]) -> Tuple[bool, List[str]]:
        """
        Valida que los datos de entrada sean correctos
        
        Args:
            data: Datos a validar
        
        Returns:
            Tuple[bool, List[str]]: (es_válido, lista_de_errores)
        """
        errors = []
        
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        # Validar columnas mínimas
        required_cols = ['spotify_track_uri'] if 'spotify_track_uri' in data.columns else []
        
        # Si tiene features, validar que estén completas
        if any(feature in data.columns for feature in self.feature_names):
            missing_features = [f for f in self.feature_names if f not in data.columns]
            if missing_features:
                errors.append(f"Features faltantes: {missing_features}")
        
        # Validar tipos de datos
        if 'ms_played' in data.columns:
            if not pd.api.types.is_numeric_dtype(data['ms_played']):
                errors.append("ms_played debe ser numérico")
            elif data['ms_played'].min() < 0:
                errors.append("ms_played no puede ser negativo")
        
        # Validar que no esté vacío
        if len(data) == 0:
            errors.append("Dataset está vacío")
        
        return len(errors) == 0, errors

def create_prediction_api():
    """
    Crea una API simple para hacer predicciones
    Útil para deployment en producción
    """
    pipeline = TuneMetricsInferencePipeline("models")
    
    def predict_engagement(song_data: Dict) -> Dict:
        """
        API endpoint para predicción de engagement
        
        Args:
            song_data (Dict): Datos de la canción
            
        Returns:
            Dict: Predicción y metadatos
        """
        try:
            # Validar entrada
            is_valid, errors = pipeline.validate_input_data(song_data)
            if not is_valid:
                return {'error': f"Datos inválidos: {errors}"}
            
            # Hacer predicción
            result = pipeline.predict_single_song(song_data)
            
            # Agregar metadatos
            result['api_version'] = '1.0'
            result['model_info'] = pipeline.get_model_info()
            
            return result
            
        except Exception as e:
            return {'error': str(e)}
    
    return predict_engagement

def main():
    """Función principal para demostrar el pipeline de inferencia"""
    
    try:
        # Inicializar pipeline
        pipeline = TuneMetricsInferencePipeline("models")
        
        # Mostrar información del pipeline
        info = pipeline.get_model_info()
        print("📊 Información del Pipeline:")
        print(f"  Modelos disponibles: {info['available_models']}")
        print(f"  Modelo por defecto: {info['default_model']}")
        print(f"  Clases de engagement: {info['engagement_classes']}")
        
        # Ejemplo de predicción para una canción
        print("\n🎵 Ejemplo de predicción para una canción:")
        
        song_example = {
            'spotify_track_uri': 'example_track_123',
            'ms_played': 180000,  # 3 minutos
            'skipped': False,
            'shuffle': False,
            'platform': 'android',
            'total_plays': 15
        }
        
        result = pipeline.predict_single_song(song_example)
        print(f"  Canción: {song_example['spotify_track_uri']}")
        print(f"  Engagement predicho: {result['predicted_engagement']}")
        print(f"  Confianza: {result['confidence']:.3f}")
        print(f"  Recomendación: {'Alta inversión' if result['predicted_engagement'] == 'High' and result['confidence'] > 0.8 else 'Análisis adicional'}")
        
        # Ejemplo de predicción batch
        print("\n📊 Ejemplo de predicción batch:")
        
        batch_data = pd.DataFrame([
            {'spotify_track_uri': 'track_001', 'ms_played': 200000, 'skipped': False, 'total_plays': 20},
            {'spotify_track_uri': 'track_002', 'ms_played': 50000, 'skipped': True, 'total_plays': 5},
            {'spotify_track_uri': 'track_003', 'ms_played': 190000, 'skipped': False, 'total_plays': 35}
        ])
        
        batch_results = pipeline.predict_batch(batch_data)
        print(f"  Predicciones para {len(batch_results)} canciones:")
        for _, row in batch_results.iterrows():
            print(f"    {row['spotify_track_uri']}: {row['predicted_engagement']} (conf: {row['confidence']:.3f})")
        
        print("\n✅ Pipeline de inferencia funcionando correctamente!")
        
    except Exception as e:
        print(f"❌ Error en pipeline de inferencia: {e}")
        raise

if __name__ == "__main__":
    main()