# Generador de dashboards y datos para Looker Studio - TuneMetrics

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class TuneMetricsDashboardGenerator:
    """
    Generador de dashboards y datos para visualización en Looker Studio
    Crea datasets optimizados y visualizaciones para TuneMetrics
    """
    
    def __init__(self, data_sources: Dict):
        """
        Inicializa el generador de dashboards
        
        Args:
            data_sources (Dict): Rutas a fuentes de datos
                - gold_data: Datos procesados Gold
                - predictions: Predicciones de modelos
                - monitoring: Datos de monitoreo
                - models_metrics: Métricas de modelos
        """
        self.data_sources = data_sources
        self.dashboard_data = {}
        self.business_kpis = {}
        
        # Configuración de visualizaciones
        self.color_palette = {
            'high_engagement': '#2E8B57',      # Verde oscuro
            'medium_engagement': '#FFD700',    # Oro
            'low_engagement': '#DC143C',       # Rojo
            'primary': '#1f77b4',              # Azul
            'secondary': '#ff7f0e',            # Naranja
            'accent': '#2ca02c'                # Verde
        }
        
        # Cargar datos
        self._load_data_sources()
    
    def _load_data_sources(self):
        """Carga todas las fuentes de datos necesarias"""
        print("📊 Cargando fuentes de datos para dashboard...")
        
        # Cargar datos Gold (principales)
        if 'gold_data' in self.data_sources:
            gold_path = self.data_sources['gold_data']
            if Path(gold_path).exists():
                if str(gold_path).endswith('.parquet'):
                    self.gold_data = pd.read_parquet(gold_path)
                else:
                    self.gold_data = pd.read_csv(gold_path)
                print(f"  ✅ Gold data: {len(self.gold_data)} canciones")
            else:
                print(f"  ⚠️  Gold data no encontrado: {gold_path}")
                self.gold_data = pd.DataFrame()
        
        # Cargar predicciones recientes
        if 'predictions' in self.data_sources:
            pred_path = self.data_sources['predictions']
            if Path(pred_path).exists():
                self.predictions_data = pd.read_csv(pred_path)
                print(f"  ✅ Predicciones: {len(self.predictions_data)} registros")
            else:
                print(f"  ⚠️  Predicciones no encontradas: {pred_path}")
                self.predictions_data = pd.DataFrame()
        
        # Cargar métricas de modelos
        if 'models_metrics' in self.data_sources:
            metrics_path = self.data_sources['models_metrics']
            if Path(metrics_path).exists():
                with open(metrics_path, 'r') as f:
                    self.models_metrics = json.load(f)
                print(f"  ✅ Métricas de modelos: {len(self.models_metrics)} modelos")
            else:
                print(f"  ⚠️  Métricas no encontradas: {metrics_path}")
                self.models_metrics = {}
    
    def create_executive_summary_data(self) -> Dict:
        """Crea datos para resumen ejecutivo de TuneMetrics"""
        print("📈 Generando datos de resumen ejecutivo...")
        
        executive_data = {
            'timestamp': datetime.now().isoformat(),
            'summary_metrics': {},
            'engagement_distribution': {},
            'investment_recommendations': {},
            'model_performance': {},
            'business_insights': {}
        }
        
        if not self.gold_data.empty:
            total_songs = len(self.gold_data)
            
            # Distribución de engagement
            engagement_dist = self.gold_data['engagement_category'].value_counts()
            executive_data['engagement_distribution'] = {
                'high_engagement_count': int(engagement_dist.get('High', 0)),
                'medium_engagement_count': int(engagement_dist.get('Medium', 0)),
                'low_engagement_count': int(engagement_dist.get('Low', 0)),
                'high_engagement_percentage': float(engagement_dist.get('High', 0) / total_songs * 100),
                'medium_engagement_percentage': float(engagement_dist.get('Medium', 0) / total_songs * 100),
                'low_engagement_percentage': float(engagement_dist.get('Low', 0) / total_songs * 100)
            }
            
            # Métricas de summary
            executive_data['summary_metrics'] = {
                'total_songs_analyzed': int(total_songs),
                'avg_engagement_score': float(self.gold_data['final_engagement_score'].mean()),
                'high_potential_songs': int(engagement_dist.get('High', 0)),
                'total_plays_analyzed': int(self.gold_data['total_plays'].sum()),
                'avg_completion_rate': float(self.gold_data['completion_rate_score'].mean()),
                'avg_skip_resistance': float(self.gold_data['skip_resistance_score'].mean())
            }
            
            # Recomendaciones de inversión
            high_confidence_high_eng = len(self.gold_data[
                (self.gold_data['engagement_category'] == 'High') & 
                (self.gold_data['final_engagement_score'] > 0.8)
            ])
            
            executive_data['investment_recommendations'] = {
                'high_investment_recommended': int(high_confidence_high_eng),
                'moderate_investment_recommended': int(engagement_dist.get('Medium', 0)),
                'avoid_investment': int(engagement_dist.get('Low', 0)),
                'additional_analysis_required': int(total_songs - high_confidence_high_eng - 
                                                  engagement_dist.get('Medium', 0) - 
                                                  engagement_dist.get('Low', 0))
            }
        
        # Performance de modelos
        if self.models_metrics:
            best_model = max(self.models_metrics.items(), 
                           key=lambda x: x[1].get('accuracy', 0))
            executive_data['model_performance'] = {
                'best_model_name': best_model[0],
                'best_model_accuracy': float(best_model[1].get('accuracy', 0)),
                'best_model_f1_score': float(best_model[1].get('f1_macro', 0)),
                'models_available': len(self.models_metrics)
            }
        
        return executive_data
    
    def create_engagement_analysis_data(self) -> Dict:
        """Crea datos para análisis detallado de engagement"""
        print("🎯 Generando datos de análisis de engagement...")
        
        if self.gold_data.empty:
            return {'error': 'No hay datos Gold disponibles'}
        
        engagement_analysis = {
            'timestamp': datetime.now().isoformat(),
            'engagement_by_features': {},
            'correlation_analysis': {},
            'engagement_trends': {},
            'top_performers': {},
            'bottom_performers': {}
        }
        
        # Análisis por features
        feature_columns = [
            'completion_rate_score', 'skip_resistance_score', 'context_preference_score',
            'consistency_score', 'platform_appeal_score', 'popularity_score'
        ]
        
        available_features = [col for col in feature_columns if col in self.gold_data.columns]
        
        for feature in available_features:
            by_engagement = self.gold_data.groupby('engagement_category')[feature].agg([
                'mean', 'median', 'std', 'count'
            ]).round(4)
            
            engagement_analysis['engagement_by_features'][feature] = {
                'high_avg': float(by_engagement.loc['High', 'mean']) if 'High' in by_engagement.index else 0,
                'medium_avg': float(by_engagement.loc['Medium', 'mean']) if 'Medium' in by_engagement.index else 0,
                'low_avg': float(by_engagement.loc['Low', 'mean']) if 'Low' in by_engagement.index else 0
            }
        
        # Correlaciones con engagement score
        numeric_columns = self.gold_data.select_dtypes(include=[np.number]).columns
        correlations = self.gold_data[numeric_columns].corrwith(
            self.gold_data['final_engagement_score']
        ).sort_values(ascending=False)
        
        engagement_analysis['correlation_analysis'] = {
            'top_positive_correlations': correlations.head(10).to_dict(),
            'top_negative_correlations': correlations.tail(5).to_dict()
        }
        
        # Top y Bottom performers
        top_performers = self.gold_data.nlargest(20, 'final_engagement_score')[
            ['track_name_first', 'artist_name_first', 'final_engagement_score', 
             'total_plays', 'engagement_category']
        ].to_dict('records')
        
        bottom_performers = self.gold_data.nsmallest(10, 'final_engagement_score')[
            ['track_name_first', 'artist_name_first', 'final_engagement_score', 
             'total_plays', 'engagement_category']
        ].to_dict('records')
        
        engagement_analysis['top_performers'] = top_performers
        engagement_analysis['bottom_performers'] = bottom_performers
        
        return engagement_analysis
    
    def create_business_kpis_data(self) -> Dict:
        """Crea KPIs de negocio para TuneMetrics"""
        print("💼 Generando KPIs de negocio...")
        
        if self.gold_data.empty:
            return {'error': 'No hay datos disponibles para KPIs'}
        
        business_kpis = {
            'timestamp': datetime.now().isoformat(),
            'roi_simulation': {},
            'market_insights': {},
            'portfolio_analysis': {},
            'risk_assessment': {}
        }
        
        # Simulación de ROI
        high_eng_count = len(self.gold_data[self.gold_data['engagement_category'] == 'High'])
        medium_eng_count = len(self.gold_data[self.gold_data['engagement_category'] == 'Medium'])
        low_eng_count = len(self.gold_data[self.gold_data['engagement_category'] == 'Low'])
        
        # Asumiendo costos y retornos típicos de la industria
        investment_per_song = 50000  # $50K promedio por promoción
        
        business_kpis['roi_simulation'] = {
            'high_engagement_expected_roi': 3.5,  # 3.5x retorno
            'medium_engagement_expected_roi': 1.2,  # 1.2x retorno
            'low_engagement_expected_roi': 0.3,   # 0.3x retorno (pérdida)
            'optimal_portfolio_high_pct': 40,
            'optimal_portfolio_medium_pct': 45,
            'optimal_portfolio_low_pct': 15,
            'current_portfolio_high_pct': float(high_eng_count / len(self.gold_data) * 100),
            'current_portfolio_medium_pct': float(medium_eng_count / len(self.gold_data) * 100),
            'current_portfolio_low_pct': float(low_eng_count / len(self.gold_data) * 100),
            'estimated_savings_per_campaign': float((low_eng_count * investment_per_song * 0.7))
        }
        
        # Análisis de mercado
        if 'total_plays' in self.gold_data.columns:
            play_stats = self.gold_data['total_plays'].describe()
            business_kpis['market_insights'] = {
                'avg_plays_per_song': float(play_stats['mean']),
                'median_plays_per_song': float(play_stats['50%']),
                'top_10_pct_plays_threshold': float(self.gold_data['total_plays'].quantile(0.9)),
                'viral_potential_songs': int(len(self.gold_data[self.gold_data['total_plays'] > play_stats['75%']])),
                'underperformed_songs': int(len(self.gold_data[self.gold_data['total_plays'] < play_stats['25%']]))
            }
        
        # Análisis de portfolio
        engagement_scores = self.gold_data['final_engagement_score']
        business_kpis['portfolio_analysis'] = {
            'portfolio_score': float(engagement_scores.mean()),
            'portfolio_consistency': float(1 - engagement_scores.std()),
            'high_potential_percentage': float(len(self.gold_data[engagement_scores > 0.75]) / len(self.gold_data) * 100),
            'diversification_score': float(engagement_scores.std() * 100),  # Mayor diversidad = mayor std
            'concentration_risk': float(len(self.gold_data[engagement_scores < 0.3]) / len(self.gold_data) * 100)
        }
        
        # Evaluación de riesgo
        business_kpis['risk_assessment'] = {
            'high_risk_songs': int(len(self.gold_data[self.gold_data['final_engagement_score'] < 0.4])),
            'medium_risk_songs': int(len(self.gold_data[
                (self.gold_data['final_engagement_score'] >= 0.4) & 
                (self.gold_data['final_engagement_score'] < 0.6)
            ])),
            'low_risk_songs': int(len(self.gold_data[self.gold_data['final_engagement_score'] >= 0.6])),
            'portfolio_risk_score': float(1 - engagement_scores.mean()),
            'recommended_action': self._get_portfolio_recommendation(engagement_scores)
        }
        
        return business_kpis
    
    def _get_portfolio_recommendation(self, engagement_scores: pd.Series) -> str:
        """Genera recomendación basada en scores de engagement"""
        avg_score = engagement_scores.mean()
        
        if avg_score > 0.7:
            return "Excellent portfolio - Continue current strategy"
        elif avg_score > 0.6:
            return "Good portfolio - Optimize medium performers"
        elif avg_score > 0.5:
            return "Average portfolio - Focus on high-potential tracks"
        else:
            return "Poor portfolio - Major strategy revision needed"
    
    def create_model_performance_data(self) -> Dict:
        """Crea datos de performance de modelos para dashboard técnico"""
        print("🤖 Generando datos de performance de modelos...")
        
        if not self.models_metrics:
            return {'error': 'No hay métricas de modelos disponibles'}
        
        model_performance = {
            'timestamp': datetime.now().isoformat(),
            'model_comparison': {},
            'best_model_details': {},
            'feature_importance': {},
            'model_stability': {}
        }
        
        # Comparación de modelos
        comparison_data = []
        for model_name, metrics in self.models_metrics.items():
            comparison_data.append({
                'model_name': model_name,
                'accuracy': float(metrics.get('accuracy', 0)),
                'f1_macro': float(metrics.get('f1_macro', 0)),
                'f1_weighted': float(metrics.get('f1_weighted', 0)),
                'precision': float(metrics.get('precision', 0)),
                'recall': float(metrics.get('recall', 0)),
                'auc_score': float(metrics.get('auc_ovr', 0)) if metrics.get('auc_ovr') else 0
            })
        
        model_performance['model_comparison'] = comparison_data
        
        # Mejor modelo
        if comparison_data:
            best_model = max(comparison_data, key=lambda x: x['accuracy'])
            model_performance['best_model_details'] = best_model
            
            # Feature importance del mejor modelo
            best_model_name = best_model['model_name']
            if best_model_name in self.models_metrics:
                feature_imp = self.models_metrics[best_model_name].get('feature_importance', {})
                if feature_imp:
                    # Top 10 features más importantes
                    sorted_features = sorted(feature_imp.items(), key=lambda x: x[1], reverse=True)[:10]
                    model_performance['feature_importance'] = {
                        'features': [f[0] for f in sorted_features],
                        'importance_scores': [float(f[1]) for f in sorted_features]
                    }
        
        return model_performance
    
    def create_time_series_data(self) -> Dict:
        """Crea datos de series temporales para análisis de tendencias"""
        print("📅 Generando datos de series temporales...")
        
        if self.gold_data.empty or 'year_min' not in self.gold_data.columns:
            return {'error': 'No hay datos temporales disponibles'}
        
        time_series_data = {
            'timestamp': datetime.now().isoformat(),
            'engagement_trends': {},
            'yearly_statistics': {},
            'seasonal_patterns': {}
        }
        
        # Tendencias de engagement por año
        yearly_engagement = self.gold_data.groupby('year_min').agg({
            'final_engagement_score': ['mean', 'std', 'count'],
            'total_plays': 'mean'
        }).round(4)
        
        yearly_stats = []
        for year in yearly_engagement.index:
            yearly_stats.append({
                'year': int(year),
                'avg_engagement': float(yearly_engagement.loc[year, ('final_engagement_score', 'mean')]),
                'engagement_std': float(yearly_engagement.loc[year, ('final_engagement_score', 'std')]),
                'song_count': int(yearly_engagement.loc[year, ('final_engagement_score', 'count')]),
                'avg_plays': float(yearly_engagement.loc[year, ('total_plays', 'mean')])
            })
        
        time_series_data['yearly_statistics'] = yearly_stats
        
        # Distribución de engagement por año
        engagement_by_year = self.gold_data.groupby(['year_min', 'engagement_category']).size().unstack(fill_value=0)
        
        yearly_distribution = []
        for year in engagement_by_year.index:
            total = engagement_by_year.loc[year].sum()
            yearly_distribution.append({
                'year': int(year),
                'high_count': int(engagement_by_year.loc[year].get('High', 0)),
                'medium_count': int(engagement_by_year.loc[year].get('Medium', 0)),
                'low_count': int(engagement_by_year.loc[year].get('Low', 0)),
                'high_percentage': float(engagement_by_year.loc[year].get('High', 0) / total * 100) if total > 0 else 0,
                'medium_percentage': float(engagement_by_year.loc[year].get('Medium', 0) / total * 100) if total > 0 else 0,
                'low_percentage': float(engagement_by_year.loc[year].get('Low', 0) / total * 100) if total > 0 else 0
            })
        
        time_series_data['engagement_trends'] = yearly_distribution
        
        return time_series_data
    
    def generate_looker_studio_datasets(self, output_dir: str) -> Dict:
        """Genera todos los datasets optimizados para Looker Studio"""
        print("📊 Generando datasets para Looker Studio...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        datasets_created = {}
        
        try:
            # 1. Executive Summary Dataset
            executive_data = self.create_executive_summary_data()
            exec_file = output_path / "executive_summary.json"
            with open(exec_file, 'w') as f:
                json.dump(executive_data, f, indent=2)
            datasets_created['executive_summary'] = str(exec_file)
            
            # 2. Engagement Analysis Dataset
            engagement_data = self.create_engagement_analysis_data()
            engagement_file = output_path / "engagement_analysis.json"
            with open(engagement_file, 'w') as f:
                json.dump(engagement_data, f, indent=2)
            datasets_created['engagement_analysis'] = str(engagement_file)
            
            # 3. Business KPIs Dataset
            business_data = self.create_business_kpis_data()
            business_file = output_path / "business_kpis.json"
            with open(business_file, 'w') as f:
                json.dump(business_data, f, indent=2)
            datasets_created['business_kpis'] = str(business_file)
            
            # 4. Model Performance Dataset
            model_data = self.create_model_performance_data()
            model_file = output_path / "model_performance.json"
            with open(model_file, 'w') as f:
                json.dump(model_data, f, indent=2)
            datasets_created['model_performance'] = str(model_file)
            
            # 5. Time Series Dataset
            timeseries_data = self.create_time_series_data()
            timeseries_file = output_path / "time_series_analysis.json"
            with open(timeseries_file, 'w') as f:
                json.dump(timeseries_data, f, indent=2)
            datasets_created['time_series'] = str(timeseries_file)
            
            # 6. Raw Data for Detailed Analysis (CSV para Looker Studio)
            if not self.gold_data.empty:
                # Dataset principal limpio para Looker Studio
                looker_main_data = self.gold_data[[
                    'spotify_track_uri', 'track_name_first', 'artist_name_first',
                    'final_engagement_score', 'engagement_category',
                    'completion_rate_score', 'skip_resistance_score', 'context_preference_score',
                    'total_plays', 'popularity_score', 'data_split'
                ]].copy()
                
                # Limpiar nombres de columnas para Looker Studio
                looker_main_data.columns = [
                    'Track_URI', 'Track_Name', 'Artist_Name',
                    'Engagement_Score', 'Engagement_Category',
                    'Completion_Rate', 'Skip_Resistance', 'Context_Preference',
                    'Total_Plays', 'Popularity_Score', 'Data_Split'
                ]
                
                main_data_file = output_path / "main_engagement_data.csv"
                looker_main_data.to_csv(main_data_file, index=False)
                datasets_created['main_data'] = str(main_data_file)
            
            print(f"✅ {len(datasets_created)} datasets creados para Looker Studio")
            
        except Exception as e:
            print(f"❌ Error generando datasets: {e}")
            datasets_created['error'] = str(e)
        
        return datasets_created
    
    def create_static_visualizations(self, output_dir: str) -> Dict:
        """Crea visualizaciones estáticas para reportes"""
        print("📈 Creando visualizaciones estáticas...")
        
        if self.gold_data.empty:
            return {'error': 'No hay datos para visualizar'}
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        visualizations_created = {}
        
        # 1. Dashboard Ejecutivo
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'Distribución de Engagement', 'Top 10 Artistas por Engagement',
                'Correlación Features vs Engagement', 'Tendencia por Año',
                'ROI Simulation', 'Performance de Modelos'
            ],
            specs=[[{"type": "pie"}, {"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "bar"}, {"type": "bar"}]]
        )
        
        # Distribución de engagement (pie chart)
        engagement_dist = self.gold_data['engagement_category'].value_counts()
        fig.add_trace(
            go.Pie(labels=engagement_dist.index, values=engagement_dist.values,
                   marker_colors=[self.color_palette['high_engagement'], 
                                self.color_palette['medium_engagement'], 
                                self.color_palette['low_engagement']]),
            row=1, col=1
        )
        
        # Top artistas por engagement promedio
        if 'artist_name_first' in self.gold_data.columns:
            top_artists = self.gold_data.groupby('artist_name_first')['final_engagement_score'].mean().nlargest(10)
            fig.add_trace(
                go.Bar(x=top_artists.values, y=top_artists.index, orientation='h',
                       marker_color=self.color_palette['primary']),
                row=1, col=2
            )
        
        # Correlación de features principales
        feature_cols = ['completion_rate_score', 'skip_resistance_score', 'context_preference_score']
        available_features = [col for col in feature_cols if col in self.gold_data.columns]
        
        if available_features:
            correlations = [self.gold_data[feat].corr(self.gold_data['final_engagement_score']) 
                          for feat in available_features]
            fig.add_trace(
                go.Bar(x=available_features, y=correlations,
                       marker_color=self.color_palette['accent']),
                row=1, col=3
            )
        
        fig.update_layout(height=800, showlegend=False, 
                         title_text="TuneMetrics - Dashboard Ejecutivo")
        
        # Guardar dashboard ejecutivo
        exec_dash_file = output_path / "executive_dashboard.html"
        fig.write_html(exec_dash_file)
        visualizations_created['executive_dashboard'] = str(exec_dash_file)
        
        # 2. Análisis de Engagement Detallado
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('TuneMetrics - Análisis Detallado de Engagement', fontsize=16, fontweight='bold')
        
        # Distribución de engagement score
        axes[0, 0].hist(self.gold_data['final_engagement_score'], bins=30, 
                       alpha=0.7, color=self.color_palette['primary'], edgecolor='black')
        axes[0, 0].set_title('Distribución de Engagement Score')
        axes[0, 0].set_xlabel('Engagement Score')
        axes[0, 0].set_ylabel('Frecuencia')
        
        # Box plot por categoría
        engagement_categories = ['Low', 'Medium', 'High']
        engagement_scores_by_cat = [
            self.gold_data[self.gold_data['engagement_category'] == cat]['final_engagement_score'].values
            for cat in engagement_categories if cat in self.gold_data['engagement_category'].values
        ]
        
        if engagement_scores_by_cat:
            axes[0, 1].boxplot(engagement_scores_by_cat, 
                              labels=[cat for cat in engagement_categories 
                                     if cat in self.gold_data['engagement_category'].values])
            axes[0, 1].set_title('Engagement Score por Categoría')
            axes[0, 1].set_ylabel('Engagement Score')
        
        # Scatter plot: Total Plays vs Engagement
        if 'total_plays' in self.gold_data.columns:
            scatter = axes[1, 0].scatter(self.gold_data['total_plays'], 
                                       self.gold_data['final_engagement_score'],
                                       alpha=0.6, c=self.gold_data['final_engagement_score'],
                                       cmap='viridis')
            axes[1, 0].set_title('Total Plays vs Engagement Score')
            axes[1, 0].set_xlabel('Total Plays')
            axes[1, 0].set_ylabel('Engagement Score')
            plt.colorbar(scatter, ax=axes[1, 0])
        
        # Heatmap de correlaciones
        if len(available_features) > 1:
            corr_matrix = self.gold_data[available_features + ['final_engagement_score']].corr()
            im = axes[1, 1].imshow(corr_matrix, cmap='coolwarm', aspect='auto')
            axes[1, 1].set_xticks(range(len(corr_matrix.columns)))
            axes[1, 1].set_yticks(range(len(corr_matrix.columns)))
            axes[1, 1].set_xticklabels(corr_matrix.columns, rotation=45)
            axes[1, 1].set_yticklabels(corr_matrix.columns)
            axes[1, 1].set_title('Matriz de Correlaciones')
            plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        
        # Guardar análisis detallado
        detailed_analysis_file = output_path / "detailed_engagement_analysis.png"
        plt.savefig(detailed_analysis_file, dpi=300, bbox_inches='tight')
        plt.close()
        visualizations_created['detailed_analysis'] = str(detailed_analysis_file)
        
        print(f"✅ {len(visualizations_created)} visualizaciones creadas")
        
        return visualizations_created
    
    def generate_dashboard_config_for_looker(self) -> Dict:
        """Genera configuración sugerida para dashboard en Looker Studio"""
        config = {
            "dashboard_structure": {
                "executive_page": {
                    "title": "TuneMetrics - Executive Summary",
                    "charts": [
                        {
                            "type": "scorecard",
                            "title": "Total Songs Analyzed",
                            "metric": "total_songs_analyzed",
                            "source": "executive_summary.json"
                        },
                        {
                            "type": "pie_chart", 
                            "title": "Engagement Distribution",
                            "dimensions": ["engagement_category"],
                            "metrics": ["song_count"],
                            "source": "main_engagement_data.csv"
                        },
                        {
                            "type": "bar_chart",
                            "title": "Investment Recommendations",
                            "dimensions": ["recommendation_type"],
                            "metrics": ["song_count"],
                            "source": "business_kpis.json"
                        }
                    ]
                },
                "detailed_analysis_page": {
                    "title": "TuneMetrics - Detailed Analysis", 
                    "charts": [
                        {
                            "type": "scatter_plot",
                            "title": "Engagement vs Popularity",
                            "x_axis": "Total_Plays",
                            "y_axis": "Engagement_Score",
                            "color": "Engagement_Category",
                            "source": "main_engagement_data.csv"
                        },
                        {
                            "type": "line_chart",
                            "title": "Engagement Trends Over Time",
                            "x_axis": "year",
                            "y_axis": "avg_engagement",
                            "source": "time_series_analysis.json"
                        }
                    ]
                },
                "model_performance_page": {
                    "title": "TuneMetrics - Model Performance",
                    "charts": [
                        {
                            "type": "bar_chart",
                            "title": "Model Comparison",
                            "dimensions": ["model_name"],
                            "metrics": ["accuracy", "f1_macro"],
                            "source": "model_performance.json"
                        }
                    ]
                }
            },
            "suggested_filters": [
                "Engagement_Category",
                "Data_Split", 
                "Artist_Name"
            ],
            "color_scheme": self.color_palette
        }
        
        return config

def main():
    """Función principal para generar dashboard de TuneMetrics"""
    
    # Configurar rutas de datos
    data_sources = {
        'gold_data': 'data/processed/gold_data.parquet',
        'predictions': 'reports/predictions_latest.csv', 
        'models_metrics': 'models/metrics/model_metrics.json'
    }
    
    # Directorios de salida
    dashboard_output_dir = "dashboard_data"
    visualizations_output_dir = "reports/visualizations"
    
    try:
        # Crear generador de dashboard
        dashboard_generator = TuneMetricsDashboardGenerator(data_sources)
        
        # Generar datasets para Looker Studio
        datasets = dashboard_generator.generate_looker_studio_datasets(dashboard_output_dir)
        print(f"\n📊 Datasets generados: {list(datasets.keys())}")
        
        # Crear visualizaciones estáticas
        visualizations = dashboard_generator.create_static_visualizations(visualizations_output_dir)
        print(f"📈 Visualizaciones creadas: {list(visualizations.keys())}")
        
        # Generar configuración para Looker Studio
        looker_config = dashboard_generator.generate_dashboard_config_for_looker()
        
        # Guardar configuración
        config_file = Path(dashboard_output_dir) / "looker_studio_config.json"
        with open(config_file, 'w') as f:
            json.dump(looker_config, f, indent=2)
        
        print(f"\n✅ Dashboard completo generado!")
        print(f"📁 Datasets: {dashboard_output_dir}")
        print(f"📁 Visualizaciones: {visualizations_output_dir}")
        print(f"⚙️  Configuración Looker: {config_file}")
        
        # Instrucciones para Looker Studio
        print(f"""
🔧 INSTRUCCIONES PARA LOOKER STUDIO:
1. Subir main_engagement_data.csv como fuente de datos principal
2. Usar archivos JSON para scorecards y métricas específicas
3. Aplicar esquema de colores sugerido en looker_studio_config.json
4. Configurar filtros por Engagement_Category y Artist_Name
5. Crear páginas según estructura sugerida en config
        """)
        
    except Exception as e:
        print(f"❌ Error generando dashboard: {e}")
        raise

if __name__ == "__main__":
    main()