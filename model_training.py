import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class InsuranceModelTrainer:
    def __init__(self):
        self.models = {}
        self.encoders = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        self.feature_importance = None
        self.training_history = []
        
    def load_data(self, file_path='medical_insurance_dataset.csv'):
        """Load and prepare the dataset"""
        try:
            self.df = pd.read_csv(file_path)
            print(f"✅ Dataset loaded successfully!")
            print(f"Dataset shape: {self.df.shape}")
            print(f"\nDataset info:")
            print(self.df.info())
            return True
        except FileNotFoundError:
            print("❌ Dataset not found. Please generate dataset first.")
            return False
        except Exception as e:
            print(f"❌ Error loading dataset: {str(e)}")
            return False
    
    def explore_data(self):
        """Perform exploratory data analysis"""
        print("\n" + "="*50)
        print("📊 EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        # Basic statistics
        print("\n📈 Basic Statistics:")
        print(self.df.describe())
        
        # Missing values
        print("\n🔍 Missing Values:")
        print(self.df.isnull().sum())
        
        # Data types
        print("\n📋 Data Types:")
        print(self.df.dtypes)
        
        # Unique values for categorical columns
        categorical_cols = ['sex', 'smoker', 'region']
        print("\n🏷️ Categorical Variables:")
        for col in categorical_cols:
            print(f"{col}: {self.df[col].unique()}")
        
        # Correlation analysis
        print("\n🔗 Correlation with Target (charges):")
        # Encode categorical variables for correlation
        df_corr = self.df.copy()
        for col in categorical_cols:
            le = LabelEncoder()
            df_corr[col] = le.fit_transform(df_corr[col])
        
        correlations = df_corr.corr()['charges'].sort_values(ascending=False)
        print(correlations)
        
        return correlations
    
    def visualize_data(self):
        """Create visualizations for data analysis"""
        print("\n📊 Creating visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('Medical Insurance Dataset Analysis', fontsize=16, fontweight='bold')
        
        # 1. Age distribution
        axes[0,0].hist(self.df['age'], bins=20, color='skyblue', alpha=0.7, edgecolor='black')
        axes[0,0].set_title('Age Distribution')
        axes[0,0].set_xlabel('Age')
        axes[0,0].set_ylabel('Frequency')
        
        # 2. BMI distribution
        axes[0,1].hist(self.df['bmi'], bins=20, color='lightgreen', alpha=0.7, edgecolor='black')
        axes[0,1].set_title('BMI Distribution')
        axes[0,1].set_xlabel('BMI')
        axes[0,1].set_ylabel('Frequency')
        
        # 3. Charges distribution
        axes[0,2].hist(self.df['charges'], bins=30, color='coral', alpha=0.7, edgecolor='black')
        axes[0,2].set_title('Insurance Charges Distribution')
        axes[0,2].set_xlabel('Charges ($)')
        axes[0,2].set_ylabel('Frequency')
        
        # 4. Sex distribution
        sex_counts = self.df['sex'].value_counts()
        axes[1,0].pie(sex_counts.values, labels=sex_counts.index, autopct='%1.1f%%', 
                      colors=['lightblue', 'pink'])
        axes[1,0].set_title('Gender Distribution')
        
        # 5. Smoker distribution
        smoker_counts = self.df['smoker'].value_counts()
        axes[1,1].pie(smoker_counts.values, labels=smoker_counts.index, autopct='%1.1f%%',
                      colors=['lightcoral', 'lightgreen'])
        axes[1,1].set_title('Smoking Status Distribution')
        
        # 6. Region distribution
        region_counts = self.df['region'].value_counts()
        axes[1,2].bar(region_counts.index, region_counts.values, 
                      color=['gold', 'lightblue', 'lightgreen', 'coral'])
        axes[1,2].set_title('Region Distribution')
        axes[1,2].tick_params(axis='x', rotation=45)
        
        # 7. Age vs Charges
        axes[2,0].scatter(self.df['age'], self.df['charges'], alpha=0.6, color='blue')
        axes[2,0].set_title('Age vs Insurance Charges')
        axes[2,0].set_xlabel('Age')
        axes[2,0].set_ylabel('Charges ($)')
        
        # 8. BMI vs Charges
        axes[2,1].scatter(self.df['bmi'], self.df['charges'], alpha=0.6, color='green')
        axes[2,1].set_title('BMI vs Insurance Charges')
        axes[2,1].set_xlabel('BMI')
        axes[2,1].set_ylabel('Charges ($)')
        
        # 9. Smoker vs Charges
        sns.boxplot(data=self.df, x='smoker', y='charges', ax=axes[2,2])
        axes[2,2].set_title('Smoking Status vs Insurance Charges')
        axes[2,2].set_xlabel('Smoker')
        axes[2,2].set_ylabel('Charges ($)')
        
        plt.tight_layout()
        plt.savefig('data_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualizations saved as 'data_analysis.png'")
    
    def preprocess_data(self):
        """Preprocess the data for training"""
        print("\n🔄 Preprocessing data...")
        
        # Make a copy of the dataframe
        self.df_processed = self.df.copy()
        
        # Encode categorical variables
        categorical_columns = ['sex', 'smoker', 'region']
        
        for col in categorical_columns:
            self.encoders[col] = LabelEncoder()
            self.df_processed[col] = self.encoders[col].fit_transform(self.df_processed[col])
        
        # Split features and target
        self.X = self.df_processed.drop('charges', axis=1)
        self.y = self.df_processed['charges']
        
        # Split into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=None
        )
        
        # Scale features for some models
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"✅ Data preprocessed successfully!")
        print(f"Training set size: {self.X_train.shape}")
        print(f"Test set size: {self.X_test.shape}")
        print(f"Feature columns: {list(self.X.columns)}")
        
        return True
    
    def train_multiple_models(self):
        """Train multiple models and compare performance"""
        print("\n🤖 Training multiple models...")
        print("="*50)
        
        # Define models to train
        models_to_train = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'SVR': SVR(kernel='rbf', C=1000, gamma=0.1)
        }
        
        results = {}
        
        for name, model in models_to_train.items():
            print(f"\n🔄 Training {name}...")
            
            try:
                # Use scaled data for SVR and linear models
                if name in ['SVR', 'Linear Regression', 'Ridge Regression', 'Lasso Regression']:
                    model.fit(self.X_train_scaled, self.y_train)
                    y_pred = model.predict(self.X_test_scaled)
                    
                    # Cross-validation
                    cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, 
                                              cv=5, scoring='neg_mean_absolute_error')
                else:
                    model.fit(self.X_train, self.y_train)
                    y_pred = model.predict(self.X_test)
                    
                    # Cross-validation
                    cv_scores = cross_val_score(model, self.X_train, self.y_train, 
                                              cv=5, scoring='neg_mean_absolute_error')
                
                # Calculate metrics
                mae = mean_absolute_error(self.y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
                r2 = r2_score(self.y_test, y_pred)
                cv_mae = -cv_scores.mean()
                cv_std = cv_scores.std()
                
                results[name] = {
                    'model': model,
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'cv_mae': cv_mae,
                    'cv_std': cv_std,
                    'predictions': y_pred
                }
                
                print(f"✅ {name} completed:")
                print(f"   MAE: ${mae:.2f}")
                print(f"   RMSE: ${rmse:.2f}")
                print(f"   R² Score: {r2:.4f}")
                print(f"   CV MAE: ${cv_mae:.2f} (±{cv_std:.2f})")
                
            except Exception as e:
                print(f"❌ Error training {name}: {str(e)}")
                continue
        
        self.models = results
        
        # Find best model based on lowest MAE
        best_model_name = min(results.keys(), key=lambda x: results[x]['mae'])
        self.best_model = results[best_model_name]['model']
        self.best_model_name = best_model_name
        
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"   MAE: ${results[best_model_name]['mae']:.2f}")
        print(f"   R² Score: {results[best_model_name]['r2']:.4f}")
        
        return results
    
    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning for the best models"""
        print("\n🔧 Hyperparameter Tuning...")
        print("="*40)
        
        # Hyperparameter grids
        param_grids = {
            'Random Forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'Gradient Boosting': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10]
            }
        }
        
        tuned_models = {}
        
        for model_name in ['Random Forest', 'Gradient Boosting']:
            if model_name in self.models:
                print(f"\n🔄 Tuning {model_name}...")
                
                # Get base model
                if model_name == 'Random Forest':
                    base_model = RandomForestRegressor(random_state=42)
                else:
                    base_model = GradientBoostingRegressor(random_state=42)
                
                # Grid search
                grid_search = GridSearchCV(
                    base_model,
                    param_grids[model_name],
                    cv=3,
                    scoring='neg_mean_absolute_error',
                    n_jobs=-1,
                    verbose=0
                )
                
                grid_search.fit(self.X_train, self.y_train)
                
                # Get best model
                best_model = grid_search.best_estimator_
                y_pred = best_model.predict(self.X_test)
                
                # Calculate metrics
                mae = mean_absolute_error(self.y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
                r2 = r2_score(self.y_test, y_pred)
                
                tuned_models[f"{model_name} (Tuned)"] = {
                    'model': best_model,
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'best_params': grid_search.best_params_,
                    'predictions': y_pred
                }
                
                print(f"✅ {model_name} tuning completed:")
                print(f"   Best MAE: ${mae:.2f}")
                print(f"   Best R² Score: {r2:.4f}")
                print(f"   Best Parameters: {grid_search.best_params_}")
        
        # Update models with tuned versions
        self.models.update(tuned_models)
        
        # Update best model if tuned version is better
        all_models = {**self.models, **tuned_models}
        best_model_name = min(all_models.keys(), key=lambda x: all_models[x]['mae'])
        
        if best_model_name != self.best_model_name:
            self.best_model = all_models[best_model_name]['model']
            self.best_model_name = best_model_name
            print(f"\n🏆 New Best Model: {best_model_name}")
        
        return tuned_models
    
    def analyze_feature_importance(self):
        """Analyze feature importance for tree-based models"""
        print("\n📊 Feature Importance Analysis...")
        
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            feature_names = self.X.columns
            
            # Create feature importance dataframe
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            self.feature_importance = importance_df
            
            print("\n📈 Feature Importance Rankings:")
            for idx, row in importance_df.iterrows():
                print(f"{row['feature']}: {row['importance']:.4f}")
            
            # Visualize feature importance
            plt.figure(figsize=(10, 6))
            sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
            plt.title(f'Feature Importance - {self.best_model_name}')
            plt.xlabel('Importance Score')
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            return importance_df
        else:
            print("❌ Best model doesn't support feature importance analysis")
            return None
    
    def create_performance_comparison(self):
        """Create performance comparison visualization"""
        print("\n📊 Creating performance comparison...")
        
        # Prepare data for visualization
        model_names = list(self.models.keys())
        mae_scores = [self.models[name]['mae'] for name in model_names]
        r2_scores = [self.models[name]['r2'] for name in model_names]
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # MAE comparison
        bars1 = ax1.bar(model_names, mae_scores, color='skyblue', alpha=0.7)
        ax1.set_title('Model Comparison - Mean Absolute Error')
        ax1.set_ylabel('MAE ($)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, mae in zip(bars1, mae_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'${mae:.0f}', ha='center', va='bottom')
        
        # R² comparison
        bars2 = ax2.bar(model_names, r2_scores, color='lightgreen', alpha=0.7)
        ax2.set_title('Model Comparison - R² Score')
        ax2.set_ylabel('R² Score')
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, r2 in zip(bars2, r2_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{r2:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Performance comparison saved as 'model_comparison.png'")
    
    def create_prediction_analysis(self):
        """Analyze predictions vs actual values"""
        print("\n📊 Creating prediction analysis...")
        
        best_predictions = self.models[self.best_model_name]['predictions']
        
        # Create prediction vs actual plot
        plt.figure(figsize=(12, 5))
        
        # Scatter plot
        plt.subplot(1, 2, 1)
        plt.scatter(self.y_test, best_predictions, alpha=0.6, color='blue')
        plt.plot([self.y_test.min(), self.y_test.max()], 
                 [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Charges ($)')
        plt.ylabel('Predicted Charges ($)')
        plt.title(f'Predictions vs Actual - {self.best_model_name}')
        
        # Residuals plot
        plt.subplot(1, 2, 2)
        residuals = self.y_test - best_predictions
        plt.scatter(best_predictions, residuals, alpha=0.6, color='green')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Charges ($)')
        plt.ylabel('Residuals ($)')
        plt.title('Residuals Plot')
        
        plt.tight_layout()
        plt.savefig('prediction_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Prediction analysis saved as 'prediction_analysis.png'")
    
    def save_best_model(self):
        """Save the best model and preprocessing components"""
        print("\n💾 Saving best model...")
        
        try:
            # Save model
            joblib.dump(self.best_model, 'best_insurance_model.pkl')
            
            # Save encoders
            joblib.dump(self.encoders, 'model_encoders.pkl')
            
            # Save scaler
            joblib.dump(self.scaler, 'model_scaler.pkl')
            
            # Save model metadata
            metadata = {
                'best_model_name': self.best_model_name,
                'performance_metrics': {
                    'mae': self.models[self.best_model_name]['mae'],
                    'rmse': self.models[self.best_model_name]['rmse'],
                    'r2': self.models[self.best_model_name]['r2']
                },
                'feature_columns': list(self.X.columns),
                'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            joblib.dump(metadata, 'model_metadata.pkl')
            
            print(f"✅ Best model ({self.best_model_name}) saved successfully!")
            print("   Files saved:")
            print("   - best_insurance_model.pkl")
            print("   - model_encoders.pkl")
            print("   - model_scaler.pkl")
            print("   - model_metadata.pkl")
            
        except Exception as e:
            print(f"❌ Error saving model: {str(e)}")
    
    def generate_model_report(self):
        """Generate comprehensive model training report"""
        print("\n📄 Generating Model Training Report...")
        
        report = f"""
{'='*80}
                    MEDICAL INSURANCE COST PREDICTION
                           MODEL TRAINING REPORT
{'='*80}

📊 DATASET SUMMARY:
   • Total Records: {len(self.df):,}
   • Features: {len(self.X.columns)}
   • Target Variable: Insurance Charges ($)
   • Training Set: {len(self.X_train):,} records
   • Test Set: {len(self.X_test):,} records

🎯 BEST MODEL PERFORMANCE:
   • Model: {self.best_model_name}
   • Mean Absolute Error (MAE): ${self.models[self.best_model_name]['mae']:.2f}
   • Root Mean Square Error (RMSE): ${self.models[self.best_model_name]['rmse']:.2f}
   • R² Score: {self.models[self.best_model_name]['r2']:.4f}
   • Accuracy: {self.models[self.best_model_name]['r2']*100:.1f}%

📈 ALL MODELS COMPARISON:
"""
        
        for name, metrics in self.models.items():
            report += f"""
   {name}:
     • MAE: ${metrics['mae']:.2f}
     • RMSE: ${metrics['rmse']:.2f}  
     • R² Score: {metrics['r2']:.4f}
"""
        
        if self.feature_importance is not None:
            report += f"""
🔍 FEATURE IMPORTANCE (Top 5):
"""
            for idx, row in self.feature_importance.head().iterrows():
                report += f"   {idx+1}. {row['feature']}: {row['importance']:.4f}\n"
        
        report += f"""
💡 MODEL INSIGHTS:
   • The model explains {self.models[self.best_model_name]['r2']*100:.1f}% of the variance in insurance costs
   • Average prediction error is ${self.models[self.best_model_name]['mae']:.2f}
   • Model is suitable for cost estimation and risk assessment
   
📁 SAVED FILES:
   • best_insurance_model.pkl - Trained model
   • model_encoders.pkl - Label encoders
   • model_scaler.pkl - Feature scaler
   • model_metadata.pkl - Model information
   • data_analysis.png - Dataset visualizations
   • model_comparison.png - Performance comparison
   • feature_importance.png - Feature importance chart
   • prediction_analysis.png - Prediction analysis
   
🕒 Training Completed: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
"""
        
        # Save report to file
        with open('model_training_report.txt', 'w') as f:
            f.write(report)
        
        print(report)
        print("✅ Report saved as 'model_training_report.txt'")
    
    def run_complete_training_pipeline(self):
        """Run the complete model training pipeline"""
        print("🚀 Starting Complete Model Training Pipeline...")
        print("="*60)
        
        steps = [
            ("Loading Data", self.load_data),
            ("Exploring Data", self.explore_data),
            ("Creating Visualizations", self.visualize_data),
            ("Preprocessing Data", self.preprocess_data),
            ("Training Multiple Models", self.train_multiple_models),
            ("Hyperparameter Tuning", self.hyperparameter_tuning),
            ("Feature Importance Analysis", self.analyze_feature_importance),
            ("Performance Comparison", self.create_performance_comparison),
            ("Prediction Analysis", self.create_prediction_analysis),
            ("Saving Best Model", self.save_best_model),
            ("Generating Report", self.generate_model_report)
        ]
        
        for step_name, step_function in steps:
            try:
                print(f"\n{'🔄'} Step: {step_name}")
                result = step_function()
                if result is False:
                    print(f"❌ Failed at step: {step_name}")
                    return False
                print(f"✅ Completed: {step_name}")
            except Exception as e:
                print(f"❌ Error in {step_name}: {str(e)}")
                return False
        
        print(f"\n🎉 Training pipeline completed successfully!")
        print(f"🏆 Best Model: {self.best_model_name}")
        print(f"📊 Performance: MAE=${self.models[self.best_model_name]['mae']:.2f}, R²={self.models[self.best_model_name]['r2']:.4f}")
        
        return True

def main():
    """Main function to run model training"""
    trainer = InsuranceModelTrainer()
    
    print("🏥 Medical Insurance Cost Prediction - Model Training")
    print("="*60)
    
    # Ask user for training mode
    print("\nSelect training mode:")
    print("1. Complete Pipeline (Recommended)")
    print("2. Quick Training (Basic models only)")
    print("3. Custom Training (Select specific steps)")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == '1':
        trainer.run_complete_training_pipeline()
    
    elif choice == '2':
        if trainer.load_data():
            trainer.preprocess_data()
            trainer.train_multiple_models()
            trainer.save_best_model()
            print(f"\n🎉 Quick training completed!")
            print(f"🏆 Best Model: {trainer.best_model_name}")
    
    elif choice == '3':
        # Custom training - let user select steps
        if trainer.load_data():
            trainer.preprocess_data()
            
            steps = {
                '1': ('Explore Data', trainer.explore_data),
                '2': ('Visualize Data', trainer.visualize_data),
                '3': ('Train Models', trainer.train_multiple_models),
                '4': ('Hyperparameter Tuning', trainer.hyperparameter_tuning),
                '5': ('Feature Analysis', trainer.analyze_feature_importance),
                '6': ('Save Model', trainer.save_best_model)
            }
            
            print("\nAvailable steps:")
            for key, (name, _) in steps.items():
                print(f"{key}. {name}")
            
            selected = input("\nEnter step numbers (comma-separated): ").strip().split(',')
            
            for step_num in selected:
                step_num = step_num.strip()
                if step_num in steps:
                    name, func = steps[step_num]
                    print(f"\n🔄 Running: {name}")
                    func()
    
    else:
        print("❌ Invalid choice. Exiting.")

if __name__ == "__main__":
    main()