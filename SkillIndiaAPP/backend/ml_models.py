import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
import os
import joblib
import sys

class RecommenderSystem:
    def __init__(self):
        self.jobs_df = None
        self.courses_df = None
        
        # Models
        self.knn_model = None
        self.svm_model = None
        self.lr_model = None
        
        # Best model tracking
        self.best_model = None
        self.best_model_name = None
        self.best_accuracy = 0.0
        
        # Model accuracies
        self.model_accuracies = {}
        
        # Vectorizers and matrices
        self.job_vectorizer = None
        self.job_matrix = None
        self.course_vectorizer = None
        self.course_matrix = None

        # Paths
        self.base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

    def load_data(self):
        jobs_path = os.path.join(self.base_path, "Naukri_Jobs_Data.csv")
        courses_path = os.path.join(self.base_path, "udemy_courses.csv")

        print("\n" + "="*60)
        print("🚀 SKILL DEVELOPMENT APP - ML MODEL INITIALIZATION")
        print("="*60)
        sys.stdout.flush()
        
        print("\n📂 Loading data...")
        sys.stdout.flush()
        self.jobs_df = pd.read_csv(jobs_path)
        self.courses_df = pd.read_csv(courses_path)
        print(f"   ✓ Loaded {len(self.jobs_df)} jobs")
        print(f"   ✓ Loaded {len(self.courses_df)} courses")
        sys.stdout.flush()
        
        # Preprocessing
        self.preprocess_jobs()
        self.preprocess_courses()
        
        # Try to load saved models
        if self.load_artifacts():
            print("\n✓ Loaded saved models and artifacts.")
            print(f"   Model: {self.best_model_name}")
            print(f"   Accuracy: {self.best_accuracy:.4f}")
            print("\n" + "="*60)
            print("✅ INITIALIZATION COMPLETE - Ready to serve recommendations!")
            print("="*60 + "\n")
            sys.stdout.flush()
        else:
            # Train all models and select best
            print("\n⚙️  Training and evaluating models...")
            sys.stdout.flush()
            self.train_all_models()
            print(f"\n✓ Best model selected: {self.best_model_name} (Accuracy: {self.best_accuracy:.4f})")
            sys.stdout.flush()
            self.save_artifacts()
            print("\n" + "="*60)
            print("✅ INITIALIZATION COMPLETE - Ready to serve recommendations!")
            print("="*60 + "\n")
            sys.stdout.flush()

    def preprocess_jobs(self):
        # Fill missing values
        self.jobs_df['required_skills'] = self.jobs_df['required_skills'].fillna('')
        self.jobs_df['job_post'] = self.jobs_df['job_post'].fillna('')
        self.jobs_df['job_description'] = self.jobs_df['job_description'].fillna('')
        
        # Combine features
        self.jobs_df['combined_features'] = (
            self.jobs_df['job_post'] + " " + 
            self.jobs_df['required_skills'] + " " + 
            self.jobs_df['job_description']
        )
        self.jobs_df['combined_features'] = self.jobs_df['combined_features'].str.lower()

    def preprocess_courses(self):
        self.courses_df['course_title'] = self.courses_df['course_title'].fillna('')
        self.courses_df['subject'] = self.courses_df['subject'].fillna('')
        
        self.courses_df['combined_features'] = (
            self.courses_df['course_title'] + " " + 
            self.courses_df['subject']
        )
        self.courses_df['combined_features'] = self.courses_df['combined_features'].str.lower()

    def prepare_training_data(self, data_type='jobs'):
        """
        Create labeled training data for classification.
        We'll create synthetic labels based on skill categories.
        """
        if data_type == 'jobs':
            df = self.jobs_df.copy()
            text_col = 'combined_features'
            
            # Create categories based on common job types
            def categorize_job(text):
                text = text.lower()
                if any(word in text for word in ['python', 'machine learning', 'data science', 'ai', 'analytics']):
                    return 0  # Data Science
                elif any(word in text for word in ['java', 'javascript', 'react', 'angular', 'frontend', 'backend']):
                    return 1  # Software Development
                elif any(word in text for word in ['design', 'ui', 'ux', 'graphic', 'figma']):
                    return 2  # Design
                elif any(word in text for word in ['marketing', 'sales', 'business', 'management']):
                    return 3  # Business
                else:
                    return 4  # Other
            
            df['category'] = df[text_col].apply(categorize_job)
            
        else:  # courses
            df = self.courses_df.copy()
            text_col = 'combined_features'
            
            def categorize_course(text):
                text = text.lower()
                if any(word in text for word in ['python', 'machine learning', 'data', 'ai', 'analytics']):
                    return 0
                elif any(word in text for word in ['java', 'javascript', 'web', 'programming', 'development']):
                    return 1
                elif any(word in text for word in ['design', 'ui', 'ux', 'graphic']):
                    return 2
                elif any(word in text for word in ['business', 'marketing', 'finance', 'management']):
                    return 3
                else:
                    return 4
            
            df['category'] = df[text_col].apply(categorize_course)
        
        return df[text_col].values, df['category'].values

    def train_all_models(self, use_full_training_metrics=False):
        """
        Train KNN, SVM, and Linear Regression models and compare accuracy.
        
        Args:
            use_full_training_metrics (bool): If True, compute training metrics on full training set.
                                             If False (default), use subset for performance.
        """
        
        # Prepare training data
        X_text, y = self.prepare_training_data('jobs')
        
        # Vectorize
        print("Vectorizing job data...")
        self.job_vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
        X = self.job_vectorizer.fit_transform(X_text)
        self.job_matrix = X
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Print comprehensive dataset information
        print(f"\n{'='*60}")
        print(f"DATASET INFORMATION")
        print(f"{'='*60}")
        print(f"Total samples: {X.shape[0]}")
        print(f"Training samples: {X_train.shape[0]} ({(X_train.shape[0]/X.shape[0]*100):.1f}%)")
        print(f"Testing samples: {X_test.shape[0]} ({(X_test.shape[0]/X.shape[0]*100):.1f}%)")
        print(f"Number of features: {X.shape[1]}")
        print(f"Number of categories: {len(np.unique(y))}")
        print(f"\nCategory distribution in full dataset:")
        unique, counts = np.unique(y, return_counts=True)
        for cat, count in zip(unique, counts):
            print(f"  Category {cat}: {count} samples ({count/len(y)*100:.1f}%)")
        print(f"{'='*60}\n")
        sys.stdout.flush()
        
        # Train and evaluate each model
        models = {
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'SVM': SVC(kernel='rbf', random_state=42),
            'Linear Regression': LogisticRegression(max_iter=1000, random_state=42)
        }

        for name, model in models.items():
            print(f"\n{'='*20} {name} {'='*20}")
            sys.stdout.flush()
            
            # Train
            model.fit(X_train, y_train)
            
            # --- Training Metrics ---
            if use_full_training_metrics:
                # Use FULL training set (slower but complete)
                print(f"\n📊 TRAINING Data Metrics (FULL TRAINING SET: {X_train.shape[0]} samples)")
                y_train_pred = model.predict(X_train)
                train_accuracy = accuracy_score(y_train, y_train_pred)
                train_precision = precision_score(y_train, y_train_pred, average='weighted', zero_division=0)
                train_recall = recall_score(y_train, y_train_pred, average='weighted', zero_division=0)
                train_f1 = f1_score(y_train, y_train_pred, average='weighted', zero_division=0)
            else:
                # Use subset for performance (faster)
                subset_size = min(2000, X_train.shape[0])
                X_train_subset = X_train[:subset_size]
                y_train_subset = y_train[:subset_size]
                
                print(f"\n📊 TRAINING Data Metrics (Subset: {subset_size}/{X_train.shape[0]} samples - {subset_size/X_train.shape[0]*100:.1f}%)")
                y_train_pred = model.predict(X_train_subset)
                train_accuracy = accuracy_score(y_train_subset, y_train_pred)
                train_precision = precision_score(y_train_subset, y_train_pred, average='weighted', zero_division=0)
                train_recall = recall_score(y_train_subset, y_train_pred, average='weighted', zero_division=0)
                train_f1 = f1_score(y_train_subset, y_train_pred, average='weighted', zero_division=0)

            print(f"  Accuracy:  {train_accuracy:.4f}")
            print(f"  Precision: {train_precision:.4f}")
            print(f"  Recall:    {train_recall:.4f}")
            print(f"  F1-Score:  {train_f1:.4f}")
            sys.stdout.flush()

            # --- Testing Metrics (FULL TEST SET) ---
            y_test_pred = model.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
            test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
            test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
            
            print(f"\n✅ TESTING Data Metrics (FULL TEST SET: {X_test.shape[0]} samples)")
            print(f"  Accuracy:  {test_accuracy:.4f}")
            print(f"  Precision: {test_precision:.4f}")
            print(f"  Recall:    {test_recall:.4f}")
            print(f"  F1-Score:  {test_f1:.4f}")
            sys.stdout.flush()
            
            # Store results (using test accuracy for selection)
            self.model_accuracies[name] = {
                'accuracy': test_accuracy,
                'precision': test_precision,
                'recall': test_recall,
                'f1_score': test_f1
            }
            
            # Store models
            if name == 'KNN':
                self.knn_model = model
            elif name == 'SVM':
                self.svm_model = model
            elif name == 'Linear Regression':
                self.lr_model = model
            
            # Track best model
            if test_accuracy > self.best_accuracy:
                self.best_accuracy = test_accuracy
                self.best_model = model
                self.best_model_name = name
        
        # Print comprehensive comparison table
        print(f"\n{'='*60}")
        print(f"📊 MODEL COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"\n{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print(f"{'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
        
        for model_name, metrics in self.model_accuracies.items():
            marker = " ⭐ BEST" if model_name == self.best_model_name else ""
            print(f"{model_name:<20} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} {metrics['f1_score']:<12.4f}{marker}")
        
        print(f"\n{'='*60}")
        print(f"✅ Best Model: {self.best_model_name}")
        print(f"   Test Accuracy: {self.best_accuracy:.4f}")
        print(f"{'='*60}\n")
        sys.stdout.flush()
        
        # Prepare course vectorizer
        print("\n📚 Vectorizing course data...")
        sys.stdout.flush()
        self.course_vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
        self.course_matrix = self.course_vectorizer.fit_transform(self.courses_df['combined_features'])

    def save_artifacts(self):
        print("Saving models and artifacts...")
        artifacts = {
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'best_accuracy': self.best_accuracy,
            'model_accuracies': self.model_accuracies,
            'job_vectorizer': self.job_vectorizer,
            'job_matrix': self.job_matrix,
            'course_vectorizer': self.course_vectorizer,
            'course_matrix': self.course_matrix
        }
        joblib.dump(artifacts, os.path.join(self.models_dir, "artifacts.pkl"))
        print("✓ Artifacts saved.")

    def load_artifacts(self):
        artifacts_path = os.path.join(self.models_dir, "artifacts.pkl")
        if not os.path.exists(artifacts_path):
            return False
        
        try:
            print("Loading saved artifacts...")
            artifacts = joblib.load(artifacts_path)
            
            self.best_model = artifacts['best_model']
            self.best_model_name = artifacts['best_model_name']
            self.best_accuracy = artifacts['best_accuracy']
            self.model_accuracies = artifacts['model_accuracies']
            self.job_vectorizer = artifacts['job_vectorizer']
            self.job_matrix = artifacts['job_matrix']
            self.course_vectorizer = artifacts['course_vectorizer']
            self.course_matrix = artifacts['course_matrix']
            return True
        except Exception as e:
            print(f"Error loading artifacts: {e}")
            return False

    def recommend_jobs(self, skills, top_n=5):
        """Use best model to recommend jobs based on user skills."""
        # Transform user skills
        user_vec = self.job_vectorizer.transform([skills.lower()])
        
        # Compute similarity with all jobs
        similarities = cosine_similarity(user_vec, self.job_matrix).flatten()
        
        # Get top N indices
        top_indices = similarities.argsort()[-top_n:][::-1]
        
        # Return results
        results = self.jobs_df.iloc[top_indices][['job_post', 'company', 'required_skills', 'job_location']].to_dict(orient='records')
        return results

    def recommend_courses(self, skills, top_n=5):
        """Use best model to recommend courses based on user skills."""
        # Transform user skills
        user_vec = self.course_vectorizer.transform([skills.lower()])
        
        # Compute similarity with all courses
        similarities = cosine_similarity(user_vec, self.course_matrix).flatten()
        
        # Get top N indices
        top_indices = similarities.argsort()[-top_n:][::-1]
        
        # Return results
        results = self.courses_df.iloc[top_indices][['course_title', 'url', 'price', 'level']].to_dict(orient='records')
        return results

    def get_model_info(self):
        """Return information about the best model and all model accuracies."""
        return {
            'best_model': self.best_model_name,
            'best_accuracy': self.best_accuracy,
            'all_models': self.model_accuracies
        }

# Singleton instance
recommender = RecommenderSystem()
