import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
import os
import joblib
import sys
import csv

class RecommenderSystem:
    def __init__(self):
        self.jobs_df = []  # List of dicts
        self.courses_df = [] # List of dicts
        
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
        print(f"Base Path: {self.base_path}")
        print(f"Jobs Path: {jobs_path} (Exists: {os.path.exists(jobs_path)})")
        print(f"Courses Path: {courses_path} (Exists: {os.path.exists(courses_path)})")
        print("="*60)
        sys.stdout.flush()
        
        print("\n📂 Loading data...")
        sys.stdout.flush()
        
        # Load Jobs
        try:
            with open(jobs_path, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                self.jobs_df = [row for row in reader]
        except Exception as e:
            print(f"Error loading jobs CSV: {e}")
            self.jobs_df = []
            
        # Load Courses
        try:
            with open(courses_path, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                self.courses_df = [row for row in reader]
        except Exception as e:
            print(f"Error loading courses CSV: {e}")
            self.courses_df = []

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
        for row in self.jobs_df:
            skills = row.get('required_skills') or ''
            post = row.get('job_post') or ''
            desc = row.get('job_description') or ''
            
            combined = f"{post} {skills} {desc}".lower()
            row['combined_features'] = combined

    def preprocess_courses(self):
        for row in self.courses_df:
            title = row.get('course_title') or ''
            subj = row.get('subject') or ''
            
            combined = f"{title} {subj}".lower()
            row['combined_features'] = combined

    def prepare_training_data(self, data_type='jobs'):
        if data_type == 'jobs':
            data = self.jobs_df
            
            def categorize_job(text):
                text = text.lower()
                if any(word in text for word in ['python', 'machine learning', 'data science', 'ai', 'analytics']):
                    return 0
                elif any(word in text for word in ['java', 'javascript', 'react', 'angular', 'frontend', 'backend']):
                    return 1
                elif any(word in text for word in ['design', 'ui', 'ux', 'graphic', 'figma']):
                    return 2
                elif any(word in text for word in ['marketing', 'sales', 'business', 'management']):
                    return 3
                else:
                    return 4
            
            X_text = [row.get('combined_features', '') for row in data]
            y = [categorize_job(row.get('combined_features', '')) for row in data]
            
        else:  # courses
            data = self.courses_df
            
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
            
            X_text = [row.get('combined_features', '') for row in data]
            y = [categorize_course(row.get('combined_features', '')) for row in data]
        
        return np.array(X_text), np.array(y)

    def train_all_models(self):
        X_text, y = self.prepare_training_data('jobs')
        
        if len(X_text) == 0:
            print("Warning: No data for training.")
            return

        self.job_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        X = self.job_vectorizer.fit_transform(X_text)
        self.job_matrix = X
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        from sklearn.ensemble import RandomForestClassifier
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42),
            'Support Vector Machine': SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
        }

        self.best_accuracy = 0.0
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)
            test_acc = accuracy_score(y_test, y_test_pred)
            
            self.model_accuracies[name] = {'accuracy': float(test_acc)}
            
            if test_acc > self.best_accuracy:
                self.best_accuracy = float(test_acc)
                self.best_model = model
                self.best_model_name = name
        
        self.course_vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
        course_features = [row.get('combined_features', '') for row in self.courses_df]
        if course_features:
            self.course_matrix = self.course_vectorizer.fit_transform(course_features)

    def save_artifacts(self):
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

    def load_artifacts(self):
        artifacts_path = os.path.join(self.models_dir, "artifacts.pkl")
        if not os.path.exists(artifacts_path):
            return False
        
        try:
            artifacts = joblib.load(artifacts_path)
            self.best_model = artifacts.get('best_model')
            self.best_model_name = artifacts.get('best_model_name')
            self.best_accuracy = artifacts.get('best_accuracy')
            self.model_accuracies = artifacts.get('model_accuracies')
            self.job_vectorizer = artifacts.get('job_vectorizer')
            self.job_matrix = artifacts.get('job_matrix')
            self.course_vectorizer = artifacts.get('course_vectorizer')
            self.course_matrix = artifacts.get('course_matrix')
            return True
        except Exception as e:
            print(f"Error loading artifacts: {e}")
            return False

    def recommend_jobs(self, skills, top_n=5):
        if self.job_vectorizer is None or self.job_matrix is None:
            return []
        user_vec = self.job_vectorizer.transform([skills.lower()])
        similarities = cosine_similarity(user_vec, self.job_matrix).flatten()
        top_indices = similarities.argsort()[-top_n:][::-1]
        
        results = []
        for idx in top_indices:
            if idx < len(self.jobs_df):
                row = self.jobs_df[idx]
                results.append({
                    'job_post': row.get('job_post', ''),
                    'company': row.get('company', ''),
                    'required_skills': row.get('required_skills', ''),
                    'job_location': row.get('job_location', '')
                })
        return results

    def recommend_courses(self, skills, top_n=5):
        if self.course_vectorizer is None or self.course_matrix is None:
            return []
        user_vec = self.course_vectorizer.transform([skills.lower()])
        similarities = cosine_similarity(user_vec, self.course_matrix).flatten()
        top_indices = similarities.argsort()[-top_n:][::-1]
        
        results = []
        for idx in top_indices:
            if idx < len(self.courses_df):
                row = self.courses_df[idx]
                results.append({
                    'course_title': row.get('course_title', ''),
                    'url': row.get('url', ''),
                    'price': row.get('price', ''),
                    'level': row.get('level', '')
                })
        return results

    def get_model_info(self):
        return {
            'best_model': self.best_model_name,
            'best_accuracy': self.best_accuracy,
            'all_models': self.model_accuracies
        }

recommender = RecommenderSystem()
