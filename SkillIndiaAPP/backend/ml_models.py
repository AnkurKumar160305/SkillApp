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

    def load_data(self):
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        jobs_path = os.path.join(base_path, "Naukri_Jobs_Data.csv")
        courses_path = os.path.join(base_path, "udemy_courses.csv")

        print("Loading data...")
        self.jobs_df = pd.read_csv(jobs_path)
        self.courses_df = pd.read_csv(courses_path)
        
        # Preprocessing
        self.preprocess_jobs()
        self.preprocess_courses()
        
        # Train all models and select best
        print("\nTraining and evaluating models...")
        self.train_all_models()
        print(f"\n✓ Best model selected: {self.best_model_name} (Accuracy: {self.best_accuracy:.4f})")

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

    def train_all_models(self):
        """Train KNN, SVM, and Linear Regression models and compare accuracy."""
        
        # Prepare training data
        X_text, y = self.prepare_training_data('jobs')
        
        # Vectorize
        print("Vectorizing job data...")
        self.job_vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
        X = self.job_vectorizer.fit_transform(X_text)
        self.job_matrix = X
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train and evaluate each model
        models = {
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'SVM': SVC(kernel='rbf', random_state=42),
            'Linear Regression': LogisticRegression(max_iter=1000, random_state=42)
        }
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Evaluate
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Store results
            self.model_accuracies[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            
            # Store models
            if name == 'KNN':
                self.knn_model = model
            elif name == 'SVM':
                self.svm_model = model
            elif name == 'Linear Regression':
                self.lr_model = model
            
            # Track best model
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_model = model
                self.best_model_name = name
        
        # Prepare course vectorizer
        print("\nVectorizing course data...")
        self.course_vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
        self.course_matrix = self.course_vectorizer.fit_transform(self.courses_df['combined_features'])

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
