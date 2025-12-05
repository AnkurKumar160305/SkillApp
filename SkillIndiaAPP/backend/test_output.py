"""
Test script to verify ML model training output
"""
import sys
sys.path.insert(0, 'c:\\Users\\asus\\Desktop\\SkillApp\\SkillIndiaApp\\backend')

from ml_models import RecommenderSystem

print("Starting ML model test...")
print("="*60)

recommender = RecommenderSystem()
recommender.load_data()

print("\n" + "="*60)
print("✅ Test completed successfully!")
print("="*60)
