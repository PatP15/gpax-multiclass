
import annomi_common as common
import pandas as pd

try:
    df = common.load_annomi_data()
    print("\nClass Counts (y_mot):")
    print(df['y_mot'].value_counts().sort_index())
    print("\nMapping: 0=Sustain, 1=Neutral, 2=Change")
    
    print("\nOriginal Label Counts (main_therapist_behaviour):")
    if 'main_therapist_behaviour' in df.columns:
        print(df['main_therapist_behaviour'].value_counts())
        
    print("\nClient Motivation Label Counts (client_motivation):")
    if 'client_motivation' in df.columns:
        print(df['client_motivation'].value_counts())
        
except Exception as e:
    print(f"Error: {e}")

