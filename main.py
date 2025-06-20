import pandas as pd
import seaborn as sns

def load_penguin_data():
    
    data = sns.load_dataset('penguins').dropna()
    return data.shape

if __name__ == "__main__":
    shape = load_penguin_data()
    print(f"Penguin dataset shape: {shape}")
