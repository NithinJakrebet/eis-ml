import matplotlib.pyplot as plt
import pandas as pd

def plot_degradation(df: pd.DataFrame):
      capacity = df['Capacity/mA.h']
      cycle_number = df['cycle number']
      
      plt.figure(figsize=(8, 8))
      plt.scatter(cycle_number, capacity, alpha=0.6)
      plt.xlabel('Cycle Number')
      plt.ylabel('Capacity (mA.h)')
      plt.title('Capacity vs. Cycle Number')
      plt.show()

      
      