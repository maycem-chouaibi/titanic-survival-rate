import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def histograms():
    # Load the data
    data = pd.read_csv('data/train_cleaned.csv')
    
    # # Create a histogram of the data
    # data.hist()
    
    # # Save the histogram
    # plt.savefig('histogram.png')
    
    # # Show the histogram
    # plt.show()

    # Plot heatmap
    sns.heatmap(data.corr(), annot=True)
    plt.savefig('heatmap.png')
    plt.show()
    
    return
