import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def data_extractor(path):
	data = pd.read_csv(path)
	data = data.values

path = ''
