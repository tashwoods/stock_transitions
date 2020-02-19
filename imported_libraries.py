import sys, os, math, shutil, time, argparse, matplotlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
from pylab import *
from pandas.plotting import scatter_matrix
import seaborn as sns
from matplotlib.pyplot import cm
import multiprocessing
matplotlib.use("Agg") #disable python gui rocket in mac os dock
import matplotlib.pyplot as plt
from functools import partial
