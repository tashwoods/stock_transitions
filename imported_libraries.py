import sys, os, math, shutil, time, argparse, matplotlib, csv, math, itertools
import random as stock_random
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
from sklearn.base import BaseEstimator, TransformerMixin #for attribute adder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from hmmlearn.hmm import GaussianHMM

from classes import *
from organize_input_output import *
from modelling import *
