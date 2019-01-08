import matplotlib
import matplotlib.pyplot as plt

def initialize_plot_settings():
   
    matplotlib.rcParams['figure.figsize'] = (8, 8)
    matplotlib.rcParams['font.size'] = 14
    matplotlib.rcParams['axes.labelsize'] = 18
    matplotlib.rcParams['xtick.labelsize'] = 14
    matplotlib.rcParams['ytick.labelsize'] = 14
    matplotlib.rcParams['legend.fontsize'] = 14

    matplotlib.rcParams['axes.labelcolor'] ='k'

    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
    matplotlib.rcParams['axes.spines.right'] = False
    matplotlib.rcParams['axes.spines.top'] = False
    matplotlib.rcParams['axes.grid'] = False
    
    matplotlib.rcParams['axes.facecolor'] = 'w'
    matplotlib.rcParams['axes.edgecolor'] = 'k'
    matplotlib.rcParams['xtick.color'] = 'k'
    matplotlib.rcParams['ytick.color'] = 'k'
    
    matplotlib.rcParams['legend.frameon'] = False
    
    matplotlib.rcParams['savefig.dpi'] = 300 
    matplotlib.rcParams['savefig.pad_inches'] = 0
