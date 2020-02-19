from imported_libraries import *

class stock_object_class:
  def __init__(self, file_name, stock_name, test_set, train_set, input_args):
    self.file_name = file_name
    self.stock_name = stock_name
    self.test_set = test_set
    self.train_set = train_set
    self.input_args = input_args

  def make_histograms(self, var):
    dataset = self.train_set
    fig, ax = plt.subplots()
    ax.hist(dataset[var])
    plt.xlabel(var)
    plt.ylabel('Count')
    plt.title(var + ' Histogram')
    var_stats = str(dataset[var].describe())
    var_stats = var_stats[:var_stats.find('Name')]
    plt.text(0.75, 0.6, str(var_stats), transform = ax.transAxes)
    plt.savefig(self.input_args.output_dir + '/' + self.stock_name + '/' + var + '_hist.pdf')
    plt.close('all')

  def make_overlay_plot(self):
    for var in self.train_set.columns:
      if var != self.input_args.date_name and var!= self.input_args.volume_name and var!= self.input_args.open_int_name:
        plt.plot(self.input_args.date_name, var, data = self.train_set, linewidth = 0.2)
    plt.legend()
    plt.savefig(self.input_args.output_dir + '/' + self.stock_name + '/stock_variables_overlay.pdf')

  def make_scatter_plots(self):
    for i in range(len(self.train_set.columns) - 1):
      for j in range(i+1, len(self.train_set.columns) - 1):
        var1 = self.train_set.iloc[:,i]
        var2 = self.train_set.iloc[:,j]

        fig = self.train_set.plot(kind = 'scatter', x = self.train_set.columns[i], y = self.train_set.columns[j], alpha = 0.1)
        plt.savefig(self.input_args.output_dir + '/' + self.stock_name + '/scatter_' + var1.name + '_' + var2.name + '.pdf')
        plt.close('all')
      
  def make_scatter_heat_plots(self):
    for i in range(len(self.train_set.columns) - 1):
      for j in range(i+1, len(self.train_set.columns) - 1):
        for k in range(j+1, len(self.train_set.columns) - 1):
          for l in range(k+1, len(self.train_set.columns) - 1):
            var1 = self.train_set.iloc[:,i]
            var2 = self.train_set.iloc[:,j]
            var3 = self.train_set.iloc[:,k]
            var4 = self.train_set.iloc[:,l]

            ax = self.train_set.plot.scatter(x = self.train_set.columns[i], y = self.train_set.columns[j], c = self.train_set.columns[l], colormap = 'viridis')
            plt.savefig(self.input_args.output_dir + '/' + self.stock_name + '/heat_scatter_' + var1.name + '_' + var2.name + '_' + var3.name + '_' + var4.name + '.pdf')
            plt.close('all')

  def make_correlation_plots(self):
    def make_complex_heatmap(x, y, size):
      fig, ax = plt.subplots()

      #Map from column names to integer coordinates
      x_labels = [v for v in sorted(x.unique())]
      y_labels = [v for v in sorted(y.unique())]
      x_to_num = {p[1]:p[0] for p in enumerate(x_labels)}
      y_to_num = {p[1]:p[0] for p in enumerate(y_labels)}

      size_scale = 500

      palette = sns.diverging_palette(self.input_args.pal_min, self.input_args.pal_max, n = self.input_args.n_colors)

      plot_grid = plt.GridSpec(1, 15, hspace=0.2, wspace=0.1) # Setup a 1x15 grid
      ax = plt.subplot(plot_grid[:,:-1]) # Use the leftmost 14 columns of the grid for the main plot

      ax.scatter(x = x.map(x_to_num), y = y.map(y_to_num), s = size * size_scale, c = size.apply(value_to_color) , marker = 's')

      ax.set_xticks([x_to_num[v] for v in x_labels])
      ax.set_xticklabels(x_labels, rotation = 45, horizontalalignment = 'right')
      ax.set_yticks([y_to_num[v] for v in y_labels])
      ax.set_yticklabels(y_labels)

      ax.grid(False, 'major')
      ax.grid(True, 'minor')
      ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor = True)
      ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor = True)
      ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
      ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])

      # Add color legend on the right side of the plot
      ax = plt.subplot(plot_grid[:,-1]) 
      col_x = [0]*len(palette) 
      bar_y=np.linspace(self.input_args.color_min, self.input_args.color_max, self.input_args.n_colors)
      bar_height = bar_y[1] - bar_y[0]
      ax.barh(y=bar_y,width=[5]*len(palette),height=bar_height,color=palette,linewidth=0)
      ax.set_xlim(1, 2) 
      ax.set_ylim(-1,1)
      ax.grid(False) 
      ax.set_facecolor('white')

      #Adjust ticks on correlation plot
      ax.set_xticks([]) # Remove horizontal ticks
      ax.set_yticks(np.linspace(min(bar_y), max(bar_y), 3)) # Show vertical ticks for min, middle and max
      ax.yaxis.tick_right() # Show vertical ticks on the right

      plt.savefig(self.input_args.output_dir + '/' + self.stock_name + '/correlations_heatmap_fancy.pdf')
     

    def value_to_color(val):
      palette = sns.diverging_palette(self.input_args.pal_min, self.input_args.pal_max, n = self.input_args.n_colors)
      if math.isnan(val):
        val = 0
      val_position = float((val - self.input_args.color_min))/(self.input_args.color_max - self.input_args.color_min) 
      ind = int(val_position * (self.input_args.n_colors - 1))
      palette_ind = palette[ind]

      return palette_ind

    #simple correlation plot
    corr = self.train_set.corr()
    axes = scatter_matrix(self.train_set[['Date', 'Volume', 'Open', 'High', 'Low', 'Close']], alpha = 0.2)
    for i, j in zip(*plt.np.triu_indices_from(axes, k=1)):
      axes[i,j].annotate("%.3f" %corr.iloc[i][j], (0.8, 0.8), xycoords = 'axes fraction', ha = 'center', va = 'center')
    plt.savefig(self.input_args.output_dir + '/' + self.stock_name + '/correlations_simple.pdf')
    plt.close('all')

    #heatmap correlation plot
    ax = sns.heatmap(corr, vmin = -1, vmax = 1, center = 0, square = True)
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, horizontalalignment = 'right')
    plt.savefig(self.input_args.output_dir + '/' + self.stock_name + '/correlations_heatmap.pdf')

    #heatmap with variable box size plot
    corr = pd.melt(corr.reset_index(), id_vars = 'index')
    corr.columns = ['x', 'y', 'value']
    make_complex_heatmap(corr['x'], corr['y'], size = corr['value'].abs())
    


def make_time_dependent_plots(self):
  time_plots = list()
  for var in self.train_set.columns:
    if var != args.date_name:
      ax = time_plots.append(self.train_set.plot(x = args.date_name, y = var))
      plt.xlabel('Date')
      plt.ylabel(var)
      plt.savefig(self.input_args.output_dir + '/' + self.stock_name + '/' + var + '_time.pdf')
  plt.close('all')
  return time_plots
