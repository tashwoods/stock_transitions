import matplotlib.pyplot as plt

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

