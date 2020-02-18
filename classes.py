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

    

