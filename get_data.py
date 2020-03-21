from imported_libraries import *

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description = 'arg parser for get_data.py')
  parser.add_argument('-o', '--output_dir', type = str, dest = 'output_dir', default = 'newoutput', help = 'name of output_dir')
  parser.add_argument('-f', '--input_file', type = str, dest = 'input_file', default = 'all_stock_names.txt', help = 'text file with input file names')
  parser.add_argument('-d', '--date_name', type = str, dest = 'date_name', default = 'Date', help = 'name of Date variable in dataset')
  args = parser.parse_args()

  #Organize and format input and output
  make_output_dir(args.output_dir)
  
  #Process input file list
  input_file = open(args.input_file, "r")
  for stock in input_file:
    if len(stock.strip()) > 0:
      stock = stock.rstrip()
      data = yf.download(stock)
      data[args.date_name] = data.index #for backwards compatibility
      data.to_csv(args.output_dir + '/' + stock + '.us.txt', sep = ',', mode = 'a')



