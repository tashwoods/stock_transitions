import sys, os
import pandas as pd

def main():
  filename=sys.argv[1]
  diagnostic_plots(filename)

   
def diagnostic_plots(filename):
  stock_name = extract_stock_name(filename)
  print('-------------------------------------------------------')
  print('STOCK: {}'.format(stock_name))
  print('DATA FROM: {}'.format(filename))

  formatted_data = get_data(filename)
  formatted_data.info()
  print(formatted_data.head())
   
def extract_stock_name(filename):
  if filename.endswith('.us.txt'):
    filename=filename[:-7].upper()
  return filename
    

def get_data(filename):
 return pd.read_csv(filename) 


 
  
if __name__ == '__main__':
    main()
