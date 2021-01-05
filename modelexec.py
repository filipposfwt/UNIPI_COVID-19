import os
import numpy as np
import Extended_SEIRS_model
import pandas as pd
import sys, getopt

def main(argv):
   scenario = ''
   runs = 1
   plot = False
   outfile = ''
   vacoutfile = ''
   try:
      opts, args = getopt.getopt(argv,"hs:r:po:v:",["help","scenario=","runs=","plot","outfile=","vaccination="])
   except getopt.GetoptError:
      print ('modelexec.py -s [scenario] -r [number of runs] -o [output file] -v [vaccination output file]')
      sys.exit(2)
   for opt, arg in opts:
      if opt in("-h","--help"):
         print ('modelexec.py -s [scenario] -r [number of runs] -o [output file] -v [vaccination output file]')
         sys.exit()
      elif opt in ("-s", "--scenario"):
         scenario = arg
         print('Running',scenario,'scenario...')
      elif opt in ("-r", "--runs"):
         runs = int(arg)
         print('Running',runs,'runs...')
      elif opt in ("-p","--plot"):
         plot = True
         print('Plotting is enabled')
      elif opt in ("-o","--outfile"):
         outfile = arg
         print('Writing to file',outfile)
      elif opt in ("-v","--vaccination"):
         vacoutfile = arg
         print('Writing vaccination run to file',vacoutfile)

   for i in range(1,int(runs),1):
       print('run no',i)
       dataSeries = Extended_SEIRS_model.run_model(scenario,runs,plot,outfile)
       dataSeries = np.array(dataSeries)
       timeSeries   = np.array(dataSeries[0])
       Sseries      = np.array(dataSeries[1])
       Eseries      = np.array(dataSeries[2])
       I_preseries  = np.array(dataSeries[3])
       I_symseries  = np.array(dataSeries[4])
       I_asymseries = np.array(dataSeries[5])
       Rseries      = np.array(dataSeries[6])
       Hseries      = np.array(dataSeries[7])
       Fseries      = np.array(dataSeries[8])
       vactimeSeries   = np.array(dataSeries[9])
       vacSseries      = np.array(dataSeries[10])
       vacEseries      = np.array(dataSeries[11])
       vacI_preseries  = np.array(dataSeries[12])
       vacI_symseries  = np.array(dataSeries[13])
       vacI_asymseries = np.array(dataSeries[14])
       vacRseries      = np.array(dataSeries[15])
       vacHseries      = np.array(dataSeries[16])
       vacFseries      = np.array(dataSeries[17])

       #Exporting run data into a csv defined by the user
       if(not outfile):
           df = pd.DataFrame({"time" : timeSeries, "susceptibles" : Sseries, "exposed" : Eseries,"i_pre" : I_preseries, "i_sym" : I_symseries,"i_asym" : I_asymseries, "recovered" : Rseries,"hospitalized" : Hseries, "fatalities" : Fseries})
           df.to_csv(outfile+str(i)+'.csv', index=False)
       if(not vacoutfile):
           vacdf = pd.DataFrame({"time" : vactimeSeries, "susceptibles" : vacSseries, "exposed" : vacEseries,"i_pre" : vacI_preseries, "i_sym" : vacI_symseries,"i_asym" : vacI_asymseries, "recovered" : vacRseries,"hospitalized" : vacHseries, "fatalities" : vacFseries})
           vacdf.to_csv(vacoutfile+str(i)+'.csv', index=False)

if __name__ == "__main__":
   main(sys.argv[1:])
