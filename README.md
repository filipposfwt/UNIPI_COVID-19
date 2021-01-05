# UNIPI_COVID-19
A two-phase stochastic dynamic network compartmental model (a pre-vaccination SEIR and a post-vaccination SVEIR) is developed. The implementation was based on the code available at https://github.com/ryansmcgee/seirsplus .The post-vaccination phase model uses a modified version of models.py code. More specific, some state transition propensities are modified in order to express the vaccination rate and efficacy.

In order to run the code, after cloning the repository you should install all the required depedencies with pip install -r requirements.txt.

After that code is run by passing some arguments to the modelexec.py script.

Usage: python3 modelexec.py -s [scenario] -r [number of runs] -o [output file name] -v [vaccination output file name] -p

-s : semi/lockdown/freedom 
-r : int
-o : path/string
-v : path/string
-p : True/False (enabling plotting of every run)


