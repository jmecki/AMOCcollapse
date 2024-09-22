import sys

import CMIPtransports as ct

# Get information:
inputs = sys.argv

cmip    = '6'
model   = inputs[1]
EXP     = inputs[2]
ENS     = inputs[3]
gtype   = inputs[4]
ctype   = 'None'
region  = 'Atlantic'
outfile = inputs[5]
runfile = inputs[6]

ct.meridional(cmip,model,EXP,ENS,gtype,ctype,region,outfile,runfile)