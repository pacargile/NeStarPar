import NeStarPar
import numpy as np

STARPAR = NeStarPar.StarPar()

print '-----------------------------------'
print ' Doing: Test                   '
print ' MOCK:                         '
print ' EEP = 353                     '
print ' Age = 4.5 Gyr                 '
print ' [Fe/H]in = 0.0                '
print ' Parallax = 100 mas (10pc -> DM=0)'
print ' Av = 0.1                      '
print ' PREDICTIONS:                  '
print ' Mass = 0.9997                 '
print ' log(L) = 0.04059              '
print ' Teff = 5845.924               '
print ' Rad = 1.0216                  '
print ' log(g) = 4.41944              '
print ' [Fe/H] =  -0.0199055          '
print ' APPLYING STANDARD ERRORS FOR  '
print ' MEASUREMENTS                  '
print '----------------------------------'

bfpar = {}
# bfpar['Teff'] = 5846.0
# bfpar['log(g)'] = 4.42
# bfpar['[Fe/H]'] = -0.02
bfpar['parallax'] = 100.0
# PREDICTED PHOTOMETRY + A_X (Av=0.1)
bfpar['B_T'] = 5.380 + 0.13682287
bfpar['V_T'] = 4.749 + 0.10439093
bfpar['H_P'] = 4.787 + 0.103268
bfpar['J'] = 3.593 + 0.02874294
bfpar['H'] = 3.267 + 0.01815954
bfpar['Ks'] = 3.235 + 0.0117162
bfpar['W1'] = 3.222 + 0.00592681
bfpar['W2'] = 3.232 + 0.00352907
bfpar['W3'] = 3.207 + 0.00096177
bfpar['W4'] = 3.204 + 0.00028261

epar = {}
# epar['Teff'] = 50.0
# epar['log(g)'] = 0.1
# epar['[Fe/H]'] = 0.05
epar['parallax'] = 1.0
epar['B_T'] = 0.1
epar['V_T'] = 0.1
epar['H_P'] = 0.1
epar['J'] = 0.1
epar['H'] = 0.1
epar['Ks'] = 0.1
epar['W1'] = 0.1
epar['W2'] = 0.1
epar['W3'] = 0.1
epar['W4'] = 0.1


# define a tight prior on distance
priordict = {'Para':[50.0,150.0],'Av':[0.0,3.0]}

# outfile name
outfile = 'MOCK.dat'

# Now, set up and run the sampler:

results,p,cov = STARPAR(bfpar,epar,outfile=outfile,priordict=priordict,sampler='nestle')

print results.summary()
for ii,pp in enumerate(p):
	print pp, np.sqrt(cov[ii,ii])

