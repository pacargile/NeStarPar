import nestle
import emcee
import numpy as np
import sys,time,datetime, math,re
from datetime import datetime
from scipy.interpolate import LinearNDInterpolator,NearestNDInterpolator
from scipy.spatial import cKDTree
from astropy.table import Table
import numpy as np
import h5py
from matplotlib import cm
import warnings

class StarPar(object):
	"""
	Class for StarParMCMC
	"""
	def __init__(self,model=None):

		if model == None:
			model = 'MIST'

		if model == 'MIST':
			# read in MIST models
			modtab_i = Table(np.array(h5py.File('/Users/pcargile/Astro/SteEvoMod/MIST_full.h5','r')['data']))

			# parse MIST models to only include up to end of helium burning (TACHeB), only to ages < 16 Gyr, and 
			# stellar masses < 100 Msol
			modtab = modtab_i[(modtab_i['EEP'] <= 707) & (modtab_i['log_age'] <= np.log10(18.0*10.0**9.0)) & (modtab_i['initial_mass'] < 100.0)]

			self.modphotbands = (['U','B','V','R','I','J','H','Ks','Kp','K_D51','SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z',
				'CFHT_u','CFHT_g','CFHT_r','CFHT_i_new','CFHT_i_old','CFHT_z','W1','W2','W3','W4'])

			# rename stuff for clarity
			modtab['log_L'].name = 'log(L)'
			modtab['log_Teff'].name = 'log(Teff)'
			modtab['initial_mass'].name = 'Mass'
			modtab['initial_[Fe/H]'].name = '[Fe/H]in'
			modtab['log_g'].name = 'log(g)'

			modtab['Teff'] = 10.0**(modtab['log(Teff)'])

			Ysurf = 0.249 + ((0.2703-0.249)/0.0142)*modtab['Z_surf'] # from Asplund et al. 2009
			Xsurf = 1 - Ysurf - modtab['Z_surf']
			modtab['[Fe/H]'] = np.log10((modtab['Z_surf']/Xsurf)) - np.log10(0.0199)

		else:
			print 'ONLY WORKS ON MIST'
			exit()

		modtab['Rad'] = 10.0**((modtab['log(L)'] - 4.0*(modtab['log(Teff)']-np.log10(5770.0)))/2.0)
		self.modtab = modtab

		# create parnames
		self.parnames = modtab.keys()
		# remove interpolated parameters
		self.parnames.remove('EEP')
		self.parnames.remove('log_age')
		self.parnames.remove('[Fe/H]in')

		# define age/mass bounds for use later as uninformative priors
		self.max_e = max(modtab['EEP'])
		self.min_e = min(modtab['EEP'])

		self.max_a = max(modtab['log_age'])
		self.min_a = min(modtab['log_age'])

		self.max_m = max(modtab['[Fe/H]in'])
		self.min_m = min(modtab['[Fe/H]in'])

		self.ss = {}
		print 'Growing the KD-Tree...'
		self.age_scale = 0.05
		self.eep_scale = 1.0
		self.feh_scale = 0.25
		self.age_n = modtab['log_age']*(1.0/self.age_scale) 
		self.eep_n = modtab['EEP']*(1.0/self.eep_scale) 
		self.feh_n = modtab['[Fe/H]in']*(1.0/self.feh_scale) 
		self.pts_n = zip(self.age_n,self.eep_n,self.feh_n)

		self.tree = cKDTree(self.pts_n)

		self.dist = np.sqrt( (2.0**2.0) + (2.0**2.0) + (2.0**2.0) )


	def __call__(self, bfpar,epar,priordict=None,outname='TEST.dat',sampler='emcee',p0mean=None,p0std=None,nwalkers=100,nthreads=0,nburnsteps=0,nsteps=100):
		# write input into self
		self.bfpar = bfpar
		self.epar = epar
		self.priordict = priordict
		self.outname = outname

		if sampler == 'emcee':
			self.p0mean = p0mean
			self.p0std = p0std
			self.nwalkers = nwalkers
			self.nthreads = nthreads
			self.nburnsteps = nburnsteps
			self.nsteps = nsteps

		# number of dim, standard is 3 EEP, log(age), [Fe/H]in
		self.ndim = 3

		# see if photometry is being fit
		self.fitphotbool = False
		for kk in self.bfpar.keys():
			if kk in self.modphotbands:
				self.fitphotbool = True
				self.ndim = 5					
				break

		# see if distance is being fit, only set if photometry is given
		if self.fitphotbool:
			self.fitdistorpara = False
			for kk in self.bfpar.keys():
				if kk in ['Distance','distance','dist','Dist']:
					self.fitdistorpara = "Dist"
					bfpar["Dist"] = bfpar.pop(kk)
					epar["Dist"] = epar.pop(kk)
					self.DM = self.DM_distance
					break
				elif kk in ['Parallax','parallax','par','Par']:
					self.fitdistorpara = "Para"
					bfpar["Para"] = bfpar.pop(kk)
					epar["Para"] = epar.pop(kk)
					self.DM = self.DM_parallax
					break
				else:
					pass


		# start timer
		startbin = datetime.now()

		if sampler == 'emcee':
			self.run_emcee()
		elif sampler == 'nestle':
			return self.run_nestle()
		else:
			print 'DID NOT UNDERSTAND WHICH SAMPLER TO USE!'
			return None

	def run_emcee(self):

			# if photometry is being fit, add distance and reddening to p0
			if self.fitphotbool:
				# if prior set for dist and/or A_v, then sample ball from prior * 1/10
				if 'Dist' in self.priordict.keys():
					self.p0mean = np.append(self.p0mean,(self.priordict['Dist'][1]+self.priordict['Dist'][0])/2.0)
					self.p0std = np.append(self.p0std,(self.priordict['Dist'][1]-self.priordict['Dist'][0])/10.0)
				elif 'Dist' in self.bfpar.keys():
					self.p0mean = np.append(self.p0mean,self.bfpar['Dist'])
					self.p0std = np.append(self.p0std,self.epar['Dist'])
				else:
					# just set it to 1kpc with a stddev of 250 pc and cross your fingers
					self.p0mean = np.append(self.p0mean,1000.0)
					self.p0std = np.append(self.p0std,250.0)
				if 'A_v' in self.priordict.keys():
					self.p0mean = np.append(self.p0mean,(self.priordict['A_v'][1]+self.priordict['A_v'][0])/2.0)
					self.p0std = np.append(self.p0std,(self.priordict['A_v'][1]-self.priordict['A_v'][0])/10.0)
				else:
					self.p0mean = np.append(self.p0mean,1.0)
					self.p0std = np.append(self.p0std,0.1)


			p0 = [[x*np.random.randn()+y for x,y in zip(self.p0std,self.p0mean)] for _ in range(self.nwalkers)]


			# The sampler object:
			print 'Build sampler'
			sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.lnp_call, threads=self.nthreads)

			ballscale = 1.0
			print "Testing inital walker ball..."
			while True:
				p0_t = emcee.utils.sample_ball(self.p0mean,self.p0std*ballscale,self.nwalkers)
				goodint = True
				minlp = np.array([sampler.get_lnprob(pp)[0] for pp in p0_t]).ravel().min()
				if minlp < -1.0*10**100:
					goodint = False
					ballscale = ballscale / 2.0
				else:
					pass
				if goodint == True:
					p0 = p0_t
					break
				else:
					print "P0 guess was bad -- min(lnprob) ~ -inf, restarting with smaller ball"

			pos0_i = p0#self.p0mean
			prob0_i = None
			rstate0_i = None
			blobs0_i = None

			# Sample, outputting to a file
			with open(outname, "w") as f:
				f.write('WN EEP Age [Fe/H]in')
				if self.fitphotbool:
					f.write(' Dist A_v')
				for pp in self.parnames:
					f.write(' {0}'.format(pp))
				f.write(' lnprob\n')

			self.walkernum = np.squeeze(np.arange(1,self.nwalkers+1,1).reshape(self.nwalkers,1))

			print 'Start Emcee'
			startmct = datetime.now()
			ii = 1
			for pos, prob, rstate, blobs in sampler.sample(
				pos0_i, lnprob0=prob0_i, rstate0=rstate0_i, blobs0=blobs0_i, iterations=self.nsteps,storechain=False):
				# build output array
				pos_matrix=pos.reshape(self.nwalkers,self.ndim)
				prob_array=np.squeeze(prob.reshape(self.nwalkers,1))
				blobs = [[blobs_i[kk] for kk in self.parnames] for blobs_i in blobs]
				blobs_array = np.array(blobs).reshape(self.nwalkers,len(self.parnames))
				steparray = np.column_stack([self.walkernum,pos_matrix,blobs_array,prob_array])

				# Write the current position to a file, one line per walker
				f = open(outname, "a")
				f.write("\n".join(["\t".join([str(q) for q in p]) for p in steparray]))
				f.write("\n")
				f.close()

				if ii % 50 == 0.0:
					print('Finished iteration: {0} -- Time: {1} -- mean(AF): {2}'.format(
						ii,datetime.now()-startmct,np.mean(sampler.acceptance_fraction)))
				ii = ii + 1
			print 'Total Time -> ', datetime.now()-startbin

	def lnp_call(self,par):
		# define a default prediction array for radius, [Fe/H], plus the other params
		pred_v = [np.nan_to_num(-np.inf) for _ in range(len(self.parnames)+2)]

		# run modcall to get predicted parameters
		moddict = self.modcall(par)
		if moddict == "ValueError":
			return [np.nan_to_num(-np.inf), pred_v]

		# check priors
		lnprior = self.logprior(moddict)
		if not np.isfinite(lnprior):
			return [np.nan_to_num(-np.inf), pred_v]

		lnlike = self.loglhood(moddict)      
		if not np.isfinite(lnlike):
			return [np.nan_to_num(-np.inf),pred_v]

		results = [lnprior + lnlike, moddict]
		return results

	def logprior(self,moddict):
		# # make sure mass and age are on the model grid
		# mass,age,feh = t

		if self.priordict != None:
			for kk in self.priordict.keys():
				minval = self.priordict[kk][0]
				maxval = self.priordict[kk][1]
				if (moddict[kk] < minval) or (moddict[kk] > maxval):
					# print kk,minval,maxval,moddict[kk]
					return -np.inf
				else:
					pass

		return 0.0

	def run_nestle(self):
		print 'Start Nestle'
		startmct = datetime.now()
		result = nestle.sample(self.lnp_call_nestle,self.prior_trans,self.ndim,method='multi',npoints=200,callback=self.nestle_callback)
		p,cov = nestle.mean_and_cov(result.samples,result.weights)
		return result,p,cov

	def nestle_callback(self,iterinfo):
		# print iteration number and evidence at specific iterations
		if iterinfo['it'] % 1000 == 0:
			if iterinfo['logz'] < -10E+6:
				print 'Iter: {0} < -10M'.format(iterinfo['it'])
			else:
				print 'Iter: {0} = {1}'.format(iterinfo['it'],iterinfo['logz'])


	def prior_trans(self,par):
		eepran = (self.max_e-self.min_e)*0.999
		ageran = (self.max_a-self.min_a)*0.999
		fehran = (self.max_m-self.min_m)*0.999
		if self.fitphotbool:
			age,eep,feh,dist,A_v = par
			distran = 10000.0
			Avran = 5.0
			return np.array([
				ageran*age+self.min_a,
				eepran*eep+self.min_e,
				fehran*feh+self.min_m,
				distran*dist,
				Avran*A_v
				])

		else:
			age,eep,feh = par
			return np.array([
				ageran*age+self.min_a,
				eepran*eep+self.min_e,
				fehran*feh+self.min_m
				])



	def lnp_call_nestle(self,par):
		# run modcall to get predicted parameters
		moddict = self.modcall(par)
		if moddict == "ValueError":
			return np.nan_to_num(-np.inf)
		lnlike = self.loglhood(moddict)      
		return lnlike


	def modcall(self,par,tprof=False):
		if tprof == True:
			starttime = datetime.now()

		try:
			if self.fitphotbool:
				age,eep,feh,dist,A_v = par
				if A_v < 0:
					return 'ValueError'
				if dist <= 0:
					return 'ValueError'
			else:
				age,eep,feh = par
		except AttributeError:
			age,eep,feh = par

		# change age to log(age)
		if age > 0:
			lage = np.log10(age*10.0**9.0)
		else:
			# print 'age < 0'
			return 'ValueError'

		# make sure parameters are within the grid, no extrapolations
		if ((eep > self.max_e) or 
			(eep < self.min_e) or 
			(lage > self.max_a)  or 
			(lage < self.min_a) or 
			(feh > self.max_m) or
			(feh < self.min_m)
			):
			# print 'HIT MODEL BOUNDS'
			return 'ValueError'
	
		# build dictionary to handle everything
		datadict = {}

		# stick in input parameters
		datadict['EEP'] = eep
		datadict['Age'] = 10.0**(lage-9.0)
		datadict['[Fe/H]in'] = feh

		try:
			if self.fitphotbool:
				datadict['A_v'] = A_v
			if self.fitdistorpara == "Dist":
				datadict['Dist'] = dist
			if self.fitdistorpara == "Para":
				datadict['Para'] = dist

		except:
			pass

		# sample KD-Tree and build interpolator
		new_a = lage*(1.0/0.05) 
		new_e = eep*(1.0/1.0) 
		new_m = feh*(1.0/0.25) 

		# run tree query, check to make sure you get enough points in each dimension
		ind = self.tree.query_ball_point([new_a,new_e,new_m],self.dist,p=5)
		modtab_i = self.modtab[ind]

		# check each dimension to make sure it has at least two values to interpolate over
		for testkk in ['log_age','EEP','[Fe/H]in']:
			if len(np.unique(modtab_i[testkk])) < 2:
				return 'ValueError'

		# check to make sure iteration lage,eep,feh is within grid
		if (lage < min(modtab_i['log_age'])) or (lage > max(modtab_i['log_age'])):
			return 'ValueError'
		if (eep < min(modtab_i['EEP'])) or (eep > max(modtab_i['EEP'])):
			return 'ValueError'
		if (feh < min(modtab_i['[Fe/H]in'])) or (feh > max(modtab_i['[Fe/H]in'])):
			return 'ValueError'

		values = np.stack([modtab_i[par] for par in self.parnames],axis=1)
		try:
			dataint = LinearNDInterpolator(
					(modtab_i['log_age'],modtab_i['EEP'],modtab_i['[Fe/H]in']),
					values,
					fill_value=np.nan_to_num(-np.inf),
					rescale=False
					)(lage,eep,feh)
		except:
			print min(modtab_i['log_age']),max(modtab_i['log_age']),min(modtab_i['EEP']),max(modtab_i['EEP']),min(modtab_i['[Fe/H]in']),max(modtab_i['[Fe/H]in'])
			print (lage,eep,feh)
			return 'ValueError'
							
		for ii,par in enumerate(self.parnames):
			if dataint[ii] != np.nan_to_num(-np.inf):
				datadict[par] = dataint[ii]
			else:
				return 'ValueError'

		if (tprof == True):
			print 'total time: ',datetime.now()-starttime

		# build a dictionary that has an interpolation object for each parameter
		return datadict

	def loglhood(self,moddict):
		deltadict = {}
		if self.fitdistorpara == "Dist":
			dist = moddict['Dist']
		if self.fitdistorpara == "Para":
			dist = moddict['Para']

		for kk in self.bfpar.keys():
			if kk in self.modphotbands:
				obsphot = moddict[kk]+self.DM(dist)+self.red(band=kk,A_v=moddict['A_v'])
				deltadict[kk] = (obsphot-self.bfpar[kk])/self.epar[kk]
			else:
				deltadict[kk] = (moddict[kk]-self.bfpar[kk])/self.epar[kk]
		chisq = np.sum([deltadict[kk]**2.0 for kk in self.bfpar.keys()])
		return -0.5 * chisq


	def DM_parallax(self,parallax):
		#parallax in mas
		return 5.0*np.log10(1.0/(parallax/1000.0))-5.0

	def DM_distance(self,distance):
		#distance in parsecs
		return 5.0*np.log10(distance)-5.0

	def red(self,Teff=None,band='V',A_v=0.0):
		# Right now don't do anything with Teff, eventually have it correctly calculate Av/E(B-V), 
		# currently only returns the reddening value for a solar template.
		# All reddening laws taken from ADPS which uses the Fitz99 extinction curves (ssuming Av=3.1) 
		# unless otherwise noted.

		# Gaia_G taken from Jordi+2010 w/ V-I = 0.702 (Sun)
		# assume the CFHT ugriz is the same as SDSS ugriz (likely untrue)
		# K_D51 manually calculated using Cadelli law
		# Kp taken from Tim Morton's isochrone.py code
		# WISE W1, W2, W3 taken from Davenport+2014 with Ar/Av = 0.83
		# WISE W4 taken from Bilir+2011 (since not in Davenport+2014)

		# t_s = t[np.in1d(t['Teff'],[5000.0,6000.0])*np.in1d(t['logg'],[4.0,4.5])*np.in1d(t['[Fe/H]'],[0.0,0.25])]


		reddeninglaw = (
			{
			'U':1.62,
			'B':1.31,
			'V':1.00,
			'R':0.77,
			'I':0.56,
			'B_T':1.38,
			'V_T':1.03,
			'H_P':0.95,
			'Gaia_G':0.805,
			'SDSS_u':1.61,
			'SDSS_g':1.19,
			'SDSS_r':0.83,
			'SDSS_i':0.61,
			'SDSS_z':0.45,
			'CFHT_u':1.61,
			'CFHT_g':1.19,
			'CFHT_r':0.83,
			'CFHT_i_new':0.61,
			'CFHT_i_old':0.61,
			'CFHT_z':0.45,
			'J':0.27,
			'H':0.17,
			'Ks':0.12,
			'K_D51':1.0942,
			'Kp':0.859,
			'W1':0.09*0.83,
			'W2':0.05*0.83,
			'W3':0.13*0.83,
			'W4':0.056,
			}
			)
		# pull ratio for band
		AxAv = reddeninglaw[band]
		# calculate specific Ax
		Ax = AxAv*A_v

		return Ax

if __name__ == '__main__':

	# initialize StarPar class
	STARPAR = StarPar()

	print '-------------------------------'
	print ' Doing: Test                   '
	print ' KELT-11                       '
	print '-------------------------------'
	# test case
	bfpar = {}
	bfpar['Teff'] = 5390.0
	bfpar['log(g)'] = 3.79
	bfpar['[Fe/H]'] = 0.19
	bfpar['parallax'] = 9.76
	bfpar['J'] = 6.616
	bfpar['H'] = 6.251
	bfpar['Ks'] = 6.122
	bfpar['W1'] = 6.152
	bfpar['W2'] = 6.068
	bfpar['W3'] = 6.157
	bfpar['W4'] = 6.088


	epar = {}
	epar['Teff'] = 50.0
	epar['log(g)'] = 0.1
	epar['[Fe/H]'] = 0.08
	epar['parallax'] = 0.85
	epar['J'] = 0.024
	epar['H'] = 0.042
	epar['Ks'] = 0.018
	epar['W1'] = 0.1
	epar['W2'] = 0.036
	epar['W3'] = 0.015
	epar['W4'] = 0.048

	# define any priors with dictionary
	priordict = {}
	priordict['Age'] = [1.0,15.0]
	priordict['Mass'] = [0.25,5.0]
	priordict['Rad'] = [0.1,5.0]

	# name output file
	outname = 'KELT11_test.dat'

	# Now, set up and run the sampler:
	results,p,cov = STARPAR(bfpar,epar,outname=outname,sampler='nestle')

	print results.summary()
	for ii,pp in enumerate(p):
		print pp, np.sqrt(cov[ii,ii])

	"""
	print '-------------------------------'
	print ' Doing: Test                   '
	print ' alpha Cen A w/ astroseis pars '
	print ' Lit. Results say:             '
	print ' Mass = 1.105+-0.007 Msol        '
	print ' Age = 6.52+-0.3 Gyr            '
	print ' Radius = 1.224+-0.003 Rsol   '
	print '-------------------------------'
	# test case
	bfpar = {}
	bfpar['Teff'] = 5810.0
	bfpar['log(g)'] = 4.307
	bfpar['[Fe/H]'] = 0.22
	bfpar['parallax'] = 747.23
	bfpar['V'] = -0.01
	bfpar['B'] = 0.70
	bfpar['U'] = 0.94

	epar = {}
	epar['Teff'] = 50.0
	epar['log(g)'] = 0.005
	epar['[Fe/H]'] = 0.05
	epar['parallax'] = 1.17
	epar['V'] = 0.01
	epar['B'] = 0.01
	epar['U'] = 0.01

	# define any priors with dictionary
	priordict = {}
	priordict['Age'] = [1.0,15.0]
	priordict['Mass'] = [0.25,3.0]
	priordict['Rad'] = [0.1,3.0]

	# name output file
	outname = 'alphaCenA_test.dat'

	# Now, set up and run the sampler:

	results,p,cov = STARPAR(bfpar,epar,outname=outname,sampler='nestle')

	print results.summary()
	for ii,pp in enumerate(p):
		print pp, np.sqrt(cov[ii,ii])

	print '-------------------------------'
	print ' Doing: Test                   '
	print ' Gl 15 A                       '
	print ' Lit. Results say:             '
	print ' Mass = 0.375+-0.06 Msol       '
	print ' Age ~ 5 Gyr                   '
	print ' Radius = 0.3863 +- 0.0021 Rsol'
	print '-------------------------------'
	# test case
	bfpar = {}
	bfpar['Teff'] = 3567.0
	bfpar['log(g)'] = 4.90
	bfpar['[Fe/H]'] = -0.32
	bfpar['parallax'] = 278.76
	bfpar['B'] = 9.63
	bfpar['V'] = 8.08
	bfpar['J'] = 4.82
	bfpar['H'] = 4.25
	bfpar['Ks'] = 4.03
	bfpar['SDSS_i'] = 7.282

	epar = {}
	epar['Teff'] = 11.0
	epar['log(g)'] = 0.17
	epar['[Fe/H]'] = 0.17
	epar['parallax'] = 0.77
	epar['B'] = 0.05
	epar['V'] = 0.01
	epar['J'] = 0.1
	epar['H'] = 0.1
	epar['Ks'] = 0.1
	epar['SDSS_i'] = 0.01

	# define any priors with dictionary
	priordict = {}
	priordict['Age'] = [0.0,15.0]
	priordict['Mass'] = [0.1,3.0]
	priordict['Rad'] = [0.1,10.0]

	# name output file
	outname = 'Gl15A_test.dat'

	# Now, set up and run the sampler:

	results,p,cov = STARPAR(bfpar,epar,outname=outname,sampler='nestle')

	print results.summary()
	for ii,pp in enumerate(p):
		print pp, np.sqrt(cov[ii,ii])


	print '-------------------------------'
	print ' Doing: Test                   '
	print ' eps Eri                      '
	print ' Lit. Results say:             '
	print ' Mass = 0.82+-0.05 Msol        '
	print ' Age ~ 1 Gyr            '
	print ' Radius = 0.74+-0.01 Rsol   '
	print '-------------------------------'
	# test case
	bfpar = {}
	bfpar['Teff'] = 5039.0
	bfpar['log(g)'] = 4.57
	bfpar['[Fe/H]'] = -0.13
	bfpar['parallax'] = 311.73
	bfpar['V'] = 3.73
	bfpar['Ks'] = 1.78

	epar = {}
	epar['Teff'] = 126.0
	epar['log(g)'] = 0.1
	epar['[Fe/H]'] = 0.04
	epar['parallax'] = 0.11
	epar['V'] = 0.01
	epar['Ks'] = 0.29

	# define any priors with dictionary
	priordict = {}
	priordict['Age'] = [1.0,15.0]
	priordict['Mass'] = [0.25,3.0]
	priordict['Rad'] = [0.1,3.0]

	# name output file
	outname = 'epsEri_test.dat'

	# Now, set up and run the sampler:

	results,p,cov = STARPAR(bfpar,epar,outname=outname,sampler='nestle')

	print results.summary()
	for ii,pp in enumerate(p):
		print pp, np.sqrt(cov[ii,ii])

	print '-------------------------------'
	print ' Doing: Test                   '
	print ' kappa CrB                     '
	print ' Lit. Results say:             '
	print ' Mass = 1.47+-0.04 Msol        '
	print ' Age = 3.42+0.32-0.25 Myr      '
	print ' Radius = 5.06+-0.04 Rsol      '
	print '-------------------------------'
	# test case
	bfpar = {}
	bfpar['Teff'] = 4788.0
	bfpar['log(g)'] = 3.47
	bfpar['[Fe/H]'] = 0.15
	bfpar['parallax'] = 32.79
	bfpar['SDSS_u'] = 7.58
	bfpar['SDSS_g'] = 5.28
	bfpar['SDSS_r'] = 4.49
	bfpar['SDSS_i'] = 4.26
	bfpar['SDSS_z'] = 4.14


	epar = {}
	epar['Teff'] = 70.0
	epar['log(g)'] = 0.09
	epar['[Fe/H]'] = 0.05
	epar['parallax'] = 0.21
	epar['SDSS_u'] = 0.06 
	epar['SDSS_g'] = 0.04
	epar['SDSS_r'] = 0.03
	epar['SDSS_i'] = 0.02
	epar['SDSS_z'] = 0.02

	# define any priors with dictionary
	priordict = {}
	priordict['Age'] = [0.0,15.0]
	priordict['Mass'] = [0.25,3.0]
	priordict['Rad'] = [0.1,10.0]

	# name output file
	outname = 'kapCrB_test.dat'

	# Now, set up and run the sampler:

	results,p,cov = STARPAR(bfpar,epar,outname=outname,sampler='nestle')

	print results.summary()
	for ii,pp in enumerate(p):
		print pp, np.sqrt(cov[ii,ii])


	print '-------------------------------'
	print ' Doing: Test                   '
	print ' HR 8799                       '
	print ' Lit. Results say:             '
	print ' Mass = 1.47+-0.04 Msol        '
	print ' Age = 30-90 Myr               '
	print ' Radius = 1.44+-0.06 Rsol      '
	print '-------------------------------'
	# test case
	bfpar = {}
	bfpar['Teff'] = 7430.0
	bfpar['log(g)'] = 4.35
	bfpar['[Fe/H]'] = -0.47
	bfpar['parallax'] = 25.38
	bfpar['U'] = 6.191
	bfpar['B'] = 6.235
	bfpar['V'] = 5.980
	bfpar['J'] = 5.383
	bfpar['H'] = 5.280
	bfpar['Ks'] = 5.240


	epar = {}
	epar['Teff'] = 75.0
	epar['log(g)'] = 0.05
	epar['[Fe/H]'] = 0.1
	epar['parallax'] = 0.70
	epar['U'] = 0.016 
	epar['B'] = 0.016
	epar['V'] = 0.016
	epar['J'] = 0.027
	epar['H'] = 0.018
	epar['Ks'] = 0.018

	# define any priors with dictionary
	priordict = {}
	priordict['Age'] = [0.0,15.0]
	priordict['Mass'] = [0.25,3.0]
	priordict['Rad'] = [0.1,10.0]

	# name output file
	outname = 'HD8799_test.dat'

	# Now, set up and run the sampler:

	results,p,cov = STARPAR(bfpar,epar,outname=outname,sampler='nestle')

	print results.summary()
	for ii,pp in enumerate(p):
		print pp, np.sqrt(cov[ii,ii])


	print '-------------------------------'
	print ' Doing: Test                   '
	print ' HR 7924                       '
	print ' Lit. Results say:             '
	print ' Mass = 0.832+-0.03 Msol       '
	print ' Age =  11.5 +- 0.5 Myr               '
	print ' Radius = 0.7821+-0.0258 Rsol  '
	print '-------------------------------'
	# test case
	bfpar = {}
	bfpar['Teff'] = 5075.0
	bfpar['log(g)'] = 4.59
	bfpar['[Fe/H]'] = -0.15
	bfpar['parallax'] = 2.31
	bfpar['B'] = 8.011
	bfpar['V'] = 7.185
	bfpar['J'] = 5.618
	bfpar['H'] = 5.231
	bfpar['Ks'] = 5.159


	epar = {}
	epar['Teff'] = 83.0
	epar['log(g)'] = 0.02
	epar['[Fe/H]'] = 0.03
	epar['parallax'] = 0.32
	epar['B'] = 0.1
	epar['V'] = 0.1
	epar['J'] = 0.026
	epar['H'] = 0.033
	epar['Ks'] = 0.020

	# define any priors with dictionary
	priordict = {}
	priordict['Age'] = [0.0,15.0]
	priordict['Mass'] = [0.25,3.0]
	priordict['Rad'] = [0.1,10.0]

	# name output file
	outname = 'HR7924_test.dat'

	# Now, set up and run the sampler:

	results,p,cov = STARPAR(bfpar,epar,outname=outname,sampler='nestle')

	print results.summary()
	for ii,pp in enumerate(p):
		print pp, np.sqrt(cov[ii,ii])


	print '-------------------------------'
	print ' Doing: Test                   '
	print ' Gl 15 A                       '
	print ' Lit. Results say:             '
	print ' Mass = 0.375+-0.06 Msol       '
	print ' Age ~ 5 Gyr                   '
	print ' Radius = 0.3863 +- 0.0021 Rsol'
	print '-------------------------------'
	# test case
	bfpar = {}
	bfpar['Teff'] = 3567.0
	bfpar['log(g)'] = 4.90
	bfpar['[Fe/H]'] = -0.32
	bfpar['parallax'] = 278.76
	bfpar['B'] = 9.63
	bfpar['V'] = 8.08
	bfpar['J'] = 4.82
	bfpar['H'] = 4.25
	bfpar['Ks'] = 4.03
	bfpar['SDSS_i'] = 7.282

	epar = {}
	epar['Teff'] = 11.0
	epar['log(g)'] = 0.17
	epar['[Fe/H]'] = 0.17
	epar['parallax'] = 0.77
	epar['B'] = 0.05
	epar['V'] = 0.01
	epar['J'] = 0.1
	epar['H'] = 0.1
	epar['Ks'] = 0.1
	epar['SDSS_i'] = 0.01

	# define any priors with dictionary
	priordict = {}
	priordict['Age'] = [0.0,15.0]
	priordict['Mass'] = [0.1,3.0]
	priordict['Rad'] = [0.1,10.0]

	# name output file
	outname = 'Gl15A_test.dat'

	# Now, set up and run the sampler:

	results,p,cov = STARPAR(bfpar,epar,outname=outname,sampler='nestle')

	print results.summary()
	for ii,pp in enumerate(p):
		print pp, np.sqrt(cov[ii,ii])

	"""

	# # fullrun BI-> 100, Nsteps -> 400, nwalkers = 200, threads=4
	# nwalkers = 50
	# nthreads = 0 # multithreading not turned on yet, still dealing with pickling error
	# nburnsteps = 0
	# nsteps = 200

	# # Make an initial guess for the positions - uniformly
	# p0mean = np.array([352,5.0,0.0])
	# p0std  = np.array([20.0,1.0,0.1])

	# # Run MCMC
	# STARPAR(bfpar,epar,p0mean,p0std,nwalkers=nwalkers,nthreads=nthreads,
	# 	nburnsteps=nburnsteps,nsteps=nsteps,outname=outname,priordict=priordict)
			