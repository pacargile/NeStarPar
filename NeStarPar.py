import nestle
import emcee
import numpy as np
import sys,time,datetime, math,re
from datetime import datetime
from scipy.interpolate import LinearNDInterpolator,NearestNDInterpolator
from scipy.spatial import cKDTree
from scipy.special import erfinv
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
			MISTFILE = '/Users/Phill/Astro/MIST/MIST_full.h5'
			EAF_i = Table(np.array(h5py.File(MISTFILE,'r')['EAF']))
			BSP_i = Table(np.array(h5py.File(MISTFILE,'r')['BSP']))
			PHOT_i = Table(np.array(h5py.File(MISTFILE,'r')['PHOT']))
			ALLSP_i = Table(np.array(h5py.File(MISTFILE,'r')['All_SP']))

			# parse MIST models to only include up to end of helium burning (TACHeB), only to ages < 16 Gyr, and 
			# stellar masses < 50 Msol (EEP < 707)
			cond = (EAF_i['EEP'] <= 605) & (EAF_i['log_age'] <= np.log10(17.0*10.0**9.0)) & (BSP_i['star_mass'] < 30.0)
			self.EAF = EAF_i[cond]
			self.BSP = BSP_i[cond]
			self.PHOT = PHOT_i[cond]
			self.ALLSP = ALLSP_i[cond]
			Xsurf = self.ALLSP['surface_h1'].copy()
			del(self.ALLSP)


			# define the available photometric filters
			self.modphotbands = self.PHOT.keys()

			# rename stuff for clarity
			self.EAF['initial_[Fe/H]'].name = '[Fe/H]in'
			self.BSP['log_L'].name = 'log(L)'
			self.BSP['log_Teff'].name = 'log(Teff)'
			self.BSP['log_R'].name = 'log(R)'
			self.BSP['star_mass'].name = 'Mass'
			self.BSP['log_g'].name = 'log(g)'
			self.BSP['log_surf_z'].name = 'log(Z_surf)'

			self.BSP['Z_surf'] = 10.0**(self.BSP['log(Z_surf)'])
			self.BSP['Teff'] = 10.0**(self.BSP['log(Teff)'])
			self.BSP['Rad'] = 10.0**(self.BSP['log(R)'])

			# Ysurf = 0.249 + ((0.2703-0.249)/0.0142)*self.BSP['Z_surf'] # from Asplund et al. 2009
			# Xsurf = 1 - Ysurf - self.BSP['Z_surf']
			# self.BSP['[Fe/H]'] = np.log10(self.BSP['Z_surf']/Xsurf) - np.log10(0.0199)
			self.BSP['[Fe/H]'] = np.log10(self.BSP['Z_surf']/Xsurf) - np.log10(0.0181)

		else:
			print 'ONLY WORKS ON MIST'
			exit()

		# create parnames
		self.parnames = self.BSP.keys()+self.PHOT.keys()

		# create a stacked values array
		self.valuestack = np.stack(
			[self.BSP[par] for par in self.BSP.keys()]+[self.PHOT[par] for par in self.PHOT.keys()]
			,axis=1)

		# init the reddening class
		RR = Redden()
		self.red = RR.red

		# define age/mass bounds for use later as uninformative priors
		self.minmax = {}
		self.minmax['EEP'] = [self.EAF['EEP'].min(),self.EAF['EEP'].max()]
		self.minmax['AGE'] = [self.EAF['log_age'].min(),self.EAF['log_age'].max()]
		self.minmax['FEH'] = [self.EAF['[Fe/H]in'].min(),self.EAF['[Fe/H]in'].max()]

		self.ss = {}
		print 'Growing the KD-Tree...'
		self.eep_scale = 1.0
		self.age_scale = 0.05
		self.feh_scale = 0.25
		self.eep_n = self.EAF['EEP']*(1.0/self.eep_scale) 
		self.age_n = self.EAF['log_age']*(1.0/self.age_scale) 
		self.feh_n = self.EAF['[Fe/H]in']*(1.0/self.feh_scale) 
		self.pts_n = zip(self.eep_n,self.age_n,self.feh_n)

		self.tree = cKDTree(self.pts_n)

		self.dist = np.sqrt( (2.0**2.0) + (2.0**2.0) + (2.0**2.0) )


	def __call__(self, bfpar,epar,priordict={},outfile='TEST.dat',sampler='emcee',p0mean=None,p0std=None,nwalkers=100,nthreads=0,nburnsteps=0,nsteps=100):
		# write input into self
		self.bfpar = bfpar
		self.epar = epar
		self.priordict = priordict
		self.outfile = outfile

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
		self.fitdistorpara = False
		for kk in self.bfpar.keys():
			if kk in self.modphotbands:
				self.fitphotbool = True
				self.ndim = 5					
				break

		# see if distance is being fit, only set if photometry is given
		if self.fitphotbool:
			for kk in self.bfpar.keys():
				if kk in ['Distance','distance','dist','Dist']:
					print 'FITTING DISTANCE'
					self.fitdistorpara = "Dist"
					self.bfpar["Dist"] = bfpar.pop(kk)
					self.epar["Dist"] = epar.pop(kk)
					break
				elif kk in ['Parallax','parallax','par','Par','Para','para']:
					print 'FITTING PARALLAX'
					self.fitdistorpara = "Para"
					self.bfpar["Para"] = bfpar.pop(kk)
					self.epar["Para"] = epar.pop(kk)
					break
				else:
					pass

		# remove dist or para from outfilepars
		self.outfilepars = self.bfpar.keys()
		if self.fitdistorpara == "Dist":
			self.outfilepars.remove('Dist')

		# add interpolated values
		if self.fitphotbool:
			self.outfilepars = ['EEP','Age','[Fe/H]in','Dist','Av']+self.outfilepars
		else:
			self.outfilepars = ['EEP','Age','[Fe/H]in']+self.outfilepars

		if 'Mass' not in self.outfilepars:
			self.outfilepars.append('Mass')
		if 'Rad' not in self.outfilepars:
			self.outfilepars.append('Rad')
		if 'log(L)' not in self.outfilepars:
			self.outfilepars.append('log(L)')
		if 'Teff' not in self.outfilepars:
			self.outfilepars.append('Teff')
		if 'log(g)' not in self.outfilepars:
			self.outfilepars.append('log(g)')
		if '[Fe/H]' not in self.outfilepars:
			self.outfilepars.append('[Fe/H]')

		# init output file
		self.outff = open(self.outfile, 'w')
		self.outff.write('ITER ')
		for pp in self.outfilepars:
			self.outff.write('{0} '.format(pp))
		self.outff.write('logz logwt\n')

		# define some default priors based on grid limits,
		# else use the user input priors

		if 'EEP' not in self.priordict.keys():
			self.eepran = (self.minmax['EEP'][1]-self.minmax['EEP'][0])
			self.mineep = self.minmax['EEP'][0]
		else:
			self.eepran = (self.priordict['EEP'][1]-self.priordict['EEP'][0])
			self.mineep = self.priordict['EEP'][0]

		if 'Age' not in self.priordict.keys():
			self.ageran = (10.0**self.minmax['AGE'][1]-10.0**self.minmax['AGE'][0])/(10.0**9.0)
			self.minage = (10.0**self.minmax['AGE'][0])/(10.0**9.0)
		else:
			self.ageran = (self.priordict['Age'][1]-self.priordict['Age'][0])
			self.minage = self.priordict['Age'][0]

		if '[Fe/H]in' not in self.priordict.keys():
			self.fehran = (self.minmax['FEH'][1]-self.minmax['FEH'][0])
			self.minfeh = self.minmax['FEH'][0]
		else:
			self.fehran = (self.priordict['[Fe/H]in'][1]-self.priordict['[Fe/H]in'][0])
			self.minfeh = self.priordict['[Fe/H]in'][0]

		if self.fitphotbool:
			if 'Dist' not in self.priordict.keys():
				self.distran = 10000.0 # 10 kpc standard prior
				self.mindist = 0.0
			else:
				self.distran = (self.priordict['Dist'][1]-self.priordict['Dist'][0])
				self.mindist = self.priordict['Dist'][0]

			if 'Av' not in self.priordict.keys():
				self.Avran = 6.0 # set by Av grid
				self.minAv = 0.0
			else:
				self.Avran = (self.priordict['Av'][1]-self.priordict['Av'][0])
				self.minAv = self.priordict['Av'][0]


		# start timer
		startbin = datetime.now()

		if sampler == 'emcee':
			self.run_emcee()
		elif sampler == 'nestle':
			return self.run_nestle()
		elif sampler == 'init':
			return None
		else:
			print 'DID NOT UNDERSTAND WHICH SAMPLER TO USE!'
			return None

		# close output file
		self.outff.close()

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
				if 'Av' in self.priordict.keys():
					self.p0mean = np.append(self.p0mean,(self.priordict['Av'][1]+self.priordict['Av'][0])/2.0)
					self.p0std = np.append(self.p0std,(self.priordict['Av'][1]-self.priordict['Av'][0])/10.0)
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
					f.write(' Dist Av')
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

	def run_nestle(self,npoints=500):
		# generate initial random sample within Nestle volume
		modind = np.array(range(0,len(self.EAF)),dtype=int)
		selind = modind[np.random.choice(len(modind),npoints,replace=False)]
		
		cond = (
			(self.PHOT['V_T']+self.DM(self.mindist+0.5*self.distran) > self.bfpar['V_T']-1.0) & 
			(self.PHOT['V_T']+self.DM(self.mindist+0.5*self.distran) < self.bfpar['V_T'] + 1.0) &
			(self.PHOT['B_T']+self.DM(self.mindist+0.5*self.distran) > self.bfpar['B_T']-1.0) & 
			(self.PHOT['B_T']+self.DM(self.mindist+0.5*self.distran) < self.bfpar['B_T'] + 1.0)
			)

		addind = modind[cond][np.random.choice(len(modind[cond]),int(npoints*0.25),replace=False)]
		# cond = self.EAF['EEP'] <= 400
		# modind = modind[cond]

		finind = np.hstack([selind,addind])
		finind = np.unique(finind)
		initsample = self.EAF[finind]
		initsample_v = np.empty((len(initsample), self.ndim), dtype=np.float64)
		initsample_u = np.empty((len(initsample), self.ndim), dtype=np.float64)

		for i in range(len(initsample)):
			initsample_v_i = [float(initsample['EEP'][i]),10.0**(float(initsample['log_age'][i])-9.0),float(initsample['[Fe/H]in'][i])]
			if self.fitphotbool:
				# initsample_v_i.append(self.distran*np.random.rand()+self.mindist)
				# initsample_v_i.append(self.Avran*np.random.rand()+self.minAv)
				distmean = 1000.0/self.priordict['Para'][0]
				distsig = distmean-(1000.0/(self.priordict['Para'][0]-self.priordict['Para'][1]))
				initsample_v_i.append(distsig*np.random.randn()+distmean)
				initsample_v_i.append(self.Avran*np.random.rand()+self.minAv)

			initsample_v[i,:] = initsample_v_i
			initsample_u[i,:] = self.prior_inversetrans(initsample_v[i,:])

		print 'Start Nestle w/ {0} number of samples'.format(len(initsample))
		self.startmct = datetime.now()
		result = nestle.sample(
			self.lnp_call_nestle,self.prior_trans,self.ndim,method='multi',
			npoints=len(initsample),callback=self.nestle_callback,user_sample=initsample_u,
			# update_interval=

			)
		p,cov = nestle.mean_and_cov(result.samples,result.weights)
		return result,p,cov

	def nestle_callback(self,iterinfo):
		# write data to outfile
		self.outff.write('{0} '.format(iterinfo['it']))
		# write parameters at iteration if not ValueError
		if self.moddict != 'ValueError':
			for pp in self.outfilepars:
				self.outff.write('{0} '.format(self.moddict[pp]))
		else:
			for VV in iterinfo['active_v']:
				self.outff.write('{0} '.format(VV))
			for _ in range(len(self.outfilepars)-len(iterinfo['active_v'])):
				self.outff.write('-999.99 ')

		# write the evidence
		self.outff.write('{0} '.format(iterinfo['logz']))
		# write the weight
		self.outff.write('{0} '.format(iterinfo['logwt']))
		# write new line
		self.outff.write('\n')

		# print iteration number and evidence at specific iterations
		if iterinfo['it'] % 500 == 0:
			if iterinfo['logz'] < -10E+6:
				print 'Iter: {0} log(z) < -10M log(vol) = {1} Time: {2}'.format(iterinfo['it'],iterinfo['logvol'],datetime.now()-self.startmct)
			else:
				print 'Iter: {0} log(z) = {1} log(vol) = {2} Time: {3}'.format(iterinfo['it'],iterinfo['logz'],iterinfo['logvol'],datetime.now()-self.startmct)
			self.startmct = datetime.now()

	def prior_trans(self,par):
		""" project the parameters onto the prior unit cube """	

		if self.fitphotbool:
			eep,age,feh,dist,Av = par
			para = 1000.0/dist
			return np.array([
				self.eepran*eep+self.mineep,
				self.ageran*age+self.minage,
				self.fehran*feh+self.minfeh,
				self.priordict['Para'][0]+np.sqrt(2.0)*self.priordict['Para'][1]*erfinv(2.0*para-1.0),
				# self.distran*dist+self.mindist,
				self.Avran*Av+self.minAv
				])

		else:
			eep,age,feh = par
			return np.array([
				self.eepran*eep+self.mineep,
				self.ageran*age+self.minage,
				self.fehran*feh+self.minfeh
				])

	def prior_inversetrans(self,par):
		""" de-project values on the prior unit cube to parameter space """
		if self.fitphotbool:
			eep,age,feh,dist,Av = par
			para = 1000.0/dist
			return np.array([
				(eep-self.mineep)/self.eepran,
				(age-self.minage)/self.ageran,
				(feh-self.minfeh)/self.fehran,
				# (dist-self.mindist)/self.distran,
				1.0/(np.sqrt(2.0*np.pi)*self.priordict['Para'][1])*np.exp(-0.5*((para-self.priordict['Para'][0])/self.priordict['Para'][1])**2.0),
				(Av-self.minAv)/self.Avran
				])

		else:
			eep,age,feh = par
			return np.array([
				(eep-self.mineep)/self.eepran,
				(age-self.minage)/self.ageran,
				(feh-self.minfeh)/self.fehran,
				])



	def lnp_call_nestle(self,par):
		# run modcall to get predicted parameters
		self.moddict = self.modcall(par)
		if self.moddict == "ValueError":
			return np.nan_to_num(-np.inf)
		# lnprior = self.logprior(self.moddict)
		lnprior = 0.0
		lnlike = self.loglhood(self.moddict)
		return lnprior+lnlike


	def modcall(self,par,tprof=False,verbose=False):
		if tprof:
			starttime = datetime.now()

		try:
			if self.fitphotbool:
				eep,age,feh,dist,A_v = par
				if A_v < 0:
					if verbose:
						print 'Av < 0'
					return 'ValueError'
				if dist <= 0:
					if verbose:
						print 'dist < 0'
					return 'ValueError'
			else:
				eep,age,feh = par
		except AttributeError:
			eep,age,feh = par

		# change age to log(age)
		if age > 0:
			lage = np.log10(age*10.0**9.0)
		else:
			# print 'age < 0'
			if verbose:
				print 'age < 0'
			return 'ValueError'

		# make sure parameters are within the grid, no extrapolations
		if ((eep > self.minmax['EEP'][1]) or 
			(eep < self.minmax['EEP'][0]) or 
			(lage > self.minmax['AGE'][1])  or 
			(lage < self.minmax['AGE'][0]) or 
			(feh > self.minmax['FEH'][1]) or
			(feh < self.minmax['FEH'][0])
			):
			if verbose:
				print 'HIT MODEL BOUNDS'
			return 'ValueError'
	
		# build dictionary to handle everything
		datadict = {}

		# stick in input parameters
		datadict['EEP'] = eep
		datadict['Age'] = age
		datadict['[Fe/H]in'] = feh

		try:
			if self.fitphotbool:
				datadict['Av'] = A_v
				datadict['Dist'] = dist
				datadict['Para'] = 1000.0/dist
		except:
			pass

		# sample KD-Tree and build interpolator
		new_e = eep*(1.0/self.eep_scale) 
		new_a = lage*(1.0/self.age_scale) 
		new_m = feh*(1.0/self.feh_scale) 

		# run tree query, check to make sure you get enough points in each dimension
		ind = self.tree.query_ball_point([new_e,new_a,new_m],self.dist,p=5)
		EAF_i = self.EAF[ind]

		# check each dimension to make sure it has at least two values to interpolate over
		for testkk in ['EEP','log_age','[Fe/H]in']:
			if len(np.unique(EAF_i[testkk])) < 2:
				if verbose:
					print 'Not enough points in KD-Tree sample'
				return 'ValueError'

		# check to make sure iteration lage,eep,feh is within grid
		if (lage < min(EAF_i['log_age'])) or (lage > max(EAF_i['log_age'])):
			return 'ValueError'
		if (eep < min(EAF_i['EEP'])) or (eep > max(EAF_i['EEP'])):
			return 'ValueError'
		if (feh < min(EAF_i['[Fe/H]in'])) or (feh > max(EAF_i['[Fe/H]in'])):
			return 'ValueError'

		valuestack_i = self.valuestack[ind]

		try:
			dataint = LinearNDInterpolator(
					(EAF_i['EEP'],EAF_i['log_age'],EAF_i['[Fe/H]in']),
					valuestack_i,
					fill_value=np.nan_to_num(-np.inf),
					rescale=True
					)(eep,lage,feh)
		except:
			if verbose:
				print 'Problem with linear inter of KD-Tree sample'
				print min(modtab_i['log_age']),max(modtab_i['log_age']),min(modtab_i['EEP']),max(modtab_i['EEP']),min(modtab_i['[Fe/H]in']),max(modtab_i['[Fe/H]in'])
				print (eep,lage,feh)
			return 'ValueError'
							
		for ii,par in enumerate(self.parnames):
			if dataint[ii] != np.nan_to_num(-np.inf):
				datadict[par] = dataint[ii]
			else:
				if verbose:
					print 'Tried to extrapolate'
				return 'ValueError'

		if (tprof == True):
			print 'total time: ',datetime.now()-starttime

		# build a dictionary that has an interpolation object for each parameter
		return datadict

	def logprior(self,moddict):
		lnprior = 0.0

		for kk in self.priordict.keys():
			if kk not in ['EEP','Age','[Fe/H]in','Dist','Av']:
				# check to see if it is a uniform prior (len=2), or gaussian prior (len=3)
				if len(self.priordict[kk]) == 2:
					if (moddict[kk] < self.priordict[kk][0]) or (moddict[kk] > self.priordict[kk][1]):
						return np.nan_to_num(-1.0*np.inf)
					else:
						pass
				else:
					resid2 = 0.5*((moddict[kk]-self.priordict[kk][0])**2.0)/(self.priordict[kk][1]**2.0)
					lnprior = lnprior + resid2 - np.log(np.sqrt(2.0*np.pi)*self.priordict[kk][1])
		return lnprior

	def loglhood(self,moddict):
		deltadict = {}
		dist = moddict['Dist']

		for kk in self.bfpar.keys():
			if kk in self.modphotbands:
				DM_i = self.DM(dist)
				RED_i = self.red(Teff=moddict['Teff'],logg=moddict['log(g)'],FeH=moddict['[Fe/H]in'],
					band=kk,Av=moddict['Av'])
				obsphot = moddict[kk]+DM_i+RED_i
				deltadict[kk] = (obsphot-self.bfpar[kk])/self.epar[kk]
			else:
				deltadict[kk] = (moddict[kk]-self.bfpar[kk])/self.epar[kk]
		chisq = np.sum([deltadict[kk]**2.0 for kk in self.bfpar.keys()])
		return -0.5 * chisq

	def DM(self,distance):
		#distance in parsecs
		return 5.0*np.log10(distance)-5.0

class Redden(object):
	def __init__(self):
		BC = Table(np.array(h5py.File('/Users/Phill/Astro/MIST/MIST_full.h5','r')['BC']))
		BC_AV0 = BC[BC['Av'] == 0.0]
		self.bands = BC.keys()
		[self.bands.remove(x) for x in ['Teff', 'logg', '[Fe/H]', 'Av', 'Rv']]

		self.redintr = LinearNDInterpolator(
			(BC['Teff'],BC['logg'],BC['[Fe/H]'],BC['Av']),
			np.stack([BC[bb] for bb in self.bands],axis=1),
			rescale=True
			)
		self.redintr_0 = LinearNDInterpolator(
			(BC_AV0['Teff'],BC_AV0['logg'],BC_AV0['[Fe/H]']),
			np.stack([BC_AV0[bb] for bb in self.bands],axis=1),
			rescale=True
			)

	def red(self,Teff=5770.0,logg=4.44,FeH=0.0,band='V',Av=0.0):

		if (Teff > 500000.0):
			Teff = 500000.0
		if (Teff < 2500.0):
			Teff = 2500.0

		if (logg < -4.0):
			logg = -4.0
		if (logg > 9.5):
			logg = 9.5

		if (FeH > 0.5):
			FeH = 0.5
		if (FeH < -2.0):
			FeH = -2.0

		inter0 = self.redintr_0(Teff,logg,FeH)
		interi = self.redintr(Teff,logg,FeH,Av)

		bandsel = np.array([True if x == band else False for x in self.bands],dtype=bool)
		vsel = np.array([True if x == 'V' else False for x in self.bands],dtype=bool)
		A_V = interi[vsel]-inter0[vsel]
		A_X = interi[bandsel]-inter0[bandsel]
		return (A_X/A_V)*Av



if __name__ == '__main__':

	# initialize StarPar class
	STARPAR = StarPar()
		
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
			