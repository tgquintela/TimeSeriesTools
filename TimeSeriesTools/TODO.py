
from numpy.fft import fft
from numpy import zeros, floor, log10, log, mean, array, sqrt, vstack, cumsum, \
				  ones, log2, std
from numpy.linalg import svd, lstsq


###############################################################################
############################## WRAPPER ########################################
###############################################################################
# TODO Wrapper or accept matrices

# TODO:
# Box-counting fractal dimension (Hausdorf)
# Sphere-counting FD
# Wavelet-counting FD
# Information Dimension FD
# Correlation Dimension
# Generalizaed or Renyi dimension
# Packing dimension
# Hausdorff dimension
# Multifractal dimension
# Local connected dimension
# Uncertainty exponents
'''
G. Meyer-Kress, "Application of dimension algorithms to experimental
chaos," in "Directions in Chaos", Hao Bai-Lin ed., (World Scientific,
Singapore, 1987) p. 122
S. Ellner, "Estmating attractor dimensions for limited data: a new
method, with error estimates" Physi. Lettr. A 113,128 (1988)
P. Grassberger, "Estimating the fractal dimensions and entropies
of strange attractors", in "Chaos", A.V. Holden, ed. (Princeton
University Press, 1986, Chap 14)
G. Meyer-Kress, ed. "Dimensions and Entropies in Chaotic Systems -
Quantification of Complex Behaviour", vol 32 of Springer series
in Synergetics (Springer Verlag, Berlin, 1986)
N.B. Abraham, J.P. Gollub and H.L. Swinney, "Testing nonlinear
dynamics," Physica 11D, 252 (1984)
'''


#pyEEG
def ap_entropy(X, M, R):
	"""Computer approximate entropy (ApEN) of series X, specified by M and R.

	Suppose given time series is X = [x(1), x(2), ... , x(N)]. We first build
	embedding matrix Em, of dimension (N-M+1)-by-M, such that the i-th row of Em 
	is x(i),x(i+1), ... , x(i+M-1). Hence, the embedding lag and dimension are
	1 and M-1 respectively. Such a matrix can be built by calling pyeeg function 
	as Em = embed_seq(X, 1, M). Then we build matrix Emp, whose only 
	difference with Em is that the length of each embedding sequence is M + 1

	Denote the i-th and j-th row of Em as Em[i] and Em[j]. Their k-th elments 
	are	Em[i][k] and Em[j][k] respectively. The distance between Em[i] and Em[j]
	is defined as 1) the maximum difference of their corresponding scalar 
	components, thus, max(Em[i]-Em[j]), or 2) Euclidean distance. We say two 1-D
	vectors Em[i] and Em[j] *match* in *tolerance* R, if the distance between them 
	is no greater than R, thus, max(Em[i]-Em[j]) <= R. Mostly, the value of R is
	defined as 20% - 30% of standard deviation of X. 

	Pick Em[i] as a template, for all j such that 0 < j < N - M + 1, we can 
	check whether Em[j] matches with Em[i]. Denote the number of Em[j],  
	which is in the range of Em[i], as k[i], which is the i-th element of the 
	vector k. The probability that a random row in Em matches Em[i] is 
	\simga_1^{N-M+1} k[i] / (N - M + 1), thus sum(k)/ (N - M + 1), 
	denoted as Cm[i].

	We repeat the same process on Emp and obtained Cmp[i], but here 0<i<N-M 
	since the length of each sequence in Emp is M + 1.

	The probability that any two embedding sequences in Em match is then 
	sum(Cm)/ (N - M +1 ). We define Phi_m = sum(log(Cm)) / (N - M + 1) and
	Phi_mp = sum(log(Cmp)) / (N - M ).

	And the ApEn is defined as Phi_m - Phi_mp.

	Notes
	-----
	#. Please be aware that self-match is also counted in ApEn. 
	#. This function now runs very slow. We are still trying to speed it up.

	References
	----------
	Costa M, Goldberger AL, Peng CK, Multiscale entropy analysis of biolgical
	signals, Physical Review E, 71:021906, 2005

	See also
	--------
	samp_entropy: sample entropy of a time series
	
	Notes
	-----
	Extremely slow implementation. Do NOT use if your dataset is not small.

	"""
	N = len(X)

	Em = embed_seq(X, 1, M)	
	Emp = embed_seq(X, 1, M + 1) #	try to only build Emp to save time

	Cm, Cmp = zeros(N - M + 1), zeros(N - M)
	# in case there is 0 after counting. Log(0) is undefined.

	for i in xrange(0, N - M):
#		print i
		for j in xrange(i, N - M): # start from i, self-match counts in ApEn
#			if max(abs(Em[i]-Em[j])) <= R:# compare N-M scalars in each subseq v 0.01b_r1
			if in_range(Em[i], Em[j], R):
				Cm[i] += 1																						### Xin Liu
				Cm[j] += 1
				if abs(Emp[i][-1] - Emp[j][-1]) <= R: # check last one
					Cmp[i] += 1
					Cmp[j] += 1
		if in_range(Em[i], Em[N-M], R):
			Cm[i] += 1
			Cm[N-M] += 1
		# try to count Cm[j] and Cmp[j] as well here
	
#		if max(abs(Em[N-M]-Em[N-M])) <= R: # index from 0, so N-M+1 is N-M  v 0.01b_r1
#	if in_range(Em[i], Em[N - M], R):  # for Cm, there is one more iteration than Cmp
#			Cm[N - M] += 1 # cross-matches on Cm[N - M]
	
	Cm[N - M] += 1 # Cm[N - M] self-matches
#	import code;code.interact(local=locals())
	Cm /= (N - M +1 )
	Cmp /= ( N - M )
#	import code;code.interact(local=locals())
	Phi_m, Phi_mp = sum(log(Cm)),  sum(log(Cmp))

	Ap_En = (Phi_m - Phi_mp) / (N - M)

	return Ap_En


#pyEEG
def samp_entropy(X, M, R):
	"""Computer sample entropy (SampEn) of series X, specified by M and R.

	SampEn is very close to ApEn. 

	Suppose given time series is X = [x(1), x(2), ... , x(N)]. We first build
	embedding matrix Em, of dimension (N-M+1)-by-M, such that the i-th row of Em 
	is x(i),x(i+1), ... , x(i+M-1). Hence, the embedding lag and dimension are
	1 and M-1 respectively. Such a matrix can be built by calling pyeeg function 
	as Em = embed_seq(X, 1, M). Then we build matrix Emp, whose only 
	difference with Em is that the length of each embedding sequence is M + 1

	Denote the i-th and j-th row of Em as Em[i] and Em[j]. Their k-th elments 
	are	Em[i][k] and Em[j][k] respectively. The distance between Em[i] and Em[j]
	is defined as 1) the maximum difference of their corresponding scalar 
	components, thus, max(Em[i]-Em[j]), or 2) Euclidean distance. We say two 1-D
	vectors Em[i] and Em[j] *match* in *tolerance* R, if the distance between them 
	is no greater than R, thus, max(Em[i]-Em[j]) <= R. Mostly, the value of R is
	defined as 20% - 30% of standard deviation of X. 

	Pick Em[i] as a template, for all j such that 0 < j < N - M , we can 
	check whether Em[j] matches with Em[i]. Denote the number of Em[j],  
	which is in the range of Em[i], as k[i], which is the i-th element of the 
	vector k.

	We repeat the same process on Emp and obtained Cmp[i], 0 < i < N - M.

	The SampEn is defined as log(sum(Cm)/sum(Cmp))

	References
	----------
	.. [1] Costa M, Goldberger AL, Peng C-K, Multiscale entropy analysis of
	   biolgical signals, Physical Review E, 71:021906, 2005

	See also
	--------
	ap_entropy: approximate entropy of a time series

	Notes
	-----
	Extremely slow computation. Do NOT use if your dataset is not small and you
	are not patient enough.

	"""

	N = len(X)

	Em = embed_seq(X, 1, M)	
	Emp = embed_seq(X, 1, M + 1)

	Cm, Cmp = zeros(N - M - 1) + 1e-100, zeros(N - M - 1) + 1e-100
	# in case there is 0 after counting. Log(0) is undefined.

	for i in xrange(0, N - M):
		for j in xrange(i + 1, N - M): # no self-match
#			if max(abs(Em[i]-Em[j])) <= R:  # v 0.01_b_r1 
			if in_range(Em[i], Em[j], R):
				Cm[i] += 1
#			if max(abs(Emp[i] - Emp[j])) <= R: # v 0.01_b_r1
				if abs(Emp[i][-1] - Emp[j][-1]) <= R: # check last one
					Cmp[i] += 1

	Samp_En = log(sum(Cm)/sum(Cmp))

	return Samp_En
