import numpy as np


def ldr(src_data, alpha, U):

# % -------------------------------------------------------------------------
# % An implementation of
# %   C. Lee, C. Lee, and Chang-Su Kim, "Contrast enahancement based on
# %   layered difference representation of 2D histograms," IEEE Trans. Image
# %   Image Process., vol. 22, no. 12, pp. 5372-5384, Dec. 2013
# %
# % -------------------------------------------------------------------------
# % Input variables (see the paper for details)
# %   src_data : can be either 2D histogram or gray scale image. This script
# %   automatically detects based on its dimension.
# %   alpha    : controls the level of enhancement
# %   U        : U matrix in Equation (31). If it is provided, we can save
# %   the computation time.
# %
# % Output variables
# %   x    : Output transformation function.
# % 
# % -------------------------------------------------------------------------
# %                           written by Chulwoo Lee, chulwoo@mcl.korea.ac.kr





	R, C = src_data.shape
	if R==255 and C==255:
	    h2D_in = src_data
	else:
	    in_Y = src_data
	    
	    # % unordered 2D histogram acquisition
	    h2D_in = np.zeros((256,256))
	    
	    for j in range(1,R+1):
	        for i in range(1,R+1):
	            ref = in_Y[j-1,i-1]
	            
	            if j!=R:
	                trg = in_Y[j,i-1]	            
	            if i!=C:
	                trg = in_Y[j-1,i]
			
	            h2D_in[np.maximum(trg,ref),np.minimum(trg,ref)] = h2D_in[np.maximum(trg,ref),np.minimum(trg,ref)] + 1	                   
	    del ref,trg


	# Intra-Layer Optimization
	D = np.zeros((255,255))
	s = np.zeros((255,1))


	# iteration start
	for layer in range(1,255):
	    h_l = np.zeros((256-layer,1))
	    tmp_idx = 1
	    for j in range(1+layer,257):
	        i=j-layer
	        h_l[tmp_idx-1,0] = np.log(h2D_in[j-1,i-1]+1)
	        tmp_idx = tmp_idx+1
	    del tmp_idx

	    s[layer-1,0] = np.sum(h_l)

	    # % if all elements in h_l is zero, then skip
	    if s[layer-1,0] == 0:
	        continue
	    

	    # % Convolution
	    m_l = np.convolve(np.squeeze(h_l), np.ones((layer,)))           # % Equation (30)
	    
	    d_l = (m_l - np.amin(m_l))/U[:,layer-1]        # % Equation (33)

	    if (np.sum(d_l) == 0):
	        continue
	    
	    D[:,layer-1] = d_l/sum(d_l)

	# %% Inter-Layer Aggregation
	W = (s/np.amax(s))**alpha                          #% Equation (23)
	d = np.matmul(D,W)                                        #% Equation (24)

	# %% reconstruct transformation function
	d = d/np.sum(d)      # % normalization
	tmp = np.zeros((256,1))
	for k in range(1,255):
	    tmp[k] = tmp[k-1] + d[k-1]

	x = (255*tmp).astype(np.uint8)
	return x
