import tensorflow as tf
import pdb
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def TFGenerateDCPhase(nx=100,ny=120,LFac=5,QFac=0.1,SFac=2):
	# LFac = 5
	# QFac = 0.1
	# nx, ny = (3, 2)
	Linx = tf.linspace(-np.pi, np.pi, nx)
	Liny = tf.linspace(-np.pi, np.pi, ny)
	X, Y = tf.meshgrid(Linx, Liny,indexing='ij')

	Rnd=tf.random_uniform([11])

	AL=(Rnd[0]-0.5)*LFac*X+(Rnd[1]-0.5)*LFac*Y+(Rnd[2]-0.5)*QFac*( tf.pow(X,2)  )+(Rnd[3]-0.5)*QFac*(  tf.pow(Y,2)  );
	BL=(Rnd[4]-0.5)*LFac*X+(Rnd[5]-0.5)*LFac*Y+(Rnd[6]-0.5)*QFac*( tf.pow(X,2)  )+(Rnd[7]-0.5)*QFac*(  tf.pow(Y,2) );
	PX=(Rnd[8]-0.5)*tf.sin(AL)+(Rnd[9]-0.5)*tf.sin(BL);
	DCPhase=Rnd[10]*2*np.pi-np.pi;
	PX=PX*2*SFac*np.pi+DCPhase;
	Out=tf.exp(tf.complex(PX*0, PX));
	return Out

def TFGenerateRandomSinPhase(nx=100,ny=120,LFac=5,QFac=0.1,SFac=2):
	# LFac = 5
	# QFac = 0.1
	# nx, ny = (3, 2)
	Linx = tf.linspace(-np.pi, np.pi, nx)
	Liny = tf.linspace(-np.pi, np.pi, ny)
	X, Y = tf.meshgrid(Linx, Liny,indexing='ij')

	Rnd=tf.random_uniform([11])

	AL=(Rnd[0]-0.5)*LFac*X+(Rnd[1]-0.5)*LFac*Y+(Rnd[2]-0.5)*QFac*( tf.pow(X,2)  )+(Rnd[3]-0.5)*QFac*(  tf.pow(Y,2)  );
	BL=(Rnd[4]-0.5)*LFac*X+(Rnd[5]-0.5)*LFac*Y+(Rnd[6]-0.5)*QFac*( tf.pow(X,2)  )+(Rnd[7]-0.5)*QFac*(  tf.pow(Y,2) );
	PX=(Rnd[8]-0.5)*tf.sin(AL)+(Rnd[9]-0.5)*tf.sin(BL);
	DCPhase=Rnd[10]*2*np.pi-np.pi;
	PX=PX*2*SFac*np.pi+DCPhase;
	Out=tf.exp(tf.complex(PX*0, PX));
	return Out

def GenerateRandomSinPhase(nx=100,ny=120,LFac=5,QFac=0.1,SFac=2):
	# LFac = 5
	# QFac = 0.1
	# nx, ny = (3, 2)
	Linx = np.linspace(-np.pi, np.pi, nx)
	Liny = np.linspace(-np.pi, np.pi, ny)
	X, Y = np.meshgrid(Linx, Liny)

	Rnd=np.random.rand(11)
	AL=(Rnd[0]-0.5)*LFac*X+(Rnd[1]-0.5)*LFac*Y+(Rnd[2]-0.5)*QFac*( np.power(X,2)  )+(Rnd[3]-0.5)*QFac*(  np.power(Y,2)  );
	BL=(Rnd[4]-0.5)*LFac*X+(Rnd[5]-0.5)*LFac*Y+(Rnd[6]-0.5)*QFac*( np.power(X,2) )+(Rnd[7]-0.5)*QFac*( np.power(Y,2) );
	PX=(Rnd[8]-0.5)*np.sin(AL)+(Rnd[9]-0.5)*np.sin(BL);
	DCPhase=Rnd[10]*2*np.pi-np.pi;
	PX=PX*2*SFac*np.pi+DCPhase;
	Out=np.exp(1j*PX);
	return Out

def GShow(A):
	ax = plt.subplot(111)
	im = ax.imshow(A)
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	plt.colorbar(im, cax=cax)
	plt.show()
	ax.set_title('Title')

def GShowC(C):
	ax = plt.subplot(121)
	im = ax.imshow(np.abs(C))
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	plt.colorbar(im, cax=cax)
	ax.set_title('Title')

	ax2 = plt.subplot(122)
	im = ax2.imshow(np.angle(C))
	divider = make_axes_locatable(ax2)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	plt.colorbar(im, cax=cax)
	ax2.set_title('Title2')

	plt.show()
	
def GShowC4(C):
	if C.ndim==3:
		if C.shape[2]==1:
			C=np.reshape(C,C.shape[0:2])
	ax = plt.subplot(221)
	im = ax.imshow(np.abs(C))
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	plt.colorbar(im, cax=cax)
	ax.set_title('abs')

	ax2 = plt.subplot(222)
	im = ax2.imshow(np.angle(C))
	divider = make_axes_locatable(ax2)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	plt.colorbar(im, cax=cax)
	ax2.set_title('angle')

	ax = plt.subplot(223)
	im = ax.imshow(np.real(C))
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	plt.colorbar(im, cax=cax)
	ax.set_title('real')

	ax2 = plt.subplot(224)
	im = ax2.imshow(np.imag(C))
	divider = make_axes_locatable(ax2)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	plt.colorbar(im, cax=cax)
	ax2.set_title('imag')

	plt.show()


print('hello')
Q=GenerateRandomSinPhase()

# plt.matshow(samplemat((15, 15)))

# plt.show()

"""
function Out=GenerateRandomSinPhase(N,LFac,QFac)
if(numel(N)==1)
    N=[N N];
end
if(nargin<4)
    nP=2;
end
if(nargin<3)
    QFac=1;
end
if(nargin<2)
    LFac=5;
end
Linx=linspace(-pi,pi,N(1));
Liny=linspace(-pi,pi,N(2));

[X,Y]=ndgrid(Linx,Liny);

AL=(rand-0.5)*LFac*X+(rand-0.5)*LFac*Y+(rand-0.5)*QFac*(X.^2)+(rand-0.5)*QFac*(Y.^2);
BL=(rand-0.5)*LFac*X+(rand-0.5)*LFac*Y+(rand-0.5)*QFac*(X.^2)+(rand-0.5)*QFac*(Y.^2);
PX=(rand-0.5)*sin(AL)+(rand-0.5)*sin(BL);
DCPhase=rand*2*pi-pi;
PX=PX*pi+DCPhase;
Out=exp(1i*PX);
"""
