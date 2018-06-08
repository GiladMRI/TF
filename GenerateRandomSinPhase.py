import tensorflow as tf
import pdb
import numpy as np

def GP():
	LFac = 5
	QFac = 0.1
	nx, ny = (3, 2)
	Linx = np.linspace(-np.pi, np.pi, nx)
	Liny = np.linspace(np.pi, np.pi, ny)
	X, Y = np.meshgrid(Linx, Linx)

	Rnd=np.random.rand(11)
	AL=(Rnd[0]-0.5)*LFac*X+(Rnd[1]-0.5)*LFac*Y+(Rnd[2]-0.5)*QFac*( np.power(X,2)  )+(Rnd[3]-0.5)*QFac*(  np.power(Y,2)  );
	BL=(Rnd[4]-0.5)*LFac*X+(Rnd[5]-0.5)*LFac*Y+(Rnd[6]-0.5)*QFac*( np.power(X,2) )+(Rnd[7]-0.5)*QFac*( np.power(Y,2) );
	PX=(Rnd[8]-0.5)*np.sin(AL)+(Rnd[9]-0.5)*np.sin(BL);
	DCPhase=Rnd[10]*2*np.pi-np.pi;
	PX=PX*np.pi+DCPhase;
	Out=np.exp(1j*PX);
	return Out


print('hello')

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
