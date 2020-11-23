clear all
%Image size
nPix=31;
%the vector for x-axis
vX = [1 0 0];
magX=(vX(1)^2+vX(2)^2+vX(3)^2)^0.5;
% Sample points on x-axis
InX=floor(magX*nPix);
x = 0:InX;
x=x/InX;
%the unit vector for y-axis
vY = [0 1 0];
magY=(vY(1)^2+vY(2)^2+vY(3)^2)^0.5;
% Sample points on y-axis
InY=floor(magY*nPix);
y = 0:InY;
y=y/InY;
% Construct a two-dimensional grid.
[X,Y] =meshgrid(x,y);
% Load list dataIn of Ug's, h,k,l, Ug(real),Ug(imag);
dataIn=importdata('StructureFactors.txt');
%Make set of g-vectors
g=dataIn(:,1:3);
% Conversion factor: output is in volts
RScattFacToVolts=47.913838;
%Make column matrix of Ugs [Ug(real),Ug(imag)]
CUg=(dataIn(:,6)+1i*dataIn(:,7));%*RScattFacToVolts;

RreUr=zeros(InY+1,InX+1);
RimUr=zeros(InY+1,InX+1);
for i=1:InX+1
  for j=1:InY+1
    for n=1:size(g,1)%Fourier sum for the point Rr
      Rr=X(j,i)*vX+Y(j,i)*vY;
      RreUr(j,i)=RreUr(j,i)+real(CUg(n)*exp(2*pi*1i*(g(n,:)*Rr')));
      RimUr(j,i)=RimUr(j,i)+imag(CUg(n)*exp(2*pi*1i*(g(n,:)*Rr')));
    end
  end
end

RreUr=RreUr*RScattFacToVolts;
RimUr=RimUr*RScattFacToVolts;
imagesc(x,y,RreUr)%real part
axis image, axis xy
