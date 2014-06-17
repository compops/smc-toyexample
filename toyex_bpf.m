%-------------------------------------------------------------------
% Simple introductionary example to information filtering
% in a linear Gaussian state space model
%
% Written by: Johan Dahlin, Linköping University, Sweden
%             (johan.dahlin (at) isy.liu.se)
%
% Copyright (c) 2013 Johan Dahlin [ johan.dahlin (at) liu.se ]
%               
% Date:       2013-03-20
%
%-------------------------------------------------------------------

clear all;

%-------------------------------------------------------------------
% Parameters
%-------------------------------------------------------------------
sys.a=1;                % Scale parameter in the process model
sys.c=1;                % Scale parameter in the observation model
sys.sigmav=1.0;         % Standard deviation of the process noise
sys.sigmae=0.1;         % Standard deviation of the measurement noise
sys.T=100;              % Number of time steps
par.N=1000;             % Number of particles

%-------------------------------------------------------------------
% Generate data
%-------------------------------------------------------------------
% The system is
% x(t+1) = sys.a * x(t) + v(t),   v~N(0,sys.sigmav^2)
% y(t)   = sys.c * x(t) + e(t),   e~N(0,sys.sigmae^2)
%

% Set initial state
x(1)=0;            

for tt=1:sys.T
   x(tt+1) =sys.a*x(tt)+sys.sigmav*randn; 
   y(tt)   =sys.c*x(tt)+sys.sigmae*randn;
end
x=x(1:sys.T);

%-------------------------------------------------------------------
% Information filter (backward particle filtering)
%-------------------------------------------------------------------
p(:,sys.T)=zeros(par.N,1);  % Set final particle states

for tt=sys.T:-1:1
   % Selection (resampling) and mutation (propagation)
   if ~(tt==sys.T)
      nIdx=randsample(par.N,par.N,'true',W(:,tt+1));
      p(:,tt)=p(nIdx,tt+1)+sys.sigmav*randn(par.N,1);
   end
   
   % Compute the artificial distribution
   g(:,tt)=1;
   
   % Calculate weights
   if ~(tt==sys.T)
       tmp = -0.5 * log( 2 * pi * sys.sigmae^2) - 0.5/sys.sigmae^2.*( y(tt) - sys.c*p(:,tt) ).^2;
       w(:,tt) = tmp + log(g(:,tt)) - log(g(:,tt+1));
   else
       tmp = -0.5 * log( 2 * pi * sys.sigmae^2) - 0.5/sys.sigmae^2.*( y(tt) - sys.c*p(:,tt) ).^2;
       w(:,tt)= tmp + log(g(:,tt));
   end
   
   % Transform from log-weights
   wmax    = max( w(:,tt) );
   w(:,tt) = exp( w(:,tt) - wmax );
   
   % Normalise of weights
   W(:,tt)=w(:,tt)/sum(w(:,tt));
   
   % Calculate state estimate
   xhatIF(tt)=W(:,tt)'*p(:,tt);
end

%-------------------------------------------------------------------
% Plot the true and estimated states
%-------------------------------------------------------------------
plot(1:sys.T,x,'k',1:sys.T,xhatIF,'r');
xlabel('time'); ylabel('latent state (x)');
legend('true','IF est.');
