%-------------------------------------------------------------------
% Simple introductionary example to particle filtering
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
sys.a=0.5;              % Scale parameter in the process model
sys.c=1;                % Scale parameter in the observation model
sys.sigmav=1.0;         % Standard deviation of the process noise
sys.sigmae=0.1;         % Standard deviation of the measurement noise
sys.T=100;              % Number of time steps
par.N=1000;             % Number of particles
par.fixedlag=10;        % The lag in the smoother 
                        % (optimal MSE if log(sys.T))

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
% Particle filter
%-------------------------------------------------------------------
p(:,1)=zeros(par.N,1);  % Set initial particle states

for tt=1:sys.T
   % Selection (resampling) and mutation (propagation)
   if ~(tt==1)
      a(:,tt)=randsample(par.N,par.N,'true',W(:,tt-1));
      p(:,tt)=sys.a*p(a(:,tt),tt-1)+sys.sigmav*randn(par.N,1);
   end
   
   % Calculate log-weights (for increased accuracy)
   %w(:,tt)=normpdf(y(tt),sys.c*p(:,tt),sys.sigmae);
   w(:,tt) = -0.5*log(2*pi*sys.sigmae^2) - 0.5/sys.sigmae^2.*( y(tt)-sys.c*p(:,tt) ).^2;
   
   % Transform weights to usual base
   wmax = max( w(:,tt) );
   w(:,tt) = exp( w(:,tt) - wmax );
   
   % Normalise of weights
   W(:,tt)=w(:,tt)/sum(w(:,tt));
   
   % Calculate state estimate
   xhatPF(tt)=W(:,tt)'*p(:,tt);
end

%-------------------------------------------------------------------
% Fixed-lag smoothing
%-------------------------------------------------------------------

for tt=1:sys.T
   % reconstruct particle trajectory
   atmp=1:par.N;
   for ii=(min(tt+par.fixedlag,sys.T)):-1:(tt+1); atmp=a(atmp,ii); end    

   % calculate the smoothed estimate
   xhatFL(tt)=W(:,min(tt+par.fixedlag,sys.T))'*p(atmp,tt);
end


%-------------------------------------------------------------------
% Plot the true and estimated states
%-------------------------------------------------------------------
plot(1:sys.T,x,'k',1:sys.T,xhatFL,'r');
xlabel('time'); ylabel('latent state (x)');
legend('true','Fixed-lag est.');
