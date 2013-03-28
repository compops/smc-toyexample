%-------------------------------------------------------------------
% Simple introductionary example to particle smoothing
% in a linear Gaussian state space model
%
% Implements the Forward-filtering Backward-Smoother from
% the paper by Doucet, Godsill, Andrieu (2000) with the title
% "On sequential Monte Carlo sampling methods for Bayesian filtering"
%
% Written by: Johan Dahlin, Linköping University, Sweden
%             (johan.dahlin (at) isy.liu.se)
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
      nIdx=randsample(par.N,par.N,'true',W(:,tt-1));
      p(:,tt)=sys.a*p(nIdx,tt-1)+sys.sigmav*randn(par.N,1);
   end
   
   % Calculate weights 
   w(:,tt)=normpdf(y(tt),sys.c*p(:,tt),sys.sigmae);
   
   % Normalise of weights
   W(:,tt)=w(:,tt)/sum(w(:,tt));
   
   % Calculate state estimate
   xhatPF(tt)=W(:,tt)'*p(:,tt);
end

%-------------------------------------------------------------------
% Particle smoother (Forward-filtering Backward-Smoother) (FFBSm)
%-------------------------------------------------------------------
Ws(:,sys.T)=W(:,sys.T);
xhatPS(sys.T)=xhatPF(sys.T);

for tt=sys.T-1:-1:1
    % Compute the normalisation term
    for jj=1:par.N; v(jj,tt)=sum(W(:,tt).*normpdf(p(jj,tt+1),sys.a*p(:,tt),sys.sigmav)); end
    
    % Compute the smoothing weight
    for ii=1:par.N; Ws(ii,tt)=W(ii,tt)*sum(Ws(:,tt+1).*normpdf(p(:,tt+1),sys.a*p(ii,tt),sys.sigmav)./v(:,tt)); end
    
    % Compute the state estimate
    xhatPS(tt)=Ws(:,tt)'*p(:,tt);
end

%-------------------------------------------------------------------
% Plot the true and estimated states
%-------------------------------------------------------------------
plot(1:sys.T,x,'k',1:sys.T,xhatPS,'r');
xlabel('time'); ylabel('latent state (x)');
legend('true','PF est.');
