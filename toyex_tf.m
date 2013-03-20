%-------------------------------------------------------------------
% Simple introductionary example to particle filtering
% in a linear Gaussian state space model
%
% Implements the two-filter smoother from the paper by
% Briers, Doucet, Maskell (2009) with the title
% "Smoothing algorithms for state-space models"
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
sys.a=1;                % Scale parameter in the process model
sys.c=1;                % Scale parameter in the observation model
sys.sigmav=0.1;         % Standard deviation of the process noise
sys.sigmae=0.1;         % Standard deviation of the measurement noise
sys.T=100;              % Number of time steps
par.N=1000;             % Number of particles
par.P=10000;

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
p(1:par.N,1)=0;  % Set initial particle states

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
% Information filter (backward particle filtering)
%-------------------------------------------------------------------
p2(1:par.N,sys.T)=xhatPF(sys.T);  % Set final particle states

% Generate sample paths for the artificial distribution
a(1:par.P,1)=sqrt(sys.sigmav^2/(1-sys.a^2))*randn(par.P,1);  
for tt=2:sys.T
   a(:,tt)=sys.a*a(:,tt-1)+sys.sigmav*randn(par.P,1);
end

for tt=sys.T:-1:1
   % Selection (resampling) and mutation (propagation)
   if ~(tt==sys.T)
      nIdx=randsample(par.N,par.N,'true',W2(:,tt+1));
      p2(:,tt)=p2(nIdx,tt+1)+sys.sigmav*randn(par.N,1);
      %p2(:,tt)=p2(:,tt)/sys.a;
   end
   
   % Calculate weights
   if ~(tt==sys.T)
       % Compute the artificial distribution
       for jj=1:par.N
           if ~(tt==1)
               g(jj,tt)=mean(normpdf(p2(jj,tt),sys.a*a(:,tt-1),sys.sigmav));
           else
               g(jj,tt)=g(jj,tt+1);
           end
       end
       w2(:,tt)=normpdf(y(tt),sys.c*p2(:,tt),sys.sigmae).*g(:,tt)./g(:,tt+1);
   else
       g(1:par.N,tt)=1;
       w2(:,tt)=normpdf(y(tt),sys.c*p2(:,tt),sys.sigmae).*g(:,tt);
   end
   
   % Normalise of weights
   W2(:,tt)=w2(:,tt)/sum(w2(:,tt));
   
   % Calculate state estimate
   xhatIF(tt)=W2(:,tt)'*p2(:,tt);
end

%-------------------------------------------------------------------
% Two-filter smoother
%-------------------------------------------------------------------
for tt=2:sys.T
    for jj=1:par.N
        % Calculate smoothing weight for each particle
        f=normpdf(p2(jj,tt),sys.a*p(:,tt-1),sys.sigmav);
        W3(jj,tt)=W2(jj,tt)*W(:,tt-1)'*f/g(jj,tt);
    end
    
    % Normalise the smoothing weights
    w3(:,tt)=W3(:,tt)/sum(W3(:,tt));
    
    % Estimate the smoothed state
    xhatTFS(tt)=w3(:,tt)'*p2(:,tt);
end

%-------------------------------------------------------------------
% Plot the true and estimated states
%-------------------------------------------------------------------
plot(1:sys.T,x,'k',1:sys.T,xhatTFS,'r');
xlabel('time'); ylabel('latent state (x)');
legend('true','TF-smoother est.');
