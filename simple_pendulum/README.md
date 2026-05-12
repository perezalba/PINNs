# Simple pendulum

## Problem overview

The example problem we solve here is the simple pendulum. We describe the trajectory of the mass using $\theta(t)$, which corresponds to the angle between the mass and the vertical component. We describe the movement with the following differential equation: 
$$
\ddot{\theta}=-\omega^2\sin\theta
$$
with the initial conditions
$$
\theta(0) = \theta_0~~,~~\dot{\theta}(0)=0~.
$$
We incorporate two types of PINNs: the standard one using a Fully Connected Neural Network (FCNN) and a Sinusoidal Representations Network (SIREN) Neural Network. The second should improve the predictions for higher frequency oscillations. Credits to Vincent Sitzmann ($\texttt{vsitzmann}$ on github) for the definition of the SIREN neural network, and to Ben Moseley for the definition of the standard FCNN.