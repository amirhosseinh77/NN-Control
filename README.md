# NN-based Control Methods for Dynamic Systems

## Reference Tracking Neural Network Controller for Dynamic Systems
The controller utilizes Neural Networks to control a nonlinear dynamic system by tracking a given reference signal. The main goal is to minimize the error between the system output and the desired reference trajectory.

### Block Diagram
![diagram](https://github.com/amirhosseinh77/NN-Control/assets/56114938/a5be77fe-4f61-4b5d-9da5-6a41cba9ec45)

### Components Description
1. **Reference Model**:
   - The reference model is defined by the transfer function:
     ```math
     G_m(s) = \frac{K}{\frac{1}{\omega_n^2}s^2 + \frac{2\xi}{\omega_n}s + 1}
     ```
   - Generates the desired reference signal.
   - Uses the reference input which is given by:
     ```math
     r(k) = \sin\left(\frac{2\pi k}{25}\right) + \sin\left(\frac{2\pi k}{10}\right)
     ```

2. **Dynamic System**:
   - The nonlinear dynamic system is represented by:
     ```math
     y(k+1) = \frac{y(k) y(k-1) u(k) + u^3(k) + 0.5 y(k-1)}{1 + y^2(k) + y^2(k-1)}
     ```
     
3. **NN Controller**:
   - Adjusts the control input to minimize the tracking error.
   - Utilizes gradients and parameters of the RBF NN model to update the control signal.


### Result
![NN-based Control](https://github.com/amirhosseinh77/NN-Control/assets/56114938/8820083c-1c6b-42a9-8024-d386a51f6eb0)

### Reference
[1] Slema, S., Errachdi, A., & Benrejeb, M. (2018, March). A radial basis function neural network model reference adaptive controller for nonlinear systems. In 2018 15th International Multi-Conference on Systems, Signals & Devices (SSD) (pp. 958-964). IEEE.


### Actor-Critic-Identifier DRL Framework for Pendulum Balancing Problem 
![DRL Pendulum](https://github.com/amirhosseinh77/NN-Control/assets/56114938/bc79eeca-b8dc-4384-a373-cbefcde12db3)




