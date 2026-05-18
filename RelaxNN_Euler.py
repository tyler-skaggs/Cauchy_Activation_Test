import sys

sys.path.insert(0, '../Utilities/')

import torch
from collections import OrderedDict

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from scipy.interpolate import griddata
import warnings
import time

torch.manual_seed(314)

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

warnings.filterwarnings('ignore')

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using CUDA')
else:
    device = torch.device('cpu')
    print('Using CPU')

class DNN(torch.nn.Module):
    def __init__(self, layers, custom_activ):
        super(DNN, self).__init__()

        # parameters
        self.depth = len(layers)-1

        # set up layer order dict
        self.activation = custom_activ()

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
            )
            layer_list.append(('activation_%d' % i, self.activation))

        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)

        # deploy Layers
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        out = self.layers(x)
        return out

# The physics-guided nerual network
class PhysicsInformedNN():
    def __init__(self, X_IC_init, X_BC_init,  X_f, layers1, layers2, custom_activ = torch.nn.Tanh):
        self.gamma = 1.4

        # space and time data
        self.x_f = torch.tensor(X_f[:, 0:1], requires_grad=True).float().to(device)
        self.t_f = torch.tensor(X_f[:, 1:2], requires_grad=True).float().to(device)

        self.x_IC = torch.tensor(X_IC_init[:, 0:1], requires_grad=True).float().to(device)
        self.x_BC = torch.tensor(X_BC_init[:, 0:1], requires_grad=True).float().to(device)

        self.t_IC = torch.tensor(X_IC_init[:, 1:2], requires_grad=True).float().to(device)
        self.t_BC = torch.tensor(X_BC_init[:, 1:2], requires_grad=True).float().to(device)

        # Main Data
        self.rho_IC = torch.tensor(X_IC_init[:, 2:3]).float().to(device)
        self.rho_BC = torch.tensor(X_BC_init[:, 2:3]).float().to(device)

        self.v_IC = torch.tensor(X_IC_init[:, 3:4]).float().to(device)
        self.v_BC = torch.tensor(X_BC_init[:, 3:4]).float().to(device)

        self.P_IC = torch.tensor(X_IC_init[:, 4:5]).float().to(device)
        self.P_BC = torch.tensor(X_BC_init[:, 4:5]).float().to(device)

        # Relaxation 'flux' Data
        self.rho_fl_IC = torch.tensor(X_IC_init[:, 5:6]).float().to(device)
        self.rho_fl_BC = torch.tensor(X_BC_init[:, 5:6]).float().to(device)

        self.v_fl_IC = torch.tensor(X_IC_init[:, 6:7]).float().to(device)
        self.v_fl_BC = torch.tensor(X_BC_init[:, 6:7]).float().to(device)

        self.P_fl_IC = torch.tensor(X_IC_init[:, 7:8]).float().to(device)
        self.P_fl_BC = torch.tensor(X_BC_init[:, 7:8]).float().to(device)

        # Deep Neural Networks
        self.dnn = DNN(layers1, custom_activ).to(device)
        self.dnn_fl = DNN(layers2, custom_activ).to(device)

        # optimizer: using the same settings
        self.optimizer = torch.optim.Adam(list(self.dnn.parameters()) + list(self.dnn_fl.parameters()), lr = 0.001, betas=(0.99, 0.99))

    def net_rho(self, x, t):
        rho = self.dnn(torch.cat([x, t], dim=1))[:, 0:1]
        return rho

    def net_v(self, x, t):
        v = self.dnn(torch.cat([x, t], dim=1))[:, 1:2]
        return v

    def net_P(self, x, t):
        P = self.dnn(torch.cat([x, t], dim=1))[:, 2:3]
        return P

    def net_fl1(self, x, t):
        fl1 = self.dnn_fl(torch.cat([x, t], dim=1))[:, 0:1]
        return fl1

    def net_fl2(self, x, t):
        fl2 = self.dnn_fl(torch.cat([x, t], dim=1))[:, 1:2]
        return fl2

    def net_fl3(self, x, t):
        fl3 = self.dnn_fl(torch.cat([x, t], dim=1))[:, 2:3]
        return fl3

    def net_residual(self, x, t):
        """ The pytorch autograd for calculating residual """
        rho = self.net_rho(x, t)
        v = self.net_v(x,t)
        rhoXv = rho * v
        E =  self.net_P(x,t) / (self.gamma - 1) + 1/2 * rho * (v**2)

        fl1 = self.net_fl1(x, t)
        fl2 = self.net_fl2(x, t)
        fl3 = self.net_fl3(x, t)

        # Time Derivatives
        rho_t = torch.autograd.grad(
            rho, t,
            grad_outputs=torch.ones_like(rho),
            retain_graph=True,
            create_graph=True
        )[0]

        rhoXv_t = torch.autograd.grad(
            rhoXv, t,
            grad_outputs=torch.ones_like(rhoXv),
            retain_graph=True,
            create_graph=True
        )[0]

        E_t = torch.autograd.grad(
            E, t,
            grad_outputs=torch.ones_like(E),
            retain_graph=True,
            create_graph=True
        )[0]

        # Space Derivatives
        fl1_x = torch.autograd.grad(
            fl1, x,
            grad_outputs=torch.ones_like(fl1),
            retain_graph=True,
            create_graph=True
        )[0]

        fl2_x = torch.autograd.grad(
            fl2, x,
            grad_outputs=torch.ones_like(fl2),
            retain_graph=True,
            create_graph=True
        )[0]

        fl3_x = torch.autograd.grad(
            fl3, x,
            grad_outputs=torch.ones_like(fl3),
            retain_graph=True,
            create_graph=True
        )[0]

        return (rho_t + fl1_x,
                rhoXv_t + fl2_x,
                E_t + fl3_x)

    def loss_func(self):
        # Pick weights and normalize to 1
        omega = np.array([0.1, 5, 5]) #Residual, Flux,  Initial/Boundary
        omega = omega/sum(omega)

        # Grab Values
        rho = self.net_rho(self.x_f, self.t_f)
        v = self.net_v(self.x_f, self.t_f)
        P = self.net_P(self.x_f, self.t_f)
        E = P / (self.gamma - 1) + 1/2 * rho * (v**2)

        fl1 = self.net_fl1(self.x_f, self.t_f)
        fl2 = self.net_fl2(self.x_f, self.t_f)
        fl3 = self.net_fl3(self.x_f, self.t_f)

        residual = self.net_residual(self.x_f, self.t_f)

        # Prediction on Boundary and Initial
        rho_BC_pred = self.net_rho(self.x_BC, self.t_BC)
        v_BC_pred = self.net_v(self.x_BC, self.t_BC)
        P_BC_pred = self.net_P(self.x_BC, self.t_BC)

        rho_IC_pred = self.net_rho(self.x_IC, self.t_IC)
        v_IC_pred = self.net_v(self.x_IC, self.t_IC)
        P_IC_pred = self.net_P(self.x_IC, self.t_IC)

        # Calculate MSE
        loss_IC = (torch.nn.MSELoss()(rho_IC_pred, self.rho_IC) +
                   torch.nn.MSELoss()(v_IC_pred, self.v_IC) +
                   torch.nn.MSELoss()(P_IC_pred, self.P_IC))

        loss_BC = (torch.nn.MSELoss()(rho_BC_pred, self.rho_BC) +
                   torch.nn.MSELoss()(v_BC_pred, self.v_BC) +
                   torch.nn.MSELoss()(P_BC_pred, self.P_BC))

        loss_residual = (torch.mean(residual[0] ** 2) +
                         torch.mean(residual[1] ** 2) +
                         torch.mean(residual[2] ** 2))

        loss_flux = (torch.nn.MSELoss()(fl1, rho*v) +
                     torch.nn.MSELoss()(fl2, rho * (v**2) + P) +
                     torch.nn.MSELoss()(fl3, v * (E + P) ))

        loss = omega[0]*loss_residual + omega[1] * loss_flux + omega[2] * (loss_IC + loss_BC)

        return loss, loss_residual, loss_flux,  loss_IC

    def train(self, epochs):
        loss_hist = np.array([0,0,0,0])
        for epoch in range(epochs):
            # Zero Gradients
            self.optimizer.zero_grad()

            # Compute Loss and Gradients
            loss, loss_residual, loss_flux, loss_IC = self.loss_func()
            loss.backward()

            # Adjust Learning Weights
            self.optimizer.step()

            if epoch % 100 == 0:
                print(
                    'Epoch %d | Loss: %.5e, L_Residual: %.5e, L_flux: %.5e, L_IC: %.5e' % (
                        epoch,
                        loss.item(),
                        loss_residual.item(),
                        loss_flux.item(),
                        loss_IC.item()
                    )
                )
            loss_hist = np.vstack((loss_hist, np.array(([
                loss.detach().cpu().numpy(),
                loss_residual.detach().cpu().numpy(),
                loss_flux.detach().cpu().numpy(),
                loss_IC.detach().cpu().numpy()
            ]))))

        return np.delete(loss_hist, (0), axis=0)

    def predict(self, X):
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)

        self.dnn.eval()
        rho = self.net_rho(x, t)
        v = self.net_v(x, t)
        P = self.net_P(x, t)

        rho = rho.detach().cpu().numpy()
        v = v.detach().cpu().numpy()
        P = P.detach().cpu().numpy()

        return rho, v, P



def analytical_Euler(x, t):
    rho = x.copy()
    v = x.copy()
    P = x.copy()

    x_0 = 0.5

    rho_L, v_L, P_L = 1, 0, 1
    rho_R, v_R, P_R = 0.125, 0, 0.1

    gamma = 1.4
    m = (gamma - 1) / (gamma + 1)
    C_L = np.sqrt(gamma)

    for i in range(0, len(x)):
        if t == 0:
            if x[i] < x_0:
                rho[i], v[i], P[i] = rho_L, v_L, P_L
            else:
                rho[i], v[i], P[i] = rho_R, v_R, P_R
        else:

            # Region 4
            P_4 = 0.30310
            rho_4 = rho_R * (P_4 + m * P_R) / (P_R + m * P_4)
            v_4 = (P_4 - P_R) * np.sqrt((1 - m) / (rho_R * (P_4 + m * P_R)))

            x_4 = x_0 + v_4 * rho_4 / (rho_4 - rho_R) * t

            # Region 3
            P_3 = P_4
            v_3 = v_4
            rho_3 = rho_L * np.power(P_3 / P_L, 1 / gamma)

            x_3 = x_0 + v_3 * t

            # Region 2
            v_2 = 2 / (gamma + 1) * ((x[i] - x_0) / t + C_L)
            rho_2 = np.power((np.power(rho_L, gamma) / (gamma * P_L) * np.power(v_2 - (x[i] - x_0) / t, 2)),
                             1 / (gamma - 1))
            P_2 = np.power(rho_2, gamma) * P_L / np.power(rho_L, gamma)

            x_2 = x_0 + ((gamma + 1) / 2 * v_3 - C_L) * t

            x_1 = x_0 - C_L * t

            if x[i] < x_1:
                rho[i], v[i], P[i] = rho_L, v_L, P_L
            elif x[i] < x_2:
                rho[i], v[i], P[i] = rho_2, v_2, P_2
            elif x[i] < x_3:
                rho[i], v[i], P[i] = rho_3, v_3, P_3
            elif x[i] < x_4:
                rho[i], v[i], P[i] = rho_4, v_4, P_4
            else:
                rho[i], v[i], P[i] = rho_R, v_R, P_R
    return rho, v, P

class Cauchy_Activation(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lambda1 = torch.nn.Parameter(torch.tensor(0.01))
        self.lambda2 = torch.nn.Parameter(torch.tensor(0.01))
        self.d = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return (self.lambda1 * x) / (x**2 + self.d**2) + self.lambda2 / (x**2 + self.d**2)


if __name__ == "__main__":

    analytic_Sol = analytical_Euler
    init = lambda x: analytic_Sol(x, 0)
    gamma = 1.4

    lb = 0 #x bounds
    ub = 1

    rho_L, v_L, P_L = 1, 0, 1
    rho_R, v_R, P_R = 0.125, 0, 0.1


    def FL(rho, v, P):
        Energy = lambda rho, v, P: P / (gamma - 1) + 1 / 2 * rho * (v ** 2)
        return rho * v, rho * (v ** 2) + P, v * (Energy(rho, v, P) + P)


    plot_high = 1.25
    plot_low = -0.25

    TIME = 0.2

    activation = Cauchy_Activation 
    epochs = 200000

    Nx = 320
    Nt = 80
    N_f = 2540

    analytic_OffOn = 1

    layers1 = [2, 384, 384, 384, 384, 384, 384, 3]
    layers2 = [2, 384, 384, 384, 384, 384, 384, 3]

    # Initial (x, t = 0)
    x = (ub - lb) * np.random.random_sample(Nx) + lb

    rho_IC, v_IC, P_IC = init(x)
    F1_IC, F2_IC, F3_IC = FL(rho_IC, v_IC, P_IC)
    X_IC = np.vstack(( x, np.zeros(len(x)), rho_IC, v_IC, P_IC, F1_IC, F2_IC, F3_IC)).T

    # Lower x Bound
    t = np.random.random_sample(Nt) * TIME
    rho_lb, v_lb, P_lb = rho_L * np.ones(len(t)), v_L * np.ones(len(t)), P_L * np.ones(len(t))
    F1_L, F2_L, F3_L = FL(rho_lb, v_lb, P_lb)
    init_lb = np.vstack((lb * np.ones(Nt), t, rho_lb, v_lb, P_lb, F1_L, F2_L, F3_L)).T

    # Upper x Bound
    t = np.random.random_sample(Nt) * TIME
    rho_rb, v_rb, P_rb = rho_R * np.ones(len(t)), v_R * np.ones(len(t)), P_R * np.ones(len(t))
    F1_R, F2_R, F3_R = FL(rho_rb, v_rb, P_rb)
    init_ub = np.vstack((ub * np.ones(Nt), t, rho_rb, v_rb, P_rb, F1_R, F2_R, F3_R)).T

    X_BC = np.vstack((init_lb, init_ub))


    # Random N_f data to train on
    X_training = np.random.random_sample((N_f, 2))
    X_training[:, 0] = (ub - lb) * X_training[:, 0] + lb
    X_training[:, 1] = X_training[:, 1] * TIME
    x = X_training[:, 0]
    t = X_training[:, 1]

    X_training = np.vstack((X_training, X_IC[:, 0:2], X_BC[:, 0:2]))

    start_time = time.time()

    model = PhysicsInformedNN(X_IC, X_BC, X_training, layers1, layers2, activation)
    loss_history = model.train(epochs)

    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Executation Time: {elapsed_time:.6f} seconds")

    # Predictions
    x_pred = np.array([np.linspace(lb, ub, 256)]).T
    t_pred = np.array([np.linspace(0, TIME, 101)]).T
    X, T = np.meshgrid(x_pred, t_pred)

    X_pred = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

    rho_pred, v_pred, P_pred = model.predict(X_pred)
    U_pred = np.array([griddata(X_pred, rho_pred.flatten(), (X, T), method='cubic'),
                        griddata(X_pred, v_pred.flatten(), (X, T), method='cubic'),
                        griddata(X_pred, P_pred.flatten(), (X, T), method='cubic')])

    #U_pred.tofile('SavedData/SIN_CauchyWE.dat')
    #torch.save(model.dnn_u.state_dict(), 'SavedData/SIN_CauchyWE_NET')

    # Loss Graph
    plt.plot(range(1, epochs + 1), loss_history[:, 0], color='red', label=r"$L(\theta)$")
    plt.ylabel(r"$\mathcal{L}$")
    plt.xlabel("Epoch")
    plt.legend()
    plt.yscale('log')
    plt.title("Loss History")
    plt.show()

    """ The aesthetic setting has changed. """
    dx = x_pred[1] - x_pred[0]

    fig, ax = plt.subplots(3,4, figsize = (10, 9))

    # Slices
    time_list = np.array([0.05, 0.1, 0.15, 0.2])
    time_index = np.round((time_list / 0.2 *100)).astype(int)
    name_list = ['Density', 'Velocity', 'Pressure']
    for i in range(0,3):
        for j in range(0,4):
            if analytic_OffOn:

                ax[i,j].plot(x_pred, analytic_Sol(x_pred, t=time_list[j])[i], 'b-', linewidth=2, label='Reference')
                error = np.linalg.norm(analytic_Sol(x_pred, t=time_list[j])[i].T - U_pred[i][time_index[j], :], ord = 2) * np.sqrt(dx)
                patch = mpatches.Patch(fill = False, edgecolor = None, visible = False, label = f"Error = %.3f" % error)
                ax[i,j].legend(handles = [patch], loc = 'lower right')
            ax[i,j].plot(x_pred, U_pred[i][time_index[j], :], 'r--', linewidth=2, label='Prediction')
            ax[i,j].set_xlabel('$x$')
            if j == 0:
                ax[i,j].set_ylabel(name_list[i])
            if i == 0:
                ax[i,j].set_title('$t =$ %.2f' % time_list[j], fontsize=15)
            ax[i,j].axis('square')
            ax[i,j].set_xlim([lb - 0.1, ub + 0.1])
            ax[i,j].set_ylim([plot_low, plot_high])


    blue_line = mlines.Line2D([], [], color='blue', markersize=15, label='Reference')
    red_line = mlines.Line2D([], [], color='red', linestyle='dashed', markersize=15, label='Prediction')
    fig.legend(handles=[blue_line, red_line])
    plt.show()