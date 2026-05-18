import sys

sys.path.insert(0, '../Utilities/')

import torch
from collections import OrderedDict

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy
from matplotlib import gridspec
from numpy import where
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
    def __init__(self, X_IC_init, X_BC_init,  X_f, layers_u, layers_v, F, custom_activ = torch.nn.Tanh):
        self.F = F

        # data
        self.x_f = torch.tensor(X_f[:, 0:1], requires_grad=True).float().to(device)
        self.t_f = torch.tensor(X_f[:, 1:2], requires_grad=True).float().to(device)

        self.x_IC = torch.tensor(X_IC_init[:, 0:1], requires_grad=True).float().to(device)
        self.t_IC = torch.tensor(X_IC_init[:, 1:2], requires_grad=True).float().to(device)

        self.x_BC = torch.tensor(X_BC_init[:, 0:1], requires_grad=True).float().to(device)
        self.t_BC = torch.tensor(X_BC_init[:, 1:2], requires_grad=True).float().to(device)

        self.u_IC = torch.tensor(X_IC_init[:, 2:3]).float().to(device)
        self.u_BC = torch.tensor(X_BC_init[:, 2:3]).float().to(device)

        self.v_IC = torch.tensor(X_IC_init[:, 3:4]).float().to(device)
        self.v_BC = torch.tensor(X_BC_init[:, 3:4]).float().to(device)

        # Deep Neural Networks
        self.dnn_u = DNN(layers_u, custom_activ).to(device)
        self.dnn_v = DNN(layers_v, custom_activ).to(device)

        # optimizer: using the same settings
        self.optimizer = torch.optim.Adam(list(self.dnn_u.parameters()) + list(self.dnn_v.parameters()), lr = 0.001, betas=(0.99, 0.99))

    def net_u(self, x, t):
        u = self.dnn_u(torch.cat([x, t], dim=1))
        return u

    def net_v(self, x, t):
        v = self.dnn_v(torch.cat([x, t], dim=1))
        return v


    def net_f(self, x, t):
        """ The pytorch autograd for calculating residual """
        u = self.net_u(x, t)
        v = self.net_v(x, t)

        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]

        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]

        v_x = torch.autograd.grad(
            v, x,
            grad_outputs=torch.ones_like(v),
            retain_graph=True,
            create_graph=True
        )[0]

        lam = 1 / (0.5 * (torch.abs(u_x) - u_x) + 1)

        return lam * (u_t + v_x)

    def loss_func(self):
        omega = np.array([0.1, 2, 10])
        omega = omega/sum(omega)

        u = self.net_u(self.x_f, self.t_f)
        v = self.net_v(self.x_f, self.t_f)

        residual = self.net_f(self.x_f, self.t_f)

        u_BC_pred = self.net_u(self.x_BC, self.t_BC)
        u_IC_pred = self.net_u(self.x_IC, self.t_IC)

        loss_IC = torch.nn.MSELoss()(u_IC_pred, self.u_IC) + torch.nn.MSELoss()(u_BC_pred, self.u_BC)

        loss_residual = torch.mean(residual ** 2)

        loss_flux = torch.nn.MSELoss()(v, self.F(u))

        loss = omega[0]*loss_residual + omega[1] * loss_flux + omega[2] * loss_IC

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

        self.dnn_u.eval()
        u = self.net_u(x, t)

        u = u.detach().cpu().numpy()

        return u



def analytical1(x, t):
    y = np.zeros(np.size(x))
    ul = 0
    ur = 1
    for i in range(0, np.size(x)):
        if x[i] <= ul * t:
            y[i] = ul
        elif x[i] > t * ur:
            y[i] = ur
        else:
            y[i] = x[i] / t
    return y

def analytical2(x, t):
    y = np.zeros(np.size(x))
    ul = 1
    ur = 0
    s = (ul + ur)/2

    for i in range(0, np.size(x)):
        if x[i] <= s * t:
            y[i] = ul
        else:
            y[i] = ur
    return y

def analytical3(x,t):
    y = np.zeros(np.size(x))
    for i in range(0, np.size(x)):
        if np.abs(x[i]) <= np.sqrt(t+1):
            y[i] = x[i]/(t+1)
        else:
            y[i] = 0
    return y

def analyticalsin(x, t):
    data = scipy.io.loadmat('./burgers_shock.mat')
    Exact = np.real(data['usol']).T
    return Exact[int(t*100)]

class Cauchy_Activation(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lambda1 = torch.nn.Parameter(torch.tensor(0.01))
        self.lambda2 = torch.nn.Parameter(torch.tensor(0.01))
        self.d = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return (self.lambda1 * x) / (x**2 + self.d**2) + self.lambda2 / (x**2 + self.d**2)


if __name__ == "__main__":

    analytic_Sol = analyticalsin
    init = lambda x: -np.sin(np.pi * x) #analytic_Sol(x, 0)

    F = lambda u: (u ** 2) / 2

    lb = -1 #x bounds
    ub = 1

    u_lb_val = 0
    u_ub_val = 0
    v_lb_val = F(u_lb_val)
    v_ub_val = F(u_ub_val)

    plot_high = 1.25
    plot_low = -1.25

    TIME = 1

    activation = Cauchy_Activation
    epochs = 100000

    Nx = 320
    Nt = 80
    N_f = 2540

    analytic_OffOn = 1

    layers_u = [2, 128, 128, 128, 128, 1]
    layers_v = [2, 64, 64, 64, 64, 1]

    # (x, t = 0)
    x = (ub - lb) * np.random.random_sample(Nx) + lb

    X_IC = np.vstack((x, np.zeros(len(x)), init(x), F(init(x)))).T

    # Lower x Bound
    t = np.random.random_sample(Nt) * TIME
    u_lb = u_lb_val * np.ones(len(t))
    v_lb = F(u_lb)
    init_lb = np.vstack((lb * np.ones(Nt), t, u_lb, v_lb)).T

    # Upper x Bound
    t = np.random.random_sample(Nt) * TIME
    u_ub = u_ub_val * np.ones(len(t))
    v_ub = F(u_ub)
    init_ub = np.vstack((ub * np.ones(Nt), t, u_ub, v_ub)).T

    X_BC = np.vstack((init_lb, init_ub))


    # Random N_f data to train on
    X_training = np.random.random_sample((N_f, 2))
    X_training[:, 0] = (ub - lb) * X_training[:, 0] + lb
    X_training[:, 1] = X_training[:, 1] * TIME
    x = X_training[:, 0]
    t = X_training[:, 1]

    X_training = np.vstack((X_training, X_IC[:, 0:2], X_BC[:, 0:2]))

    start_time = time.time()

    model = PhysicsInformedNN(X_IC, X_BC, X_training, layers_u, layers_v, F, activation)

    loss_history = model.train(epochs)

    if activation == Cauchy_Activation:
        print(model.dnn_u.activation.lambda1.item())
        print(model.dnn_u.activation.lambda2.item())
        print(model.dnn_u.activation.d.item())

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Executation Time: {elapsed_time:.6f} seconds")

    # Predictions
    x_pred = np.array([np.linspace(lb, ub, 256)]).T
    t_pred = np.array([np.linspace(0, TIME, 100)]).T
    X, T = np.meshgrid(x_pred, t_pred)

    X_pred = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

    u_pred = model.predict(X_pred)
    U_pred = griddata(X_pred, u_pred.flatten(), (X, T), method='cubic')

    #U_pred.tofile('SavedData/SIN_CauchyWE.dat')
    #torch.save(model.dnn_u.state_dict(), 'SavedData/SIN_CauchyWE_NET')

    # Loss Graph
    plt.plot(range(1, epochs + 1), loss_history[:, 0], color='red', label=r"$L(\theta)$")
    #plt.plot(range(1001, epochs + 1), loss_history[1000:, 1], color='blue', label=r"$\mathcal{L}_{Residual}$")
    #plt.plot(range(1001, epochs + 1), loss_history[1000:, 2], color='orange', label=r"$\mathcal{L}_{Flux}$")
    #plt.plot(range(1001, epochs + 1), loss_history[1000:, 3], color='green', label=r"$\mathcal{L}_{IC + BC}$")
    plt.ylabel(r"$\mathcal{L}$")
    plt.xlabel("Epoch")
    plt.legend()
    plt.yscale('log')
    plt.title("Loss History")
    plt.show()

    # Heat Map
    ####### HEAT MAP AND SLICES ##################
    """ The aesthetic setting has changed. """
    dx = x_pred[1] - x_pred[0]

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111)
    gs1 = gridspec.GridSpec(2, 4)
    gs1.update(top=0.9, bottom=0.1, left=0.1, right=0.9, wspace=0.5)

    # HEAT MAP
    ax = plt.subplot(gs1[0, :])
    h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow',
                  extent=[t.min(), t.max(), x.min(), x.max()],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=15)

    X_initial = np.vstack((X_IC, X_BC))
    ax.plot(
        X_initial[:, 1],
        X_initial[:, 0],
        'kx', label='Data (%d points)' % (X_initial[:, 2].shape[0]),
        markersize=4,  # marker size doubled
        clip_on=False,
        alpha=.5
    )

    line = np.linspace(x.min(), x.max(), 2)[:, None]
    ax.plot(.25 * np.ones((2, 1)), line, 'w-', linewidth=1)
    ax.plot(.5 * np.ones((2, 1)), line, 'w-', linewidth=1)
    ax.plot(.75 * np.ones((2, 1)), line, 'w-', linewidth=1)

    ax.set_xlabel('$t$', size=20)
    ax.set_ylabel('$x$', size=20)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.9, -0.05),
        ncol=5,
        frameon=False,
        prop={'size': 15}
    )
    ax.set_title('$u(t,x)$', fontsize=20)  # font size doubled
    ax.tick_params(labelsize=15)

    # Slices
    ax = plt.subplot(gs1[1, 0])
    if analytic_OffOn:
        ax.plot(x_pred, analytic_Sol(x_pred, t=.25), 'b-', linewidth=2, label='Reference')
        error = np.linalg.norm(analytic_Sol(x_pred, t=.25) - U_pred[25, :], 2) * np.sqrt(dx)
        text = plt.text(-1, -0.9, f"Error = %.5f" % error)
    ax.plot(x_pred, U_pred[25, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.set_title('$t = 0.25$', fontsize=15)
    ax.axis('square')
    ax.set_xlim([lb - 0.1, ub + 0.1])
    ax.set_ylim([plot_low, plot_high])

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(15)

    ax = plt.subplot(gs1[1, 1])
    if analytic_OffOn:
        ax.plot(x_pred, analytic_Sol(x_pred, t=.5), 'b-', linewidth=2, label='Reference')
        error = np.linalg.norm(analytic_Sol(x_pred, t=.5) - U_pred[50, :], 2) * np.sqrt(dx)
        text = plt.text(-1, -0.9, f"Error = %.5f" % error)
    ax.plot(x_pred, U_pred[50, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.axis('square')
    ax.set_xlim([lb - 0.1, ub + 0.1])
    ax.set_ylim([plot_low, plot_high])
    ax.set_title('$t = 0.5$', fontsize=15)

    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=5,
        frameon=False,
        prop={'size': 15}
    )

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(15)

    ax = plt.subplot(gs1[1, 2])
    if analytic_OffOn:
        ax.plot(x_pred, analytic_Sol(x_pred, t=.75), 'b-', linewidth=2, label='Reference')
        error = np.linalg.norm(analytic_Sol(x_pred, t=.75) - U_pred[75, :], 2) * np.sqrt(dx)
        text = plt.text(-1, -0.9, f"Error = %.5f" % error)
    ax.plot(x_pred, U_pred[75, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.axis('square')
    ax.set_xlim([lb - 0.1, ub + 0.1])
    ax.set_ylim([plot_low, plot_high])
    ax.set_title('$t = 0.75$', fontsize=15)

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(15)

    ax = plt.subplot(gs1[1, 3])
    if analytic_OffOn:
        ax.plot(x_pred, analytic_Sol(x_pred, t=0.99), 'b-', linewidth=2, label='Reference')
        error = np.linalg.norm(analytic_Sol(x_pred, t=0.99) - U_pred[99, :], 2) * np.sqrt(dx)
        text = plt.text(-1, -0.9, f"Error = %.5f" % error)
    ax.plot(x_pred, U_pred[99, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.axis('square')
    ax.set_xlim([lb - 0.1, ub + 0.1])
    ax.set_ylim([plot_low, plot_high])
    ax.set_title('$t = 1$', fontsize=15)

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(15)

    plt.show()