import matplotlib.pyplot as plt
import seaborn as sns
import torch


def plot_heatmap(data, mask=None, title="Heatmap", xlabel="X-axis", ylabel="Y-axis"):
    plt.figure(figsize=(6, 5))
    
    # If mask is provided, the background (facecolor) will show through the True values
    ax = sns.heatmap(data, 
                     mask=mask, 
                     annot=True, 
                     cmap="viridis", 
                     fmt=".2f")
    
    # Set the background color to black (this is what "masked" cells will look like)
    ax.set_facecolor('black')
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()

def compute_exploitability_multigroup(solvers, policies, flows, theta_list):
    """
    Per-group exploitability at the Nash equilibrium candidate.
    """
    L_total = torch.stack(flows).sum(dim=0)
    results = {}
    with torch.no_grad():
        for k, solver in enumerate(solvers):
            q = solver.compute_q_values(
                flows[k], L_total, policies[k], theta_list[k]
            )
            pi   = policies[k]
            ent  = -torch.sum(pi * torch.log(pi + 1e-9), dim=-1)
            V_pi = torch.sum(pi * q, dim=-1) + solver.tau * ent
            V_br = solver.tau * torch.logsumexp(q / solver.tau, dim=-1)
            sr, sc = solver.group["source"]
            results[k] = (V_br[0, sr, sc] - V_pi[0, sr, sc]).item()
    return results

def plot_losses(loss_dict, title="Leader Loss Over Training", xlabel="Training Step", ylabel="Loss"):
    plt.figure(figsize=(8, 6))
    for label, losses in loss_dict.items():
        plt.plot(losses, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()