from numpy.linalg import pinv
import numpy as np
import cvxpy as cp
import pandas as pd

from tqdm import tqdm
from typing import Literal


# CHECKED
def mv_opt(mu: np.ndarray, Sigma: np.ndarray,
           R_target: float = None) -> np.ndarray:
    # solve the mean-variance optimization problem
    p = len(mu)
    w = cp.Variable(p)
    objective = cp.Minimize(cp.quad_form(w, Sigma))
    constraints = [cp.sum(w) == 1]
    if R_target is not None:
        constraints.append(mu @ w == R_target)
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL)
    # prob.solve(solver=cp.ECOS, verbose=True)
    if prob.status != cp.OPTIMAL:
        raise ValueError(
            "The optimization problem did not solve to optimality. | mv_opt | Status: {}".format(prob.status))
    return w.value


# CHECKED
def compute_alpha(X: np.ndarray) -> np.ndarray:
    # Compute the alpha vector (Rank-1 approximation of mv-PBR)
    n, p = X.shape
    X_centered = X - X.mean(axis=0)
    alpha = np.zeros(p)

    for i in range(p):
        x_i = X_centered[:, i]
        mu4 = np.mean(x_i**4)
        sigma2 = np.var(x_i, ddof=1)
        alpha[i] = (
            (1/n) * mu4 - ((n - 3)/(n*(n - 1))) * (sigma2**2)
        ) ** 0.25

    return alpha


# CHECKED
def compute_Q2(X: np.ndarray) -> np.ndarray:
    # Compute Q2 used in (Best convex quadratic approximation of (mv-PBR))
    n, p = X.shape
    X_centered = X - np.mean(X, axis=0)

    Sigma = (X_centered.T @ X_centered) / (n - 1)

    Q2 = np.zeros((p, p))
    for i in range(p):
        Xi = X_centered[:, i]
        for j in range(p):
            Xj = X_centered[:, j]

            mu4_ijij = np.mean((Xi * Xj) ** 2)

            sigma_ij_sq = Sigma[i, j] ** 2
            sigma_ii = Sigma[i, i]
            sigma_jj = Sigma[j, j]

            Q2[i, j] = (
                (mu4_ijij - sigma_ij_sq) / n
                + (sigma_ii * sigma_jj + sigma_ij_sq) / (n * (n - 1))
            )

    return Q2


# CHECKED
def compute_A_star(Q2: np.ndarray) -> np.ndarray:
    # Compute positive semi-definite matrix A_star (Best convex quadratic approximation of (mv-PBR))
    shrink_factor = 1e-10
    Q2 = Q2 / shrink_factor  # scale the matrix to avoid numerical issues
    p = Q2.shape[0]
    A = cp.Variable((p, p), PSD=True)
    objective = cp.Minimize(cp.norm(A - Q2, 'fro'))
    prob = cp.Problem(objective)
    # prob.solve(solver=cp.ECOS, verbose=True)
    prob.solve(solver=cp.CLARABEL)
    if prob.status != cp.OPTIMAL:
        raise ValueError(
            "The optimization problem did not solve to optimality. | Status: {}".format(prob.status))
    return A.value * shrink_factor


# CHECKED
def mv_pbr1_opt(X: np.ndarray, mu: np.ndarray, Sigma: np.ndarray,
                U: float, R_target: float = None, direct: bool = True) -> float:
    # solve mv-PBR-1 optimization problem
    # prepare data
    n, p = X.shape
    alpha = compute_alpha(X)

    # solve the mv-PBR-1 optimization problem
    w = cp.Variable(p)
    constraints = [cp.sum(w) == 1]
    if R_target is not None:
        constraints.append(mu @ w == R_target)
    constraints.append(w.T @ alpha <= U ** 0.25)
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma)), constraints)
    prob.solve(solver=cp.CLARABEL)
    # prob.solve(solver=cp.ECOS, verbose=True)
    if prob.status != cp.OPTIMAL:
        raise ValueError(
            "The optimization problem did not solve to optimality. | mv_pbr1_opt | Status: {}".format(prob.status))

    # return optimal weights if direct is True
    if direct:
        return w.value

    # otherwise, compute the optimal weights using the Lagrange multiplier
    one_vec = np.ones(p)
    Sigma_inv = np.linalg.inv(Sigma)
    lambda_star = constraints[-1].dual_value    # optimal lagrange multiplier
    w_mv = mv_opt(mu, Sigma, R_target)

    if R_target is not None:
        beta1_numerator = (alpha @ Sigma_inv @ mu) * (mu @ Sigma_inv @
                                                      one_vec) - (alpha @ Sigma_inv @ one_vec) * (mu @ Sigma_inv @ mu)
        beta1_denominator = (one_vec @ Sigma_inv @ one_vec) * \
            (mu @ Sigma_inv @ mu) - \
            (mu @ Sigma_inv @ one_vec) ** 2
        beta1 = beta1_numerator / beta1_denominator

        beta2_numerator = (alpha @ Sigma_inv @ mu) * (one_vec @ Sigma_inv @
                                                      one_vec) - (alpha @ Sigma_inv @ one_vec) * (one_vec @ Sigma_inv @ mu)
        beta2_denominator = beta1_denominator
        beta2 = beta2_numerator / beta2_denominator

        w = w_mv - 0.5 * lambda_star * \
            (Sigma_inv @ (beta1 * one_vec + beta2 * mu + alpha))

    else:
        beta = - (one_vec @ Sigma_inv @ alpha) / \
            (one_vec @ Sigma_inv @ one_vec)
        w = w_mv - 0.5 * lambda_star * \
            (Sigma_inv @ (beta * one_vec + alpha))

    return w


# CHECKED
def mv_pbr2_opt(X: np.ndarray, mu: np.ndarray, Sigma: np.ndarray,
                U: float, R_target: float = None, direct: bool = True) -> np.ndarray:
    # solve the mv-PBR-2 optimization problem
    # prepare data
    n, p = X.shape
    Q2 = compute_Q2(X)
    A_star = compute_A_star(Q2)

    # solve the mv-PBR-2 optimization problem
    w = cp.Variable(p)
    constraints = [cp.sum(w) == 1]
    if R_target is not None:
        constraints.append(mu @ w == R_target)
    constraints.append(cp.quad_form(w, A_star) <= np.sqrt(U))
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma)), constraints)
    prob.solve(solver=cp.CLARABEL)
    # prob.solve(solver=cp.ECOS, verbose=True)
    if prob.status != cp.OPTIMAL:
        raise ValueError(
            "The optimization problem did not solve to optimality. | mv_pbr2_opt | Status: {}".format(prob.status))

    # return optimal weights if direct is True
    if direct:
        return w.value

    # otherwise, compute the optimal weights using the Lagrange multiplier
    one_vec = np.ones(p)
    lambda_star = constraints[-1].dual_value
    # optimal Lagrange multiplier for quadratic constraint
    Sigma_tilde = Sigma + lambda_star * A_star
    Sigma_tilde_inv = np.linalg.inv(Sigma_tilde)

    if R_target is not None:
        v1_numerator = 2 * (R_target * (mu @ Sigma_tilde_inv @
                            one_vec) - (mu @ Sigma_tilde_inv @ mu))
        v1_denominator = (one_vec @ Sigma_tilde_inv @ one_vec) * (mu @
                                                                  Sigma_tilde_inv @ mu) - (mu @ Sigma_tilde_inv @ one_vec) ** 2

        v2_numerator = 2 * (-R_target * (one_vec @ Sigma_tilde_inv @
                            one_vec) + (mu @ Sigma_tilde_inv @ one_vec))
        v2_denominator = v1_denominator

        v1 = v1_numerator / v1_denominator
        v2 = v2_numerator / v2_denominator

        w = -0.5 * Sigma_tilde_inv @ (v1 * one_vec + v2 * mu)

    else:
        w = Sigma_tilde_inv @ one_vec / (one_vec @ Sigma_tilde_inv @ one_vec)

    return w


# CHECKED
def cvar_opt(X: np.ndarray, beta: float, R_target: float = None) -> tuple[np.ndarray, float]:
    # solve the CVaR optimization problem

    n, p = X.shape
    w = cp.Variable(p)
    alpha = cp.Variable()
    losses = -X @ w  # portfolio losses
    hinge = cp.pos(losses - alpha)  # (loss - alpha)^+

    # CVaR objective
    cvar = alpha + (1 / (n * (1 - beta))) * cp.sum(hinge)
    objective = cp.Minimize(cvar)

    # Constraints

    constraints = [cp.sum(w) == 1]
    mu_hat = X.mean(axis=0)
    if R_target is not None:
        constraints.append(mu_hat @ w == R_target)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL)
    # prob.solve(solver=cp.ECOS, verbose=True)

    if prob.status != cp.OPTIMAL:
        raise ValueError(
            "The optimization problem did not solve to optimality. | cvar_opt | Status: {}".format(prob.status))

    return w.value, alpha.value


# CHECKED
def cvar_pbr_opt(X: np.ndarray, beta: float, U1: float, U2: float = None,
                 R_target: float = None) -> tuple[np.ndarray, float]:
    # PBR for Mean-CVaR Portfolio Optimization
    n, p = X.shape
    w = cp.Variable(p)
    alpha = cp.Variable()
    z = cp.Variable(n)

    # Sample mean and covariance
    mu_hat = X.mean(axis=0)
    Sigma_hat = np.cov(X.T, ddof=1)  # shape (p, p)

    # Construct Omega_n = (1/(n-1)) * (I - (1/n) * 1 1^T)
    I_n = np.eye(n)
    one_n = np.ones((n, 1))
    Omega = (1 / (n - 1)) * (I_n - (1 / n) * (one_n @ one_n.T))

    # Loss vector
    losses = -X @ w
    constraints = [
        cp.sum(w) == 1,
        z >= losses - alpha,
        z >= 0,
        (1 / (n * (1 - beta)**2)) * cp.quad_form(z, Omega) <= U1,
    ]
    if R_target is not None:
        constraints.append(mu_hat @ w == R_target)
    if U2 is not None:
        constraints.append((1 / n) * cp.quad_form(w, Sigma_hat) <= U2)

    # Objective: minimize alpha + 1 / (n(1 - beta)) sum z_i
    objective = cp.Minimize(alpha + (1 / (n * (1 - beta))) * cp.sum(z))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL)
    # prob.solve(solver=cp.ECOS, verbose=True)
    if prob.status != cp.OPTIMAL:
        raise ValueError(
            "The optimization problem did not solve to optimality. | cvar_pbr_opt | Status: {}".format(prob.status))

    return w.value, alpha.value


# CHECKED
def cvar_relax_opt(X: np.ndarray, beta: float, U1: float,
                   U2: float = None, R_target: float = None) -> tuple[np.ndarray, float]:
    # PBR for Mean-CVaR Portfolio Optimization with relaxation
    n, p = X.shape
    w = cp.Variable(p)
    alpha = cp.Variable()
    z = cp.Variable(n)

    # Sample stats
    mu_hat = X.mean(axis=0)
    Sigma_hat = np.cov(X.T, ddof=1)

    # Omega matrix
    I_n = np.eye(n)
    one_n = np.ones((n, 1))
    Omega = (1 / (n - 1)) * (I_n - (1 / n) * one_n @ one_n.T)

    # Losses
    losses = -X @ w

    # Constraints
    constraints = [
        cp.sum(w) == 1,
        z >= 0,
        z >= losses - alpha,
        (1 / (n * (1 - beta)**2)) * cp.quad_form(z, Omega) <= U1
    ]
    if R_target is not None:
        constraints.append(mu_hat @ w == R_target)
    if U2 is not None:
        constraints.append((1 / n) * cp.quad_form(w, Sigma_hat) <= U2)

    # Objective
    objective = cp.Minimize(alpha + (1 / (n * (1 - beta))) * cp.sum(z))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL)
    # prob.solve(solver=cp.ECOS, verbose=True)
    if prob.status != cp.OPTIMAL:
        raise ValueError(
            "The optimization problem did not solve to optimality. | cvar_relax_opt | Status: {}".format(prob.status))

    return w.value, alpha.value


def cvar_robust_opt(X: np.ndarray, beta: float, U1: float, U2: float = None, R_target: float = None) -> tuple[np.ndarray, float]:
    print("cvar_robust_opt")
    # Robust CVaR-PBR (Proposition 6)
    n, p = X.shape
    w = cp.Variable(p)
    alpha = cp.Variable()
    z = cp.Variable(n)
    u = cp.Variable(n)  # For U1 ellipsoid dual constraint
    mu_tilde = cp.Variable(p)  # For U2 ellipsoid dual constraint

    # Sample stats
    mu_hat = X.mean(axis=0)
    Sigma_hat = np.cov(X.T, ddof=1)
    Sigma_inv = np.linalg.pinv(Sigma_hat)

    # Omega and its pseudoinverse
    I_n = np.eye(n)
    one_n = np.ones((n, 1))
    Omega = (1 / (n - 1)) * (I_n - (1 / n) * one_n @ one_n.T)
    Omega_pinv = pinv(Omega)

    losses = -X @ w

    # Constraints
    constraints = [
        cp.sum(w) == 1,
        z >= 0,
        z >= losses - alpha,

        # Dual constraint for U1: max_u z^T u s.t. u^T Omega^+ u <= 1, 1^T u = 0
        cp.quad_form(u, Omega_pinv) <= 1,
        cp.sum(u) == 0,
        z @ u <= cp.sqrt(U1),
    ]

    # Mean constraint if R is given
    if R_target is not None:
        constraints.append(mu_hat @ w == R_target)

    # Dual constraint for U2: max_mu w^T(mu - mu_hat) s.t. (mu - mu_hat)^T Sigma^-1 (mu - mu_hat) <= 1
    if U2 is not None:
        delta_mu = mu_tilde - mu_hat
        constraints += [
            cp.quad_form(delta_mu, Sigma_inv) <= 1,
            w @ delta_mu <= cp.sqrt(U2)
        ]

    # Objective
    objective = cp.Minimize(alpha + (1 / (n * (1 - beta))) * cp.sum(z))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL)

    if prob.status != cp.OPTIMAL:
        raise ValueError(
            f"Problem not solved to optimality: {prob.status}"
        )

    return w.value, alpha.value

# CHECKED


def u_min_mv1(X: np.ndarray, mu: np.ndarray, Sigma: np.ndarray,
              R_target: float = None) -> float:
    # # lower bound for mv-PBR-1
    # n, p = X.shape

    # # Assume you provide a function compute_alpha that returns a (p,) vector
    # alpha = compute_alpha(X)  # alpha should be shape (p,)
    # w = cp.Variable(p)

    # # Constraints
    # constraints = [cp.sum(w) == 1]
    # if R_target is not None:
    #     constraints.append(mu @ w == R_target)

    # # Objective: minimize w^T alpha
    # objective = cp.Minimize(alpha @ w)
    # prob = cp.Problem(objective, constraints)
    # prob.solve(solver=cp.CLARABEL)
    # # prob.solve(solver=cp.ECOS, verbose=True)

    # if prob.status != cp.OPTIMAL:
    #     raise ValueError(
    #         "Optimization did not solve to optimality. | u_min_mv1 | Status: {}".format(prob.status))

    # return prob.value
    return 1e-10


# CHECKED
def u_min_mv2(X: np.ndarray, mu: np.ndarray, Sigma: np.ndarray,
              R_target: float = None) -> float:
    # lower bound for mv-PBR-2
    n, p = X.shape

    # Assume you provide a function compute_Q2 that returns a (p, p) matrix
    Q2 = compute_Q2(X)  # Q2 should be shape (p, p)
    A_star = compute_A_star(Q2)

    w = cp.Variable(p)

    # Constraints
    constraints = [cp.sum(w) == 1]
    if R_target is not None:
        constraints.append(mu @ w == R_target)

    # Objective: minimize w^T mu
    objective = cp.Minimize(cp.quad_form(w, A_star))
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.CLARABEL)
    except cp.SolverError as e:
        print(f"SolverError: {e}")
        return 0
    # prob.solve(solver=cp.ECOS, verbose=True)

    if prob.status != cp.OPTIMAL:
        raise ValueError(
            "Optimization did not solve to optimality. | u_min_v2 | Status: {}".format(prob.status))

    return prob.value


def u_min_cvar(X: np.ndarray, beta: float, R_target: float = None) -> np.ndarray:
    # lower bound for cv-PBR
    n, p = X.shape

    # Omega matrix
    I_n = np.eye(n)
    one_n = np.ones((n, 1))
    Omega = (1 / (n - 1)) * (I_n - (1 / n) * one_n @ one_n.T)

    w = cp.Variable(p)
    z = cp.Variable(n)
    alpha = cp.Variable()
    losses = -X @ w  # portfolio losses
    hinge = cp.pos(losses - alpha)  # (loss - alpha)^+

    objective = cp.Minimize(cp.quad_form(z, Omega))

    # Constraints
    constraints = [cp.sum(w) == 1]
    mu_hat = X.mean(axis=0)
    if R_target is not None:
        constraints.append(mu_hat @ w == R_target)
    constraints.append(z >= losses - alpha)
    constraints.append(z >= 0)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL)

    # prob.solve(solver=cp.ECOS, verbose=True)

    if prob.status != cp.OPTIMAL:
        raise ValueError(
            "The optimization problem did not solve to optimality. | u_min_cvar |Status: {}".format(prob.status))

    return prob.value


class PortOpt:
    def __init__(self, X: np.ndarray, R_target: float = None):
        self.X = X.astype(float)
        self.R_target = R_target
        # check if X is a 2D array
        if self.X.ndim != 2:
            raise ValueError("Input X must be a 2D array.")
        # check if X contains NaN or infinite values
        if not np.isfinite(self.X).all():
            raise ValueError("Input X contains NaN or infinite values.")

        self.n, self.p = self.X.shape
        self.mu = np.mean(self.X, axis=0)
        self.Sigma = np.cov(self.X, rowvar=False)
        self.Sigma_inv = np.linalg.inv(self.Sigma)

    def optimize(self, method: str = 'mv') -> np.ndarray:
        pass

    def mv_opt(self) -> np.ndarray:
        self.mv_w = mv_opt(self.mu, self.Sigma, self.R_target)
        return self.mv_w

    def compute_alpha(self) -> np.ndarray:
        self.alpha = compute_alpha(self.X)
        return self.alpha


pbr_func_dict = {
    'mv_pbr1': mv_pbr1_opt,
    'mv_pbr2': mv_pbr2_opt,
    'cvar_pbr': cvar_pbr_opt,
    'cvar_pbr_relax': cvar_relax_opt,
    'cvar_robust': cvar_robust_opt,
}


pbr_u_min_dict = {
    'mv_pbr1': u_min_mv1,
    'mv_pbr2': u_min_mv2,
    'cvar_pbr': u_min_cvar,
    'cvar_pbr_relax': u_min_cvar,
}


def oos_pbcv(pbr: Literal['mv_pbr1', 'mv_pbr2', 'cvar_pbr', 'cvar_pbr_relax'],
             X: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, beta: float = 0.95,
             alpha: float = 0.4, gamma: float = 0.9, div: int = 5, bit: float = 0.05,
             R_target: float = None, k: int = 4) -> np.ndarray:

    # divide up the data randomly into k equal bins
    n_train = X.shape[0]
    indices = np.arange(n_train)
    np.random.shuffle(indices)
    bins = np.array_split(indices, k)
    train_valid_pairs = []
    for i in range(k):
        train_indices = np.concatenate(bins[:i] + bins[i + 1:])
        valid_indices = bins[i]
        train_valid_pairs.append((train_indices, valid_indices))

    #
    u_star_list = []
    for i in range(k):
        train_indices, valid_indices = train_valid_pairs[i]
        X_train = X[train_indices]
        X_valid = X[valid_indices]

        mu_i = np.mean(X_train, axis=0)
        Sigma_i = np.cov(X_train, rowvar=False)
        # solve pbr on D_train to get w_train
        if pbr in ['mv_pbr1', 'mv_pbr2']:
            w_i_train = mv_opt(mu_i, Sigma_i, R_target)
            u_i_max = w_i_train @ Sigma_i @ w_i_train
        elif pbr in ['cvar_pbr', 'cvar_pbr_relax', 'cvar_robust']:
            w_i_train, alpha_ = cvar_opt(X_train, beta, R_target)
            losses_i = -X @ w_i_train
            hinge_i = np.maximum(losses_i - alpha_, 0)
            u_i_max = alpha_ + \
                (1 / (X.shape[0] * (1 - beta))) * np.sum(hinge_i)

        # tighten the bound of u_upper if t is always 1, return u_star when t < 1
        t = 1
        u_upper = u_lower = 0
        while t == 1:
            u_star, t = oos_pbsd(pbr, X_train, X_valid, w_i_train, u_lower,
                                 u_upper, beta, alpha, gamma, div, bit, R_target=R_target)
        u_star_list.append(u_star)
        print(
            f'\t\t{i+1} th fold: u_star = {u_star:.4e}')

    u_star_final = np.exp(np.mean(np.log(u_star_list)))  # geometric mean
    return u_star_final


# CHECKED
def sharpe(X: np.ndarray, w: np.ndarray) -> float:
    mu = X.mean(axis=0)
    Sigma = np.cov(X.T, ddof=1)
    numerator = w @ mu
    denominator = np.sqrt(w @ Sigma @ w)
    if denominator == 0:
        return 0.0
    return numerator / denominator if denominator != 0 else 0.0


# CHECKED
def sharpe_gradient(w: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    numerator = (w @ Sigma @ w) * mu - (w @ mu) * Sigma @ w
    denominator = (w @ Sigma @ w) ** 1.5
    gradient = numerator / denominator
    return gradient


# CHECKED
def oos_pbsd(pbr: Literal['mv_pbr1', 'mv_pbr2', 'cvar_pbr', 'cvar_pbr_relax'], X_train: np.ndarray, X_valid: np.ndarray, w: np.ndarray, u_min: float, u_max: float, beta: float, alpha: float,  gamma: float, div: int, bit: float, R_target: float = None) -> float:
    top_percent = 0.1

    u_list = []
    w_change_norm_list = []

    if pbr in ['mv_pbr1']:
        initial_w = mv_opt(X_train.mean(axis=0),
                           np.cov(X_train.T, ddof=1), R_target)
        for exp in range(-25, -10):
            u_list.extend([10**exp, 3*10**exp])
    elif pbr in ['mv_pbr2']:
        initial_w = mv_opt(X_train.mean(axis=0),
                           np.cov(X_train.T, ddof=1), R_target)
        for exp in range(-50, -30):
            u_list.extend([10**exp, 3*10**exp])
    elif pbr in ['cvar_pbr', 'cvar_pbr_relax', 'cvar_robust']:
        initial_w, _ = cvar_opt(X_train, beta, R_target)
        for exp in range(-50, -1):
            u_list.extend([10**exp, 3*10**exp])

    sharpe_list = []
    for i, u in enumerate(u_list):
        if pbr in ['mv_pbr1', 'mv_pbr2']:
            try:
                w = pbr_func_dict[pbr](X_train, X_train.mean(
                    axis=0), np.cov(X_train.T, ddof=1), u, R_target=R_target)
            except Exception as e:
                # print(f"Error in {pbr} optimization: {e}")
                w = initial_w
        elif pbr in ['cvar_pbr', 'cvar_pbr_relax', 'cvar_robust']:
            try:
                w, _ = pbr_func_dict[pbr](
                    X_train, beta, u, R_target=R_target)
            except Exception as e:
                # print(f"Error in {pbr} optimization: {e}")
                w = initial_w

        w_change = w - initial_w
        w_change_norm = np.linalg.norm(w_change)
        w_change_norm_list.append(w_change_norm)

        sharpe_value = sharpe(X_valid, w)
        sharpe_list.append(sharpe_value)

    top_percent_index = int(len(w_change_norm_list) * top_percent)
    sorted_indices_by_norm = np.argsort(
        w_change_norm_list)[-top_percent_index:]
    filtered_sharpe_list = [sharpe_list[i] for i in sorted_indices_by_norm]
    max_sharpe_index = sorted_indices_by_norm[np.argmax(filtered_sharpe_list)]
    u_star = u_list[max_sharpe_index]
    return u_star, 0


if __name__ == "__main__":
    # Example usage
    from data_fetch import ff5_df, ff10_df
    ff5_df = ff5_df.loc[ff5_df['Date'] >= pd.to_datetime('2023-05-01')]
    X = ff5_df.drop(columns=["Date"]).values.astype(float)
    U = 1e-50  # Upper bound for Svar constraint
    R_target = None  # No target return

    w_opt = mv_opt(np.mean(X, axis=0), np.cov(X, rowvar=False), R_target)
    print("Optimal weights (MV):", w_opt)

    w_pbr1 = mv_pbr1_opt(X, np.mean(X, axis=0),
                         np.cov(X, rowvar=False), U, R_target, direct=True)
    print("Optimal weights (PBR1):", w_pbr1)
    w_pbr1 = mv_pbr1_opt(X, np.mean(X, axis=0),
                         np.cov(X, rowvar=False), U, R_target, direct=False)
    print("Optimal weights (PBR1, Lagrange):", w_pbr1)

    w_pbr2 = mv_pbr2_opt(X, np.mean(X, axis=0),
                         np.cov(X, rowvar=False), U, R_target, direct=True)
    print("Optimal weights (PBR2):", w_pbr2)
    w_pbr2 = mv_pbr2_opt(X, np.mean(X, axis=0),
                         np.cov(X, rowvar=False), U, R_target, direct=False)
    print("Optimal weights (PBR2, Lagrange):", w_pbr2)

    w_cvar, alpha_cvar = cvar_opt(X, beta=0.95, R_target=R_target)
    print("Optimal weights (CVaR):", w_cvar)
    print("CVaR alpha:", alpha_cvar)

    # import time
    # # start_time = time.time()
    # # w_cvar_pbr, alpha_cvar_pbr = cvar_pbr_opt(
    # #     X, beta=0.95, U1=U, R_target=R_target)
    # # end_time = time.time()
    # # print("Time taken for CVaR PBR optimization:", end_time - start_time)
    # # print("Optimal weights (CVaR PBR):", w_cvar_pbr)
    # # print("CVaR PBR alpha:", alpha_cvar_pbr)

    w_cvar_relax, alpha_cvar_relax = cvar_relax_opt(
        X, beta=0.95, U1=U, R_target=R_target)
    print("Optimal weights (CVaR Relax):", w_cvar_relax)
    print("CVaR Relax alpha:", alpha_cvar_relax)

    # # u_min_mv1_val = u_min_mv1(X, np.mean(X, axis=0),
    # #                           np.cov(X, rowvar=False), R_target)
    # # print("Minimum U for MV1:", u_min_mv1_val)

    # u_min_mv2_val = u_min_mv2(X, np.mean(X, axis=0),
    #                           np.cov(X, rowvar=False), R_target)
    # print("Minimum U for MV2:", u_min_mv2_val)

    # start_time = time.time()
    # u_min_cvar_val = u_min_cvar(X, beta=0.95, R_target=R_target)
    # end_time = time.time()
    # print("Time taken for CVaR U minimization:", end_time - start_time)
    # print("Minimum U for CVaR:", u_min_cvar_val)

    # # Example usage of oos_pbcv
    # pbr = 'mv_pbr2'
    # u = oos_pbcv(pbr, X, np.mean(X, axis=0), np.cov(X, rowvar=False),
    #              beta=0.95, alpha=0.4, gamma=0.9, div=5, bit=0.05,
    #              R_target=R_target, k=4)
    # print(f'u: {u}')
