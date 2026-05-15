import pandas as pd, numpy as np, datetime as dt
from scipy.special import ndtr

class Model():

    def __init__(self, lmbd: float = 0.94, ewma_conf: float = 0.9999):
        self.assets, self.df_prices, self.df_returns = self.get_data()
        self.lmbd = lmbd
        self.ewma_conf = ewma_conf
        self.lmbd_matrix, self.lmbd_serie, self.lmbd_serie_len = self.get_ewma_matrix()

    def get_data(self):
        df_prices = pd.read_csv('prices_series.csv').dropna()
        df_prices['Date'] = df_prices['Date'].apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d").date())
        df_prices = df_prices.set_index('Date')
        df_returns = (df_prices/df_prices.shift(1)-1).iloc[1:]
        assets = df_returns.columns.tolist()
        return assets, df_prices, df_returns

    def get_ewma_matrix(self):
        serie_len = int(np.ceil(np.log(1-self.ewma_conf)/np.log(self.lmbd)))
        lmbd_serie = (1-self.lmbd) * np.array([self.lmbd**i for i in range(serie_len)][::-1])
        return np.diag(lmbd_serie), lmbd_serie, serie_len

    def get_ewma_vol(self, returns: np.ndarray) -> np.ndarray:
        n = self.lmbd_serie_len
        windows = np.lib.stride_tricks.sliding_window_view(returns, n, axis=0)
        return np.sqrt((windows ** 2) @ self.lmbd_serie)

    def _copula_sample(self, sorted_marginals: np.ndarray, chol: np.ndarray) -> np.ndarray:
        n, n_assets = sorted_marginals.shape
        z = chol @ np.random.randn(n_assets)
        u = ndtr(z)
        idx = np.clip((u * n).astype(int), 0, n - 1)
        return sorted_marginals[idx, np.arange(n_assets)]

    def generate_samples(self, T: int = 10, number_of_simulations: int = 100):
        returns = self.df_returns.copy().values
        vol_ewma = self.get_ewma_vol(returns=returns)
        vol_adj_returns = returns[-vol_ewma[:-1].shape[0]:] / vol_ewma[:-1]

        n_assets = len(self.assets)
        all_paths = np.zeros((number_of_simulations, T, n_assets))

        # Vol state: EWMA covariance of raw returns
        r_window = returns[-self.lmbd_serie_len:]
        cov_init = r_window.T @ self.lmbd_matrix @ r_window

        # Correlation state: EWMA outer product of standardized residuals
        eps_window = vol_adj_returns[-self.lmbd_serie_len:]
        Q_init = eps_window.T @ self.lmbd_matrix @ eps_window

        # Sorted Historical Distribution 
        sorted_vol_adj = np.sort(vol_adj_returns, axis=0)

        for simul in range(number_of_simulations):
            cov_now = cov_init.copy()
            Q_now = Q_init.copy()

            for t in range(T):
                std = np.sqrt(np.diag(cov_now))

                # Correlation from standardized residuals, renormalized to a proper corr matrix
                std_Q = np.sqrt(np.diag(Q_now))
                corr_now = Q_now / np.outer(std_Q, std_Q)
                corr_now += 1e-8 * np.eye(n_assets)
                chol = np.linalg.cholesky(corr_now)

                # Distribution of returns adjusted to current volatility
                distribution_returns_now = sorted_vol_adj * std
                sampled = self._copula_sample(distribution_returns_now, chol)
                all_paths[simul, t] = sampled

                # Vol and correlation updated separately with their respective residuals
                shock = sampled / std
                cov_now = self.lmbd * cov_now + (1 - self.lmbd) * np.outer(sampled, sampled)
                Q_now = self.lmbd * Q_now + (1 - self.lmbd) * np.outer(shock, shock)

        return all_paths.transpose(1, 2, 0)