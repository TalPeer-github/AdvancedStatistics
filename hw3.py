import numpy as np
import pandas as pd
import scipy.stats as st


def confidence_interval(new_x):
    m = new_x.mean()
    s = st.sem(new_x)
    n = new_x.shape[0]
    confidence = 0.95
    ci = st.t.interval(confidence, n, loc=m, scale=s)
    return ci


def create_F(data: pd.DataFrame, b_estimator, X, y_true):
    y_estimator = np.matmul(X, b_estimator)
    y_mean = np.mean(y_true)
    print(f"y_true : {y_true}")
    print(f"y_estimator : {y_estimator}")
    print(f"y_mean : {y_mean}")

    ssr = np.linalg.norm(y_estimator - y_mean) ** 2
    sse = np.linalg.norm(y_true - y_estimator) ** 2
    sst = np.linalg.norm(y_true - y_mean) ** 2
    p = X.shape[1]
    k = p - 1
    n = X.shape[0]
    ssr_df = k
    sse_df = n - p
    sst_df = n - 1
    MSr = ssr / ssr_df
    MSe = sse / sse_df
    MSt = sst / sst_df
    F_test = MSr / MSe
    alpha = 0.05
    rejection_area = st.f.cdf(alpha, ssr_df, sse_df)
    reject_H0 = F_test >= rejection_area
    Pvalue = 1 - st.f.cdf(F_test, ssr_df, sse_df)
    R_squere = ssr / sst
    R_squere_adg = 1 - (MSe / MSt)
    print(f"SSr : {ssr}")
    print(f"SSe : {sse}")
    print(f"SSt : {sst}")
    print(f"F_test : {F_test}")
    print(f"R_squere : {R_squere}")
    print(f"R_squere_adg : {R_squere_adg}")
    return ssr, sse, sst


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    alpha = 0.05
    data = pd.read_csv('ex3.csv')
    y_true = data['y'].to_numpy()
    n = data.shape[0]
    data['x0'] = np.ones((n))
    X = data.loc[:, ['x0', 'x2', 'x3', 'x4', 'x5']].to_numpy()
    num_of_coeff = X.shape[1]
    b = np.random.uniform(0.0, 1.0, size=num_of_coeff)
    C = np.linalg.inv(np.matmul(X.T, X))
    cov = np.matmul(C, X.T)
    b_estimator = np.matmul(cov, y_true)

    P = np.matmul(X, cov)
    I = np.eye(n)
    noise_estimator = np.matmul((I - P), y_true)
    noise_var_estimator = (1 / (n - num_of_coeff)) * (np.linalg.norm(noise_estimator) ** 2)

    ssr_df = num_of_coeff
    sse_df = n - num_of_coeff
    sst_df = n - 1
    ssr, sse, sst = create_F(data, b_estimator, X, y_true)
    new_x = np.array([1, 20, 30, 90, 2])
    print(f"New x : {new_x}")
    estimated_y = np.dot(new_x, b_estimator)
    print(f"Estimated New y : {estimated_y}")
    se_estimated_y = np.sqrt(noise_var_estimator * np.matmul(new_x, np.matmul(C, new_x.T)))
    se_estimated_y_normal = sse * np.sqrt(np.matmul(new_x, np.matmul(C, new_x.T)))
    z_crit = st.norm.pdf(1 - (alpha / 2)) * se_estimated_y
    t_crit = st.t.pdf(1 - (alpha / 2), sse_df) * se_estimated_y_normal
    ci_expectation = [estimated_y - z_crit, estimated_y + z_crit]
    ci_observation = [estimated_y - t_crit, estimated_y + t_crit]
    print(f"ci_expectation : {ci_expectation}")
    print(f"ci_observation : {ci_observation}")
    print("F test are managed to check if liner regression is actually necessary.")
