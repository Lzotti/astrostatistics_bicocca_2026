import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
from astroML.datasets import generate_mu_z


z_sample, mu_sample, dmu = generate_mu_z(100, random_state=1234)

plt.figure()
plt.errorbar(z_sample, mu_sample, dmu, fmt='.k', ecolor='gray', lw=1,label='data')
plt.xlabel("z")
plt.ylabel("$\mu$")
plt.legend(loc='lower right')
plt.xlim(0,2)
plt.ylim(35,50)

# plt.show()

z_sample = z_sample[:,np.newaxis]
x_grid = np.linspace(0, 2, 1000)[:,np.newaxis]

plt.figure()

kernel = RBF(1.0, (1e-2, 1e2))

gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=dmu**2, normalize_y=True)

gp.fit(z_sample, mu_sample)
mu_pred, dmu_pred = gp.predict(x_grid, return_std=True)
plt.errorbar(z_sample, mu_sample, dmu, fmt='.k', ecolor='gray', lw=1,label='data')
plt.plot(x_grid, mu_pred, 'r-', lw=1, label='GP prediction')

plt.fill_between(x_grid[:, 0], mu_pred - dmu_pred, mu_pred + dmu_pred, alpha=0.3, color='r', label='68% confidence interval')

plt.fill_between(x_grid[:,0], mu_pred - 1.96*dmu_pred, mu_pred + 1.96*dmu_pred, alpha=0.2, color='r', label='95% confidence interval')
plt.xlabel("z")
plt.ylabel("$\mu$")
plt.legend(loc='lower right')
plt.xlim(0,2)
plt.ylim(35,50)
# plt.show()

plt.figure()


kernel = C(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-2, 1e2))

gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=dmu**2, normalize_y=True)

gp.fit(z_sample, mu_sample)
mu_pred, dmu_pred = gp.predict(x_grid, return_std=True)
plt.errorbar(z_sample, mu_sample, dmu, fmt='.k', ecolor='gray', lw=1,label='data')
plt.plot(x_grid, mu_pred, 'r-', lw=1, label='GP prediction')

plt.fill_between(x_grid[:, 0], mu_pred - dmu_pred, mu_pred + dmu_pred, alpha=0.3, color='r', label='68% confidence interval')
plt.fill_between(x_grid[:,0], mu_pred - 1.96*dmu_pred, mu_pred + 1.96*dmu_pred, alpha=0.2, color='r', label='95% confidence interval')
plt.xlabel("z")
plt.ylabel("$\mu$")
plt.legend(loc='lower right')
plt.xlim(0,2)
plt.ylim(35,50)
# plt.show()

plt.figure()


kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)

gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=dmu**2, normalize_y=True)
gp.fit(z_sample, mu_sample)
mu_pred, dmu_pred = gp.predict(x_grid, return_std=True)
plt.errorbar(z_sample, mu_sample, dmu, fmt='.k', ecolor='gray', lw=1,label='data')
plt.plot(x_grid, mu_pred, 'r-', lw=1, label='GP prediction')

plt.fill_between(x_grid[:, 0], mu_pred - dmu_pred, mu_pred + dmu_pred, alpha=0.3, color='r', label='68% confidence interval')
plt.fill_between(x_grid[:,0], mu_pred - 1.96*dmu_pred, mu_pred + 1.96*dmu_pred, alpha=0.2, color='r', label='95% confidence interval')
plt.xlabel("z")
plt.ylabel("$\mu$")
plt.legend(loc='lower right')
plt.xlim(0,2)
plt.ylim(35,50)
# plt.show()

plt.figure()


kernel = C(1.0, (1e-2, 1e2)) * (Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5) + RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=dmu**2, normalize_y=True)
gp.fit(z_sample, mu_sample)
mu_pred, dmu_pred = gp.predict(x_grid, return_std=True)
plt.errorbar(z_sample, mu_sample, dmu, fmt='.k', ecolor='gray', lw=1,label='data')
plt.plot(x_grid, mu_pred, 'r-', lw=1, label='GP prediction')

plt.fill_between(x_grid[:, 0], mu_pred - dmu_pred, mu_pred + dmu_pred, alpha=0.3, color='r', label='68% confidence interval')

plt.fill_between(x_grid[:,0], mu_pred - 1.96*dmu_pred, mu_pred + 1.96*dmu_pred, alpha=0.2, color='r', label='95% confidence interval')
plt.xlabel("z")
plt.ylabel("$\mu$")
plt.legend(loc='lower right')
plt.xlim(0,2)
plt.ylim(35,50)
# plt.show()
plt.close('all')







kernel_options = [
    C(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)),
    C(1.0, (1e-2, 1e2)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5),
    C(1.0, (1e-2, 1e2)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5),
    C(1.0, (1e-2, 1e2)) * (Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5) + RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)))
]

param_grid = {
    'kernel': kernel_options,
    'normalize_y': [True, False],
}

k = 10
kf = KFold(n_splits=k, shuffle=True, random_state=1234)

gp = GaussianProcessRegressor(n_restarts_optimizer=10)

scorer = make_scorer(mean_squared_error, greater_is_better=True)

grid_search = GridSearchCV(estimator=gp, param_grid=param_grid, cv=kf, scoring=scorer, n_jobs=-1)
grid_search.fit(z_sample, mu_sample)
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

print("Miglior kernel:", best_params['kernel'])
print("Normalize_y:", best_params['normalize_y'])
print("Miglior punteggio R^2:", grid_search.best_score_)

mu_pred, dmu_pred = best_model.predict(x_grid, return_std=True)

plt.figure()

plt.errorbar(z_sample, mu_sample, dmu, fmt='.k', ecolor='gray', lw=1, label='data')
plt.plot(x_grid, mu_pred, 'r-', lw=1, label='GP prediction')
plt.fill_between(x_grid[:, 0], mu_pred - dmu_pred, mu_pred + dmu_pred, alpha=0.3, color='r', label='68% confidence interval')
plt.fill_between(x_grid[:, 0], mu_pred - 1.96 * dmu_pred, mu_pred + 1.96 * dmu_pred, alpha=0.2, color='r', label='95% confidence interval')
plt.xlabel("z")
plt.ylabel("$\mu$")
plt.legend(loc='lower right')
plt.xlim(0, 2)
plt.ylim(35, 50)
plt.show()


#HO del supremo overfitting, sistemo un po' e uso con un senso la cross validation