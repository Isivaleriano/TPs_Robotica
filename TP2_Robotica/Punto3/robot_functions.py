import numpy as np

class RobotFunctions:

    def __init__(self):
        pass

    def _sample_normal(self, variance):
        """
        Genera una muestra de N(0, variance) usando Box-Muller.
        """
        u1 = np.random.uniform(0.0, 1.0)
        u2 = np.random.uniform(0.0, 1.0)
        z = np.cos(2 * np.pi * u1) * np.sqrt(-2 * np.log(u2))
        return np.sqrt(variance) * z

    def odometry_motion_model(self, xt, ut, alpha):
        """
        Modelo de movimiento basado en odometría con ruido gaussiano.
        """
        x, y, theta = xt
        delta_r1, delta_t, delta_r2 = ut
        alpha1, alpha2, alpha3, alpha4 = alpha

        var_r1 = alpha1 * (delta_r1 ** 2) + alpha2 * (delta_t ** 2)
        var_t  = alpha3 * (delta_t  ** 2) + alpha4 * (delta_r1 ** 2 + delta_r2 ** 2)
        var_r2 = alpha1 * (delta_r2 ** 2) + alpha2 * (delta_t ** 2)

        ruido_r1 = self._sample_normal(var_r1)
        ruido_t  = self._sample_normal(var_t)
        ruido_r2 = self._sample_normal(var_r2)

        delta_r1_hat = delta_r1 - ruido_r1
        delta_t_hat  = delta_t  - ruido_t
        delta_r2_hat = delta_r2 - ruido_r2

        x_new     = x + delta_t_hat * np.cos(theta + delta_r1_hat)
        y_new     = y + delta_t_hat * np.sin(theta + delta_r1_hat)
        theta_new = theta + delta_r1_hat + delta_r2_hat

        theta_new = np.arctan2(np.sin(theta_new), np.cos(theta_new))

        return np.array([x_new, y_new, theta_new])