import numpy as np
import matplotlib.pyplot as plt


# -------------------------------------------------
# Generador de N(mu, var) usando Box-Muller
# -------------------------------------------------
def normal_box_muller(mu, var, n=1):
    u1 = np.random.uniform(0.0, 1.0, size=n)
    u2 = np.random.uniform(0.0, 1.0, size=n)

    z = np.cos(2 * np.pi * u1) * np.sqrt(-2 * np.log(u2))
    x = mu + np.sqrt(var) * z

    if n == 1:
        return x[0]
    return x


# -------------------------------------------------
# Normalización de ángulo a [-pi, pi]
# -------------------------------------------------
def normalizar_angulo(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi


# -------------------------------------------------
# Modelo de movimiento basado en odometría
# -------------------------------------------------
def modelo_movimiento_odometria(x_t, u_t, alpha):
    """
    x_t = [x, y, theta]
    u_t = [delta_r1, delta_r2, delta_t]
    alpha = [alpha1, alpha2, alpha3, alpha4]

    Devuelve:
    x_t1 = [x', y', theta']
    """
    x, y, theta = x_t
    delta_r1, delta_r2, delta_t = u_t
    alpha1, alpha2, alpha3, alpha4 = alpha

    # Varianzas del ruido
    var_r1 = alpha1 * (delta_r1 ** 2) + alpha2 * (delta_t ** 2)
    var_t  = alpha3 * (delta_t ** 2) + alpha4 * (delta_r1 ** 2 + delta_r2 ** 2)
    var_r2 = alpha1 * (delta_r2 ** 2) + alpha2 * (delta_t ** 2)

    # Muestreamos ruido gaussiano
    ruido_r1 = normal_box_muller(0, var_r1, 1)
    ruido_t  = normal_box_muller(0, var_t, 1)
    ruido_r2 = normal_box_muller(0, var_r2, 1)

    # Aplicamos ruido
    delta_r1_hat = delta_r1 - ruido_r1
    delta_t_hat  = delta_t  - ruido_t
    delta_r2_hat = delta_r2 - ruido_r2

    # Nueva pose
    x_nuevo = x + delta_t_hat * np.cos(theta + delta_r1_hat)
    y_nuevo = y + delta_t_hat * np.sin(theta + delta_r1_hat)
    theta_nuevo = theta + delta_r1_hat + delta_r2_hat
    theta_nuevo = normalizar_angulo(theta_nuevo)

    return np.array([x_nuevo, y_nuevo, theta_nuevo])


# -------------------------------------------------
# Generar 5000 muestras
# -------------------------------------------------
def generar_muestras_odometria(N=5000):
    x_t = np.array([2.0, 4.0, 0.0])
    u_t = np.array([np.pi / 2, 0.0, 1.0])
    alpha = np.array([0.1, 0.1, 0.01, 0.01])

    muestras = np.zeros((N, 3))

    for i in range(N):
        muestras[i] = modelo_movimiento_odometria(x_t, u_t, alpha)

    return muestras


# -------------------------------------------------
# Main
# -------------------------------------------------
if __name__ == "__main__":
    muestras = generar_muestras_odometria(5000)

    x_vals = muestras[:, 0]
    y_vals = muestras[:, 1]

    plt.figure(figsize=(8, 8))
    plt.scatter(x_vals, y_vals, s=8, alpha=0.5)
    plt.xlabel("x(t+1)")
    plt.ylabel("y(t+1)")
    plt.title("5000 muestras del modelo de movimiento basado en odometría")
    plt.grid(True)
    plt.axis("equal")
    plt.scatter(2, 5, marker='x', s=100, label='Posición ideal')
    plt.legend()
    plt.show()