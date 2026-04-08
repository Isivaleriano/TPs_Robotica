import numpy as np
import timeit


def normal_12_uniformes(mu, var, n):
    sigma = np.sqrt(var)
    u = np.random.uniform(0.0, 1.0, size=(n, 12))
    z = np.sum(u, axis=1) - 6.0
    return mu + sigma * z


def normal_rechazo(mu, var, n):
    sigma = np.sqrt(var)
    L = 10.0
    M = 1.0 / np.sqrt(2.0 * np.pi)

    muestras = []
    total = 0

    while total < n:
        batch_size = max(10000, n)
        y = np.random.uniform(-L, L, size=batch_size)
        u = np.random.uniform(0.0, 1.0, size=batch_size)

        fy = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-(y**2) / 2.0)
        aceptadas = y[u <= fy / M]

        if len(aceptadas) > 0:
            muestras.append(aceptadas)
            total += len(aceptadas)

    muestras = np.concatenate(muestras)[:n]
    return mu + sigma * muestras


def normal_box_muller(mu, var, n):
    sigma = np.sqrt(var)
    u1 = np.random.uniform(0.0, 1.0, size=n)
    u2 = np.random.uniform(0.0, 1.0, size=n)

    z = np.cos(2 * np.pi * u1) * np.sqrt(-2 * np.log(u2))
    return mu + sigma * z


def comparar_tiempos(mu=0, var=1, n=10000, repeticiones=3):
    t1 = timeit.timeit(lambda: normal_12_uniformes(mu, var, n), number=repeticiones)
    t2 = timeit.timeit(lambda: normal_rechazo(mu, var, n), number=repeticiones)
    t3 = timeit.timeit(lambda: normal_box_muller(mu, var, n), number=repeticiones)
    t4 = timeit.timeit(lambda: np.random.normal(loc=mu, scale=np.sqrt(var), size=n), number=repeticiones)

    print(f"n = {n}, repeticiones = {repeticiones}")
    print(f"1) 12 uniformes       : {t1:.6f} s")
    print(f"2) Rechazo            : {t2:.6f} s")
    print(f"3) Box-Muller         : {t3:.6f} s")
    print(f"4) numpy.random.normal: {t4:.6f} s")


if __name__ == "__main__":
    mu = 5
    var = 4
    n = 10000

    x1 = normal_12_uniformes(mu, var, n)
    x2 = normal_rechazo(mu, var, n)
    x3 = normal_box_muller(mu, var, n)
    x4 = np.random.normal(loc=mu, scale=np.sqrt(var), size=n)

    print("Medias aproximadas:")
    print("12 uniformes :", np.mean(x1))
    print("Rechazo      :", np.mean(x2))
    print("Box-Muller   :", np.mean(x3))
    print("NumPy        :", np.mean(x4))

    print("\nVarianzas aproximadas:")
    print("12 uniformes :", np.var(x1))
    print("Rechazo      :", np.var(x2))
    print("Box-Muller   :", np.var(x3))
    print("NumPy        :", np.var(x4))

    print("\nComparación de tiempos:")
    comparar_tiempos(mu=mu, var=var, n=10000, repeticiones=3)