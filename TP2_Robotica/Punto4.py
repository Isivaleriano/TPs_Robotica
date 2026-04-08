import numpy as np
import matplotlib.pyplot as plt


def matriz_transicion(n=20, accion="avanzar"):
    """
    Devuelve una matriz T de tamaño n x n donde:
    T[j, i] = P(x_t = j | x_{t-1} = i, accion)
    """
    T = np.zeros((n, n))

    for i in range(n):
        if accion == "avanzar":
            if i == n - 1:
                # última celda
                T[n - 1, i] = 1.0
            elif i == n - 2:
                # penúltima celda
                T[n - 2, i] = 0.25
                T[n - 1, i] = 0.75
            else:
                T[i, i] = 0.25
                T[i + 1, i] = 0.50
                T[i + 2, i] = 0.25

        elif accion == "retroceder":
            if i == 0:
                # primera celda
                T[0, i] = 1.0
            elif i == 1:
                # segunda celda
                T[1, i] = 0.25
                T[0, i] = 0.75
            else:
                T[i, i] = 0.25
                T[i - 1, i] = 0.50
                T[i - 2, i] = 0.25

        else:
            raise ValueError("La acción debe ser 'avanzar' o 'retroceder'.")

    return T


def aplicar_accion(bel, T):
    """
    Actualización del belief:
    bel_nuevo = T @ bel
    """
    return T @ bel


# -----------------------------
# Parámetros del problema
# -----------------------------
n = 20
bel = np.hstack((np.zeros(10), [1.0], np.zeros(9)))

T_avanzar = matriz_transicion(n, "avanzar")
T_retroceder = matriz_transicion(n, "retroceder")

# 9 acciones de avanzar
for _ in range(9):
    bel = aplicar_accion(bel, T_avanzar)

# 3 acciones de retroceder
for _ in range(3):
    bel = aplicar_accion(bel, T_retroceder)

# -----------------------------
# Resultados
# -----------------------------
print("Belief final:")
print(bel)
print("\nSuma de probabilidades:", bel.sum())
print("Celda más probable (índice Python):", np.argmax(bel))
print("Probabilidad máxima:", bel[np.argmax(bel)])

# -----------------------------
# Gráfico
# -----------------------------
plt.figure(figsize=(10, 5))
plt.bar(np.arange(n), bel)
plt.xlabel("Celda")
plt.ylabel("Probabilidad")
plt.title("Belief final luego de 9 'avanzar' y 3 'retroceder'")
plt.grid(axis="y", alpha=0.3)
plt.show()