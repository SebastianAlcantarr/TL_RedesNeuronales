"""Grafica accuracy y loss por época a partir del log de SLURM (resultados_*.txt)."""
import re
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = sys.argv[1] if len(sys.argv) > 1 else "resultados_2203131.txt"
texto = open(log, errors="ignore").read()
texto = re.sub(r"\x1b\[[0-9;]*m", "", texto).replace("\x08", "")

corridas, actual = [], []
for bloque in re.split(r"(?=Epoch \d+/\d+\n)", texto)[1:]:
    epoca = int(re.match(r"Epoch (\d+)/", bloque).group(1))
    m = re.search(r"339/339 ━+ \d+s [^\n]*? accuracy: ([\d.]+) - loss: ([\d.]+)"
                  r" - val_accuracy: ([\d.]+) - val_loss: ([\d.]+)", bloque)
    if not m:
        continue
    if epoca == 1 and actual:
        corridas.append(actual)
        actual = []
    actual.append((epoca, *map(float, m.groups())))
corridas.append(actual)

# MobileNetV2: épocas 1-10 de la 1a corrida; EfficientNetB0: cabeza entrenada, 2a corrida
series = {"MobileNetV2 (cabeza)": corridas[0][:10], "EfficientNetB0 (cabeza)": corridas[1]}

fig, ejes = plt.subplots(2, 2, figsize=(11, 8))
for col, (nombre, datos) in enumerate(series.items()):
    ep = [d[0] for d in datos]
    ejes[0][col].plot(ep, [d[1] for d in datos], label="Train")
    ejes[0][col].plot(ep, [d[3] for d in datos], label="Validación")
    ejes[0][col].set_title(f"{nombre}: accuracy")
    ejes[0][col].set_ylim(0, 1.05)
    ejes[1][col].plot(ep, [d[2] for d in datos], label="Train")
    ejes[1][col].plot(ep, [d[4] for d in datos], label="Validación")
    ejes[1][col].set_title(f"{nombre}: loss")
    for fila in (0, 1):
        ejes[fila][col].set_xlabel("Época")
        ejes[fila][col].grid(True)
        ejes[fila][col].legend()
plt.tight_layout()
plt.savefig("graficas/acc_loss.png", dpi=150)
