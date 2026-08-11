import pandas as pd
import os

def convertir_archivo(archivo_txt):
    archivo_csv = os.path.splitext(archivo_txt)[0] + ".csv"
    filas = []
    with open(archivo_txt, "r") as f:
        lineas = [line.strip() for line in f if line.strip()]
    i = 0
    while i < len(lineas):
        coordenadas = []
        for _ in range(30):
            valores = list(map(float, lineas[i].split()))
            coordenadas.extend(valores)
            i += 1
        angulo, distancia, energia = map(float, lineas[i].split())
        i += 1
        fila = coordenadas + [
            angulo,
            distancia,
            energia
        ]
        filas.append(fila)

    columnas = []

    for atomo in range(1, 31):
        columnas.extend([
            f"x{atomo}",
            f"y{atomo}",
            f"z{atomo}"
        ])

    columnas.extend([
        "angulo",
        "distancia",
        "energia"
    ])

    df = pd.DataFrame(
        filas,
        columns=columnas
    )

    df.to_csv(
        archivo_csv,
        index=False
    )

    return archivo_csv