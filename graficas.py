
def grafica_real_vs_predicha(figura, resultado):
    figura.clear()
    ax = figura.add_subplot(111)

    ax.scatter(
        resultado["energia"],
        resultado["energia_predicha"],
        alpha=0.7
    )

    minimo = min(
        resultado["energia"].min(),
        resultado["energia_predicha"].min()
    )

    maximo = max(
        resultado["energia"].max(),
        resultado["energia_predicha"].max()
    )

    ax.plot(
        [minimo, maximo],
        [minimo, maximo],
        linestyle="--"
    )
    ax.set_title("Energía Predicha vs Energía Real")
    ax.set_xlabel("Energía Real (eV)")
    ax.set_ylabel("Energía Predicha (eV)")
    figura.tight_layout()


def grafica_error(figura, resultado):
    figura.clear()
    ax = figura.add_subplot(111)
    error = abs(
        resultado["energia"]
        - resultado["energia_predicha"]
    )

    ax.hist(
        error,
        bins=20
    )
    ax.set_title("Distribución del Error")
    ax.set_xlabel("Error absoluto (eV)")
    ax.set_ylabel("Frecuencia")

    figura.tight_layout()

def grafica_loss(figura, history):
    figura.clear()
    ax = figura.add_subplot(111)

    ax.plot(
        history["loss"],
        label="Entrenamiento"
    )

    ax.plot(
        history["val_loss"],
        label="Validación"
    )

    ax.set_title("Evolución de pérdida de error")
    ax.set_xlabel("Épocas")
    ax.set_ylabel("MSE")
    ax.legend()
    figura.tight_layout()

def grafica_mae(figura, history):
    figura.clear()
    ax = figura.add_subplot(111)

    ax.plot(
        history["mae"],
        label="Entrenamiento"
    )

    ax.plot(
        history["val_mae"],
        label="Validación"
    )

    ax.set_title("Curva MAE")
    ax.set_xlabel("Épocas")
    ax.set_ylabel("MAE (eV)")
    ax.legend()
    figura.tight_layout()

