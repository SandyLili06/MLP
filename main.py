import sys
import os
from mlp import calcular_energia
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QGridLayout, QHBoxLayout,
    QLabel, QPushButton, QTabWidget, QFileDialog
)
from convertir import convertir_archivo
from graficas import (
    grafica_real_vs_predicha,
    grafica_error,
    grafica_loss,
    grafica_mae
)
from PyQt5.QtWidgets import QScrollArea
from PyQt5.QtWidgets import QFormLayout, QLineEdit
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtCore import Qt

form = QFormLayout()

class VentanaPrincipal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Calculador de la Energía de Interación - MLP")
        self.setGeometry(200, 70, 930, 630)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.aplicar_estilos()
        self.tabs.addTab(self.crear_tab_archivos(), "Predicción desde Archivo")
        self.tabs.addTab(self.crear_tab_metricas(), "Métricas de evaluación")
        self.tabs.addTab(self.crear_tab_resultados(), "Gráficas de Resultados")

    def aplicar_estilos(self):
        estilo = """
        QMainWindow {
            background-color: #f5f7fa;
        }

        QWidget {
            font-family: Arial;
            font-size: 12px;
        }

        QTabWidget::pane {
            border: 1px solid #d0d7de;
            background: white;
        }

        QTabBar::tab {
            background: #e9eef5;
            padding: 10px 20px;
            border-radius: 5px;
            margin: 2px;
        }

        QTabBar::tab:selected {
            background: #2563eb;
            color: white;
        }

        QPushButton {
            background-color: #2563eb;
            color: white;
            border-radius: 8px;
            padding: 10px;
            font-weight: bold;
        }

        QPushButton:hover {
            background-color: #1d4ed8;
        }

        QPushButton:pressed {
            background-color: #1e40af;
        }

        QLabel {
            background-color: white;
            border-radius: 8px;
            padding: 10px;
            color: #1f2937;
        }

        QScrollArea {
            border: none;
            background-color: transparent;
        }

        QTextEdit {
            background-color: white;
            border-radius: 8px;
            padding: 10px;
        }
        
        QPushButton#btnLimpiar {
            background-color: #dc2626;
        }
        """
        self.setStyleSheet(estilo)

    def crear_tab_archivos(self):
        tab = QWidget()
        layout = QVBoxLayout()

        btn_abrir = QPushButton("Selecciona un archivo")
        btn_abrir.clicked.connect(self.abrir_archivo)
        layout.addWidget(btn_abrir)

        self.btn_convertir = QPushButton("Convertir a CSV")
        self.btn_convertir.clicked.connect(self.ejecutar_conversion)
        self.btn_convertir.hide()
        layout.addWidget(self.btn_convertir)

        self.btn_calcular = QPushButton("Calcular Energía")
        self.btn_calcular.clicked.connect(self.ejecutar_mlp)
        self.btn_calcular.hide()
        layout.addWidget(self.btn_calcular)

        btn_limpiar = QPushButton("Limpiar")
        btn_limpiar.clicked.connect(self.limpiar)
        layout.addWidget(btn_limpiar)

        self.label_archivo = QLabel("Archivo no seleccionado")
        self.label_archivo.setWordWrap(True)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.label_archivo)
        layout.addWidget(scroll)

        tab.setLayout(layout)
        return tab

    def crear_tab_metricas(self):
        tab = QWidget()
        layout = QVBoxLayout()

        self.label_metricas = QLabel("Calcula la Energía para mostrar las Métricas")
        self.label_metricas.setWordWrap(True)
        self.label_metricas.setMargin(20)
        self.label_metricas.setAlignment(
            Qt.AlignCenter
        )
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.label_metricas)

        layout.addWidget(scroll)

        layout.addLayout(layout)
        tab.setLayout(layout)
        return tab

    def crear_tab_resultados(self):
        tab = QWidget()
        layout = QVBoxLayout()

        self.label_resumen = QLabel("")
        layout.addWidget(self.label_resumen)


        self.figure1 = Figure(figsize=(5, 4))
        self.canvas1 = FigureCanvas(self.figure1)

        self.figure2 = Figure(figsize=(5, 4))
        self.canvas2 = FigureCanvas(self.figure2)

        self.figure3 = Figure(figsize=(5, 4))
        self.canvas3 = FigureCanvas(self.figure3)

        self.figure4 = Figure(figsize=(5, 4))
        self.canvas4 = FigureCanvas(self.figure4)

        graficas = QGridLayout()

        graficas.addWidget(self.canvas1, 0, 0)
        graficas.addWidget(self.canvas2, 0, 1)
        graficas.addWidget(self.canvas3, 1, 0)
        graficas.addWidget(self.canvas4, 1, 1)

        layout.addLayout(graficas)

        tab.setLayout(layout)

        return tab

    def limpiar(self):
        self.archivo_actual = None
        self.history = None
        self.metricas = None

        self.label_archivo.setText(
            "Calcula la Energía para mostrar los Resultados"
        )

        self.label_metricas.setText(
            "Calcula la Energía para mostrar las Métricas"
        )

        self.label_resumen.setText(
            "Calcula la Energía para mostrar las Gráficas"
        )

        self.figure1.clear()
        self.figure2.clear()
        self.figure3.clear()
        self.figure4.clear()
        self.canvas1.draw()
        self.canvas2.draw()
        self.canvas3.draw()
        self.canvas4.draw()

    def abrir_archivo(self):
        try:
            archivo, _ = QFileDialog.getOpenFileName(self,"Seleccionar archivo","","Todos los archivos (*)"
            )
            if archivo:
                extension = os.path.splitext(archivo)[1].lower()
                if extension == ".txt":
                    self.archivo_actual = archivo
                    self.label_archivo.setText(archivo)

                    self.btn_convertir.show()
                    self.btn_calcular.hide()

                elif extension == ".csv":
                    self.archivo_actual = archivo
                    self.label_archivo.setText(archivo)

                    self.btn_calcular.show()
                    self.btn_convertir.hide()

                else:
                    self.label_archivo.setText("Formato no válido, seleccione un archivo .txt o .csv")
                    self.btn_convertir.hide()
                    self.btn_calcular.hide()
        except Exception as e:
            print(e)

    def ejecutar_conversion(self):
        try:
            archivo_csv = convertir_archivo(
                self.archivo_actual
            )
            self.archivo_actual = archivo_csv
            self.label_archivo.setText(
                f"CSV generado:\n{archivo_csv}"
            )
            self.btn_calcular.show()
        except Exception as e:
            print(e)

    def ejecutar_mlp(self):
        try:
            resultado, history, metricas = calcular_energia(self.archivo_actual)
            self.history = history
            self.metricas = metricas
            self.actualizar_metricas(metricas)
            self.actualizar_resultados(resultado, history)

            datos_mostrar = resultado[
                ["angulo", "distancia", "energia", "energia_predicha"]
            ]

            mejor_real = resultado.loc[
                resultado["energia"].idxmin()
            ]

            angulo_real = mejor_real["angulo"]
            distancia_real = mejor_real["distancia"]
            energia_real = mejor_real["energia"]
            energia_predicha_real = mejor_real["energia_predicha"]

            mejor = resultado.loc[
                resultado["energia_predicha"].idxmin()
            ]
            angulo_pred = mejor["angulo"]
            distancia_pred = mejor["distancia"]
            energia_real_pred = mejor["energia"]
            energia_predicha = mejor["energia_predicha"]
            error = abs(energia_real - energia_predicha)

            resumen = f"""
            <h2 align="center" style="color:#1e3a8a;">
            CONFIGURACIÓN MÁS ESTABLE (Material Studio)
            </h2>

            <hr>

            <b>Ángulo:</b> {angulo_real:.1f}°<br>
            <b>Distancia:</b> {distancia_real:.1f} Å<br>
            <b>Energía real:</b> {energia_real:.6f} eV<br>
            <b>Energía predicha:</b> {energia_predicha_real:.6f} eV<br>


            <h2 align="center" style="color:#16a34a;">
            CONFIGURACIÓN ÓPTIMA PREDICHA (MLP)
            </h2>

            <hr>

            <b>Ángulo:</b> {angulo_pred:.1f}°<br>
            <b>Distancia:</b> {distancia_pred:.1f} Å<br>
            <b>Energía real:</b> {energia_real_pred:.6f} eV<br>
            <b>Energía predicha:</b> {energia_predicha:.6f} eV<br>

            <br>

            <h3 style="color:#dc2626;">
            Error absoluto: {error:.6f} eV
            </h3>

            <hr>

            <h3>
            Resultados completos:
            </h3>

            <pre>
            {datos_mostrar.to_string(index=False)}
            </pre>
            """

            self.label_archivo.setText(resumen)
            self.label_archivo.setAlignment(Qt.AlignLeft | Qt.AlignTop)
            self.label_archivo.setMargin(20)
            print(resumen)
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise

    def actualizar_metricas(self, metricas):

        texto = f"""
        <h2 align="center" style="color:#1e3a8a;">
        MÉTRICAS DEL MODELO MLP
        </h2>
        <hr>

        <h3 style="color:#2563eb;">
        Coeficiente de determinación (R²)
        </h3>

        <p align="center">
        <b style="font-size:18px;">
        {metricas['R2']:.6f}
        </b>
        </p>

        <h3 style="color:#16a34a;">
        Error Absoluto Medio (MAE)
        </h3>

        <p align="center">
        <b style="font-size:18px;">
        {metricas['MAE']:.6f} eV
        </b>
        </p>

        <h3 style="color:#f59e0b;">
        Error Cuadrático Medio (MSE)
        </h3>

        <p align="center">
        <b style="font-size:18px;">
        {metricas['MSE']:.6f}
        </b>
        </p>


        <h3 style="color:#dc2626;">
        Raíz del Error Cuadrático Medio (RMSE)
        </h3>

        <p align="center">
        <b style="font-size:18px;">
        {metricas['RMSE']:.6f} eV
        </b>
        </p>

        <hr>

        <p align="center">
        Evaluación del desempeño del modelo neuronal MLP
        </p>
        """
        self.label_metricas.setText(texto)

    def actualizar_resultados(self, resultado,history):
        grafica_real_vs_predicha(
            self.figure1,
            resultado
        )
        self.canvas1.draw()

        grafica_error(
            self.figure2,
            resultado
        )
        self.canvas2.draw()

        grafica_loss(
            self.figure3,
            history
        )
        self.canvas3.draw()

        grafica_mae(
            self.figure4,
            history
        )
        self.canvas4.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ventana = VentanaPrincipal()
    ventana.show()
    sys.exit(app.exec_())