import matplotlib

matplotlib.use("TkAgg")
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def imshow(
    img,
    new_fig=True,
    title=None,
    color_img=False,
    blocking=False,
    colorbar=False,
    ticks=False,
):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap="gray")
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:
        plt.show(block=blocking)


def imreconstruct(marker, mask, kernel=None):
    # Asegurarse de que marker y mask sean del mismo tamaño y tipo
    if marker.shape != mask.shape:
        raise ValueError("El tamaño de 'marker' y 'mask' debe ser igual")
    if marker.dtype != mask.dtype:
        marker = marker.astype(mask.dtype)

    # Definir el kernel si no se proporciona
    if kernel is None:
        kernel = np.ones((3, 3), np.uint8)

    while True:
        # Dilatación
        expanded = cv2.dilate(marker, kernel)

        # Intersección entre la imagen dilatada y la máscara
        expanded_intersection = cv2.bitwise_and(expanded, mask)

        # Verificar si la reconstrucción ha convergido
        if np.array_equal(marker, expanded_intersection):
            break

        # Actualizar el marcador
        marker = expanded_intersection

    return expanded_intersection


def imfillhole(img):
    # img: Imagen binaria de entrada. Valores permitidos: 0 (False), 255 (True).
    mask = np.zeros_like(img)
    mask = cv2.copyMakeBorder(
        mask[1:-1, 1:-1], 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=int(255)
    )
    marker = cv2.bitwise_not(
        img, mask=mask
    )  # El marcador lo defino como el complemento de los bordes.
    img_c = cv2.bitwise_not(
        img
    )  # La mascara la defino como el complemento de la imagen.
    img_r = imreconstruct(
        marker=marker, mask=img_c
    )  # Calculo la reconstrucción R_{f^c}(f_m)
    img_fh = cv2.bitwise_not(
        img_r
    )  # La imagen con sus huecos rellenos es igual al complemento de la reconstruccion.
    return img_fh


def se_mueve_no_se_mueve(foto, centroides):
    anterior = ""
    actual = ""
    lista = []
    for i in centroides:
        if i[0] == (foto - 1):
            anterior = i
        if i[0] == foto:
            actual = i
            break
    if anterior == "" or actual == "":
        return lista

    ant_cen = anterior[1]
    act_cen = actual[1]
    for j in range(len(ant_cen)):
        if (act_cen[j][1][0] - 40.0) < ant_cen[j][1][0] < (
            act_cen[j][1][0] + 40.0
        ) and (act_cen[j][1][1] - 40.0) < ant_cen[j][1][1] < (act_cen[j][1][1] + 40.0):
            lista.append(act_cen[j][0])
    return lista


def procesar_imagenes(num_foto, frame, num_foto_centroides_dados=[]):
    # ESTO ES PARA CALCULAR LOS BOUNDING BOXES DE LOS DADOS

    num_cara_caja = []
    # for num_foto in range(frame_number): img = cv2.imread(f"frames\\frame_{num_foto}.jpg", cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY)
    canny = cv2.Canny(img_gray, 50, 75)
    engrosar_bordes_generales = cv2.dilate(canny, np.ones((17, 17), np.uint8))
    relleno_dados = imfillhole(engrosar_bordes_generales)
    erocion_dado = cv2.erode(relleno_dados, np.ones((3, 3), np.uint8))

    # ESTO ES PARA CALCULAR EL VALOR DE LA CARA DEL DADO
    canny_circulos = cv2.Canny(img_gray, 175, 255)
    engrosar_circulos = cv2.dilate(canny_circulos, np.ones((3, 3), np.uint8))
    relleno_circulos = imfillhole(engrosar_circulos)
    erocion_circulos = cv2.erode(relleno_circulos, np.ones((5, 5), np.uint8))

    # ACA OBTENESMOS LOS BOUNDING BOXES DE LOS DADOS
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        erocion_dado
    )
    indice_stats = []
    centroides_dados = []
    for i in range(1, num_labels):  # Comienza desde 1 para omitir el fondo (label 0)
        # Obtén las estadísticas de cada componente
        x, y, w, h, area = stats[i]
        if 50 < w < 120 and 50 < h < 140:
            factor_forma = w / h
            if 0.5 < factor_forma < 2:
                if 4500 < area < 11000:
                    indice_stats.append(i)
                    centroides_dados.append([i, centroids[i]])

    if len(centroides_dados) >= 5:
        # print(centroides_dados)
        num_foto_centroides_dados.append([num_foto, centroides_dados])

    if len(num_foto_centroides_dados) >= 2:

        lista = se_mueve_no_se_mueve(num_foto, num_foto_centroides_dados)
        # print(lista) CALCULAR EL VALOR DE LA CARA
        num_cara_caja = []
        for j in lista:
            x2, y2, w2, h2, area2 = stats[j]
            img_circulos = np.zeros_like(img_gray)
            img_circulos[y2 : y2 + h2, x2 : x2 + w2] = erocion_circulos[
                y2 : y2 + h2, x2 : x2 + w2
            ]
            num_labels_2, labels_2, stats_2, centroids_2 = (
                cv2.connectedComponentsWithStats(img_circulos)
            )
            # imshow(labels_2, title='Original') print(num_labels_2)
            cara_dado = 0
            for i_label in range(1, num_labels_2):
                area_circulos = stats_2[i_label, cv2.CC_STAT_AREA]
                # print(j) print(area_circulos)
                if 30 < area_circulos < 150:
                    # Crear una máscara binaria para cada componente
                    mask = (labels_2 == i_label).astype(np.uint8) * 255
                    contours, _ = cv2.findContours(
                        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
                    )
                    # print(len(contours))
                    for contour in contours:
                        perimetro_circulo = cv2.arcLength(
                            contour, True
                        )  # True porque el contorno está cerrado
                        factor_forma_circulo = area_circulos / (perimetro_circulo**2)
                        # print(factor_forma_circulo)
                        if 0.050 < factor_forma_circulo < 0.10:
                            cara_dado = cara_dado + 1

            num_cara_caja.append([j, cara_dado])

    if len(indice_stats) >= 5:
        # FALTA ELIMINAR SI TIENE UN SOLO BOUDING BOXES
        for k in indice_stats:
            if not num_cara_caja:
                x3, y3, w3, h3, area3 = stats[k]
                punto1 = (x3, y3)  # Esquina superior izquierda
                punto2 = (x3 + w3, y3 + h3)  # Esquina inferior derecha

                # Dibujar el rectángulo
                cv2.rectangle(img_rgb, punto1, punto2, color=(0, 0, 255), thickness=5)
            else:
                for g in num_cara_caja:
                    if k != g[0]:
                        x3, y3, w3, h3, area3 = stats[k]
                        punto1 = (x3, y3)  # Esquina superior izquierda
                        punto2 = (x3 + w3, y3 + h3)  # Esquina inferior derecha

                        # Dibujar el rectángulo
                        cv2.rectangle(
                            img_rgb, punto1, punto2, color=(0, 0, 255), thickness=5
                        )
                    else:
                        x3, y3, w3, h3, area3 = stats[k]
                        punto1 = (x3, y3)  # Esquina superior izquierda
                        punto2 = (x3 + w3, y3 + h3)  # Esquina inferior derecha

                        # Dibujar el rectángulo
                        cv2.rectangle(
                            img_rgb, punto1, punto2, color=(0, 0, 255), thickness=5
                        )
                        # Coordenadas iniciales para el texto
                        text_x = x3 - 10
                        text_y = y3 - 10

                        # Asegurarse de que el texto no salga de los límites de la imagen
                        if text_x < 0:
                            text_x = (
                                x3 + 10
                            )  # Mueve el texto hacia la derecha si está muy cerca del borde izquierdo
                        if text_y < 0:
                            text_y = (
                                y3 + 30
                            )  # Mueve el texto hacia abajo si está muy cerca del borde superior
                        if text_x + 200 > img_rgb.shape[1]:
                            text_x = (
                                x3 - 150
                            )  # Mueve el texto hacia la izquierda si está cerca del borde derecho
                        if text_y + 30 > img_rgb.shape[0]:
                            text_y = (
                                y3 - 30
                            )  # Mueve el texto hacia arriba si está cerca del borde inferior
                        # Añadir la etiqueta (texto)
                        cv2.putText(
                            img_rgb,
                            f"Valor es: {g[1]}",
                            (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1,
                            color=(0, 255, 0),
                            thickness=2,
                        )

    # imshow(img_rgb, title='Original')
    terminada = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return terminada, num_foto_centroides_dados


# os.makedirs("frames", exist_ok = True)  # Si no existe, crea la carpeta 'frames' en el directorio actual.

# --- Leer un video ------------------------------------------------
for num_video in range(1, 5):
    cap = cv2.VideoCapture(f"tirada_{num_video}.mp4")
    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )  # Obtiene el ancho del video en píxeles usando la propiedad CAP_PROP_FRAME_WIDTH.
    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )  # Obtiene la altura del video en píxeles usando la propiedad CAP_PROP_FRAME_HEIGHT.
    fps = int(
        cap.get(cv2.CAP_PROP_FPS)
    )  # Obtiene los cuadros por segundo (FPS) del video usando CAP_PROP_FPS.
    n_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )  # Obtiene el número total de frames en el video usando CAP_PROP_FRAME_COUNT.
    frame_number = 0
    out = cv2.VideoWriter(
        f"Video{num_video}-procesado.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    num_foto_centroides_dados = []
    bandera = True
    while cap.isOpened():  # Verifica si el video se abrió correctamente.

        ret, frame = (
            cap.read()
        )  # 'ret' indica si la lectura fue exitosa (True/False) y 'frame' contiene el contenido del frame si la lectura fue exitosa.
        if ret == True:
            if bandera:
                ancho = int(len(frame[0]) / 2)
                cont = 0
                for j in range(len(frame)):
                    if frame[j][ancho][2] < 50:
                        cont = cont + 1
                bandera = False
            largo = len(frame[0])
            # print(num_foto_centroides_dados)
            foto_recortada, num_foto_centroides_dados = procesar_imagenes(
                frame_number, frame[0:cont, 0:largo], num_foto_centroides_dados
            )

            frame_number = frame_number + 1
            frame[0:cont, 0:largo] = foto_recortada
            out.write(
                frame
            )  # Escribe el frame original (sin redimensionar) en el archivo de salida 'Video-Output.mp4'. IMPORTANTE: El tamaño del frame debe coincidir con el tamaño especificado al crear 'out'.
        else:
            break
    cap.release()  # Libera el objeto 'cap', cerrando el archivo.
    out.release()  # Libera el objeto 'out',  cerrando el archivo.
    cv2.destroyAllWindows()  # Cierra todas las ventanas abiertas.
