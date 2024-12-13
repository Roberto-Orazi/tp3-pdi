import cv2
import numpy as np
import os

def recortar_transformar_hsv(frame):
    H_low, S_low, V_low = 0, 190, 50
    H_high, S_high, V_high = 100, 250, 230
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_threshold = cv2.inRange(frame_hsv, (H_low, S_low, V_low), (H_high, S_high, V_high))
    white_pixels = np.column_stack(np.where(frame_threshold == 255))
    min_y, min_x = white_pixels.min(axis=0) 
    max_y, max_x = white_pixels.max(axis=0)
    return min_x, min_y, max_x, max_y


def detectar_dados_rojos(frame):
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    rojo_bajo1, rojo_alto1 = np.array([0, 100, 50]), np.array([50, 255, 255])
    rojo_bajo2, rojo_alto2 = np.array([150, 100, 50]), np.array([200, 255, 255])
    mascara1 = cv2.inRange(frame_hsv, rojo_bajo1, rojo_alto1)
    mascara2 = cv2.inRange(frame_hsv, rojo_bajo2, rojo_alto2)
    return cv2.bitwise_or(mascara1, mascara2)


def detectar_ubicacion_dados(frame):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(frame)
    caja_centroides = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if 1500 < area < 5500 and 0.5 < w/h < 1.5:
            caja_centroides.append([x, y, w, h, centroids[i]])
    if len(caja_centroides) >= 3:
        return caja_centroides
    else:
        return []


def calcular_cara(frame, box):
    suma = 0
    x, y, w, h, centroids = box
    frame_recortado = frame[y-10:y+h+10, x-10:x+w+10]
    img_gray = cv2.cvtColor(frame_recortado, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img_gray, 120, 255, cv2.THRESH_BINARY)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh)

    for i in range(1, num_labels):  # Ignorar el fondo (etiqueta 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if 80 < area < 250:
            # Crear una máscara para el componente conectado actual
            mask = (labels == i).astype(np.uint8) * 255

            # Calcular contornos para determinar el perímetro
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                perimeter = cv2.arcLength(contours[0], True)
                factor_forma = area / (perimeter ** 2)
                if 0.05 < factor_forma < 0.09:
                    suma += 1

    return suma


def crear_caja_etiqueta(frame, box, num_img):
    centroides_act = []
    if box[num_img] != []:
        for caja in box[num_img]:
            x, y, w, h, centroides = caja
            centroides_act.append(centroides)
            # Dibujar el rectángulo
            cv2.rectangle(frame, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=5)
            
        centroides_ant = [box[4] for box in box[num_img - 1]]
        
        centroides_act.sort(key=lambda coord: np.linalg.norm(coord))
        centroides_ant.sort(key=lambda coord: np.linalg.norm(coord))
        
        for c_act, c_ant in zip(centroides_act, centroides_ant):
            if c_ant[0]-10 < c_act[0] < c_ant[0]+10 and c_ant[1]-10 < c_act[1] < c_ant[1]+10:
                for i in box[num_img]:
                    if list(i[4]) == list(c_act):
                        num_cara = calcular_cara(frame,i)
                        cv2.putText(frame, f'Valor es: {num_cara}', (i[0]-10, i[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            fontScale=0.75, color=(0, 255, 0), thickness=2) 
    
    return frame


def procesar_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps, n_frames = int(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    boxes = []
    for frame_number in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_number == 0:
            x, y, w, h = recortar_transformar_hsv(frame)
        frame_cropped = frame[y + 20:y + h - 100, x + 20:x + w - 20]
        bin_dados = detectar_dados_rojos(frame_cropped)
        boxes.append(detectar_ubicacion_dados(bin_dados))
        frame_final = crear_caja_etiqueta(frame_cropped, boxes, frame_number)
        frame[y + 20:y + h - 100, x + 20:x + w - 20] = frame_final
        out.write(frame)
    cap.release()
    out.release()


# Procesar múltiples videos
for i in range(1, 5):
    procesar_video(f'tirada_{i}.mp4', f'Video{i}-procesado.mp4')
    print(f'Video {i} procesado')
