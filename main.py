import time
import socket
import threading
import os
import io
import queue
from dataclasses import dataclass, field
from typing import Dict, Optional

# ============================================================
# CONFIGURACIÓN
# ============================================================
MODO_DRON = False
MODEL_PATH = "Modelo/yolo11n-pose.pt"

# ---- DETECCIÓN ----
DISTANCIA_PERSONAS = 250
DISTANCIA_MANO_CABEZA = 120
SAVE_INTERVAL = 5

# ---- KEYPOINTS ----
KP_CABEZA = 0
KP_MUÑECA_IZQ = 9
KP_MUÑECA_DER = 10

print(f"Iniciando Vanguardia UCE | Modo: {'DRON' if MODO_DRON else 'WEBCAM'}")

import cv2
import numpy as np
from PIL import Image, ImageFile
from ultralytics import YOLO

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ============================================================
# APP PRINCIPAL
# ============================================================
class DroneVanguardIA:
    def __init__(self):
        self.running = True
        self.frame_queue = queue.Queue(maxsize=1)

        self.window_name = "VANGUARDIA UCE"

        self.last_save_time = 0

        self.output_dir = "capturas"
        os.makedirs(self.output_dir, exist_ok=True)

        print("Cargando modelo YOLO...")
        self.model = YOLO(MODEL_PATH)

        if not MODO_DRON:
            self.cap = cv2.VideoCapture(0)

    # ========================================================
    # UTILIDADES
    # ========================================================
    def distancia(self, p1, p2):
        return np.linalg.norm(p1 - p2)

    def punto_valido(self, p):
        return p[0] > 0 and p[1] > 0

    # ========================================================
    # DETECCIÓN DE CONFLICTO
    # ========================================================
    def detectar_conflicto(self, coords, kpts):
        num_personas = len(coords)

        if num_personas < 2:
            return False

        for i in range(num_personas):
            for j in range(i + 1, num_personas):

                # centros
                c1 = np.array([(coords[i][0]+coords[i][2])/2,
                               (coords[i][1]+coords[i][3])/2])

                c2 = np.array([(coords[j][0]+coords[j][2])/2,
                               (coords[j][1]+coords[j][3])/2])

                if self.distancia(c1, c2) < DISTANCIA_PERSONAS:

                    cabeza_i = kpts[i][KP_CABEZA]
                    cabeza_j = kpts[j][KP_CABEZA]

                    manos_i = [kpts[i][KP_MUÑECA_IZQ], kpts[i][KP_MUÑECA_DER]]
                    manos_j = [kpts[j][KP_MUÑECA_IZQ], kpts[j][KP_MUÑECA_DER]]

                    # validar puntos
                    if not (self.punto_valido(cabeza_i) and self.punto_valido(cabeza_j)):
                        continue

                    for mano in manos_i:
                        if self.punto_valido(mano):
                            if self.distancia(mano, cabeza_j) < DISTANCIA_MANO_CABEZA:
                                return True

                    for mano in manos_j:
                        if self.punto_valido(mano):
                            if self.distancia(mano, cabeza_i) < DISTANCIA_MANO_CABEZA:
                                return True

        return False

    # ========================================================
    # CAPTURA
    # ========================================================
    def video_receiver(self):
        print("📷 Webcam activa")
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(frame)

    # ========================================================
    # MAIN LOOP
    # ========================================================
    def run(self):
        threading.Thread(target=self.video_receiver, daemon=True).start()

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1)
            except:
                continue

            results = self.model.predict(frame, conf=0.5, imgsz=256, verbose=False)

            for r in results:
                annotated = r.plot()

                conflicto = False

                if r.boxes is not None and r.keypoints is not None:
                    coords = r.boxes.xyxy.cpu().numpy()
                    kpts = r.keypoints.xy.cpu().numpy()

                    # DEBUG: dibujar keypoints
                    for persona in kpts:
                        for punto in persona:
                            x, y = int(punto[0]), int(punto[1])
                            if x > 0 and y > 0:
                                cv2.circle(annotated, (x, y), 4, (0,255,0), -1)

                    conflicto = self.detectar_conflicto(coords, kpts)

                if conflicto:
                    cv2.putText(annotated, "POSIBLE PELEA",
                                (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 3)

                    current_time = time.time()

                    if current_time - self.last_save_time >= SAVE_INTERVAL:
                        filename = os.path.join(
                            self.output_dir,
                            f"pelea_{int(current_time)}.jpg"
                        )

                        cv2.imwrite(filename, annotated)
                        print(f"📸 Pelea detectada: {filename}")

                        self.last_save_time = current_time

                cv2.imshow(self.window_name, annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

        self.cap.release()
        cv2.destroyAllWindows()


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    app = DroneVanguardIA()
    app.run()