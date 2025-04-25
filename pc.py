import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

# Opsi konfigurasi
use_camera = True  # Ubah ke True jika ingin menggunakan kamera
video_source = 0 if use_camera else "tes1.webm"
confidence_threshold = 0.3  # Diturunkan untuk mendeteksi lebih banyak objek
nms_threshold = 0.5  # Ditingkatkan untuk mempertahankan lebih banyak deteksi


# Memuat model YOLOv4 - pastikan hanya menggunakan CPU
print("Memuat model YOLOv4...")
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
print("Model berhasil dimuat.")

# Memuat nama kelas COCO
try:
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    print(f"Berhasil memuat {len(classes)} kelas dari coco.names")
except FileNotFoundError:   
    print("Error: File coco.names tidak ditemukan.")
    # Gunakan daftar kelas yang diberikan
    classes = [
        "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
        "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop",
        "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]
    print(f"Menggunakan {len(classes)} kelas default")
    
# Mendapatkan nama layer output
layer_names = net.getUnconnectedOutLayersNames()
print(f"Layer output: {layer_names}")

# Inisialisasi DeepSORT Tracker dengan parameter yang disesuaikan
print("Inisialisasi DeepSORT tracker...")
tracker = DeepSort(
    max_age=60,            # Ditingkatkan untuk pelacakan yang lebih lama
    n_init=1,              # Diturunkan untuk mendeteksi lebih cepat
    max_iou_distance=0.8,  # Ditingkatkan untuk mencocokkan lebih banyak deteksi
    max_cosine_distance=0.3,  # Ditingkatkan untuk toleransi lebih besar
    nn_budget=150          # Ditingkatkan untuk menyimpan lebih banyak penampilan
)

# Inisialisasi video capture
print(f"Membuka sumber video: {video_source}")
cap = cv2.VideoCapture(video_source)
if not cap.isOpened():
    print(f"Error: Tidak dapat membuka sumber video {video_source}")
    exit()

# Dapatkan properti video
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Ukuran video: {width}x{height} @ {fps} FPS")

input_size = (416, 416)  

# Parameter untuk multi-skala deteksi
scales = [0.8, 1.0, 1.2]  # Deteksi pada skala yang berbeda

# Variabel penghitung deteksi objek
object_counts = {class_name: 0 for class_name in classes}  # Penghitung untuk setiap kelas
avg_counts = {class_name: [] for class_name in classes}  # Untuk penghalusan tampilan hitungan

# Menetapkan warna untuk setiap kelas objek
np.random.seed(42)  # Untuk warna yang konsisten
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype=np.uint8).tolist()

# Variabel untuk kelas yang dipilih untuk ditampilkan
displayed_classes = ["person", "car", "motorbike", ]  # Kelas yang akan ditampilkan secara default

# Fungsi untuk menggabungkan deteksi dari multi-skala
def multi_scale_detection(frame, net, layer_names, scales, conf_threshold):
    height, width = frame.shape[:2]
    all_bboxes = []
    all_confidences = []
    all_class_ids = []
    
    for scale in scales:
        # Hitung dimensi baru berdasarkan skala
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize frame
        resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Buat blob dan jalankan deteksi
        blob = cv2.dnn.blobFromImage(resized, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward(layer_names)
        
        # Proses output
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > conf_threshold:  # Deteksi semua kelas, tidak hanya person
                    # Konversi koordinat relatif terhadap skala asli
                    center_x = int((detection[0] * new_width) / scale)
                    center_y = int((detection[1] * new_height) / scale)
                    w = int((detection[2] * new_width) / scale)
                    h = int((detection[3] * new_height) / scale)
                    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    # Validasi koordinat
                    x = max(0, x)
                    y = max(0, y)
                    w = min(width - x, w)
                    h = min(height - y, h)
                    
                    if w > 0 and h > 0:
                        all_bboxes.append([x, y, w, h])
                        all_confidences.append(float(confidence))
                        all_class_ids.append(class_id)
    
    return all_bboxes, all_confidences, all_class_ids

# Inisialisasi penghitung terdeteksi untuk setiap kelas
detected_counts = {class_name: 0 for class_name in classes}

print("Memulai pemrosesan video...")
# Loop pemrosesan utama
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Akhir aliran video")
        break
    
    # Gunakan deteksi multi-skala
    bboxes, confidences, class_ids = multi_scale_detection(frame, net, layer_names, scales, confidence_threshold)
    
    # Terapkan non-maximum suppression
    detection_list = []
    class_detection_counts = {class_name: 0 for class_name in classes}  # Reset deteksi per frame
    
    if bboxes:
        # NMS yang lebih toleran
        indices = cv2.dnn.NMSBoxes(bboxes, confidences, confidence_threshold, nms_threshold)
        
        if len(indices) > 0:
            indices = indices.flatten()
            for i in indices:
                x, y, w, h = bboxes[i]
                class_id = class_ids[i]
                class_name = classes[class_id] if class_id < len(classes) else "unknown"
                
                # Tambahkan ke daftar deteksi dan hitung berdasarkan kelas
                detection_list.append(([x, y, x+w, y+h], confidences[i], class_id))
                class_detection_counts[class_name] += 1
    
    # Perbarui tracker dengan deteksi saat ini
    tracks = tracker.update_tracks(detection_list, frame=frame)
    
    # Reset penghitung untuk setiap kelas
    track_counts = {class_name: 0 for class_name in classes}
    
    # Proses dan visualisasikan track aktif
    active_tracks = [track for track in tracks if track.is_confirmed()]
    
    for track in active_tracks:
        bbox = track.to_ltrb()  # Dapatkan koordinat kiri, atas, kanan, bawah
        x1, y1, x2, y2 = map(int, bbox)
        track_id = track.track_id
        
        # Dapatkan class_id dari track
        if hasattr(track, 'det_class') and track.det_class is not None:
            class_id = track.det_class
        else:
            class_id = 0  # Default ke person jika tidak ada informasi kelas
        
        class_name = classes[class_id] if class_id < len(classes) else "unknown"
        track_counts[class_name] += 1
        
        # Pastikan kotak berada dalam batas frame
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)
        
        # Ambil warna berdasarkan kelas
        color = colors[class_id] if class_id < len(colors) else [0, 255, 0]
        
        # Gambar kotak pembatas
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Tambahkan ID track dan nama kelas
        label = f"{class_name}: {track_id}"
        cv2.putText(frame, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Update penghitungan untuk kehalusan tampilan
    for class_name in classes:
        avg_counts[class_name].append(track_counts[class_name])
        if len(avg_counts[class_name]) > 15:  # Simpan 15 frame terakhir
            avg_counts[class_name].pop(0)
        if avg_counts[class_name]:
            object_counts[class_name] = int(sum(avg_counts[class_name]) / len(avg_counts[class_name]))
    
    # Tampilkan raw deteksi juga (sebelum tracking) dengan warna berbeda
    for i, ((x, y, w, h), conf) in enumerate(zip(bboxes, confidences)):
        if conf > 0.5:
            class_id = class_ids[i] if i < len(class_ids) else 0
            class_name = classes[class_id] if class_id < len(classes) else "unknown"
            color = colors[class_id] if class_id < len(colors) else [0, 0, 255]
            
            # Gambar kotak deteksi mentah dengan warna berdasarkan kelas
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 1)
    
    # Gambar informasi hitungan dengan latar belakang untuk visibilitas yang lebih baik
    # Hitung total deteksi mentah
    total_raw_detections = sum(class_detection_counts.values())
    
    # Gambar info panel
    y_offset = 10  # Naikkan agar tidak terlalu ke bawah

    cv2.rectangle(frame, (10, y_offset), (350, y_offset + 35 + (len(displayed_classes) * 25)), (0, 0, 0), -1)
    
    # Tampilkan total deteksi mentah
    cv2.putText(frame, f"Deteksi Mentah: {total_raw_detections}", (20, y_offset + 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Tampilkan jumlah untuk kelas yang dipilih
    for i, class_name in enumerate(displayed_classes):
        class_id = classes.index(class_name) if class_name in classes else -1
        if class_id >= 0:
            color = colors[class_id] if class_id < len(colors) else [0, 255, 255]
            y_pos = y_offset + 55 + (i * 25)
            cv2.putText(frame, f"{class_name}: {object_counts[class_name]}", 
                       (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Tampilkan informasi threshold
    cv2.putText(frame, f"Conf: {confidence_threshold:.2f}, NMS: {nms_threshold:.2f}", 
               (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # Tampilkan frame
    cv2.imshow("Deteksi Multi-Objek + Tracking (DeepSORT)", frame)
    
    # Keluar dengan menekan 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    # Tingkatkan/kurangi confidence threshold dengan '+'/'='/'p' dan '-'/'m'
    elif key == ord('p') or key == ord('=') or key == ord('+'):
        confidence_threshold = min(0.9, confidence_threshold + 0.05)
        print(f"Confidence threshold ditingkatkan menjadi {confidence_threshold:.2f}")
    elif key == ord('-') or key == ord('m'):
        confidence_threshold = max(0.1, confidence_threshold - 0.05)
        print(f"Confidence threshold diturunkan menjadi {confidence_threshold:.2f}")
    # Tingkatkan/kurangi NMS threshold dengan ']' dan '['
    elif key == ord(']'):
        nms_threshold = min(0.9, nms_threshold + 0.05)
        print(f"NMS threshold ditingkatkan menjadi {nms_threshold:.2f}")
    elif key == ord('['):
        nms_threshold = max(0.1, nms_threshold - 0.05)
        print(f"NMS threshold diturunkan menjadi {nms_threshold:.2f}")

# Lepaskan sumber daya
print("Membersihkan sumber daya...")
cap.release()
cv2.destroyAllWindows()
print("Selesai.")