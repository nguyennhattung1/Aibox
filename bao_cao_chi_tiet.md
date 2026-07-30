# 📄 BÁO CÁO CHI TIẾT SẢN PHẨM & HƯỚNG DẪN VẬN HÀNH
## Hệ Thống Computer Vision Cho Ngành F&B (Heatmap & Tracking 2D)

---

## 1. Tổng Quan Hệ Thống

Hệ thống cung cấp giải pháp phân tích mật độ di chuyển (Heatmap) và theo dõi luồng di chuyển khách hàng (Customer Tracking) dựa trên thị giác máy tính (Computer Vision), phục vụ tối ưu hóa vận hành nhà hàng/quán cafe.

### Các Tính Năng Nổi Bật:
* **Detection & Tracking**: Sử dụng YOLOv8n kết hợp thuật toán ByteTrack để phát hiện và gán ID theo dõi từng khách hàng.
* **Foot-point Extraction**: Trích xuất tọa độ gót chân (chạm đất) $X_{center}, Y_{max}$ làm điểm mốc thực tế thay vì trung tâm bounding box.
* **Homography (Perspective Transformation)**: Chuyển đổi tọa độ từ góc quay chéo của Camera sang bản đồ mặt bằng 2D nhìn từ trên xuống (Bird's-Eye View).
* **2D Floorplan Heatmap**: Tích lũy mật độ di chuyển trực tiếp trên Floorplan qua toàn bộ luồng Video thực tế.
* **RTSP Camera Stream Integration (Task 1.6)**: Kết nối trực tiếp luồng camera IP/RTSP thời gian thực qua giao thức TCP, tự động đo đạc latency, FPS và xuất báo cáo đánh giá hiệu năng Pipeline Phase 1.

---

## 2. Danh Sách & Hướng Dẫn Chạy Các File Mã Nguồn

### Môi trường khuyến nghị:
Mở Terminal và kích hoạt môi trường Anaconda trước khi chạy:
```bash
conda activate aibox
cd /Users/tungnguyen/Documents/AI_box
```

---

### 🟢 File 1: `track_heatmap.py`
* **Mục đích**: Chạy phát hiện (Detection), gán ID (Tracking) người dùng và vẽ mây nhiệt Heatmap trực tiếp đè lên khung hình Camera.
* **Công nghệ**: YOLOv8n + ByteTrack + Gaussian KDE + Colormap `INFERNO`.
* **Đầu ra**: 
  * Video đè Heatmap + Bounding Box + Track ID trong `outputs/<stem>_tracked.mp4`.
  * Ảnh Heatmap tổng kết trong `outputs/<stem>_heatmap.png`.

#### 💻 Cách chạy:
```bash
# 1. Chạy mặc định với các video mẫu:
python track_heatmap.py

# 2. Chạy với video tùy chỉnh:
python track_heatmap.py --videos "Videos/C01 South First Indoor 02.mp4"

# 3. Chạy nhiều video và chỉ định thư mục xuất kết quả:
python track_heatmap.py --videos "Videos/C01 South First Indoor 02.mp4" "Videos/CCTV Indoor Lobby 01.mp4" --output outputs
```

---

### 🟢 File 2: `test_homography.py`
* **Mục đích**: Thử nghiệm và kiểm thử toán học thuật toán Homography (Perspective Transformation) ánh xạ tọa độ 2D từ Camera View sang Floorplan 2D trên 1 frame hình mẫu.
* **Đầu ra**: 
  * Ảnh so sánh 2 panel Side-by-Side trong `outputs/<stem>_test_homography.png`.

#### 💻 Cách chạy:
```bash
# 1. Chạy kiểm thử mặc định:
python test_homography.py

# 2. Chạy kiểm thử với file video/ảnh khác:
python test_homography.py "Videos/C01 North First Indoor.mp4" outputs
```

---

### 🟢 File 3: `interactive_homography.py`
* **Mục đích**: Giao diện tương tác GUI cho phép người dùng dùng chuột chọn 4 điểm góc sàn (ROI) trên ảnh Camera thực tế để tính toán ma trận Homography $H$.
* **Hướng dẫn thao tác trên GUI**:
  * **Click chuột trái**: Chọn lần lượt 4 điểm theo thứ tự:
    1. Top-Left (TL) - Trên Trái
    2. Top-Right (TR) - Trên Phải
    3. Bottom-Right (BR) - Dưới Phải
    4. Bottom-Left (BL) - Dưới Trái
  * **Phím `c`**: Xác nhận 4 điểm & tính ma trận Homography.
  * **Phím `r`**: Reset/Xóa các điểm đã chọn để chọn lại.
  * **Phím `q` / ESC**: Thoát ứng dụng.

#### 💻 Cách chạy:
```bash
# 1. Chạy giao diện chọn điểm ROI tương tác:
python interactive_homography.py "Videos/C01 South First Indoor 02.mp4"
```

---

### 🟢 File 4: `homography_video_heatmap.py` *(Task 2.1b - Full Video Pipeline)*
* **Mục đích**: File tổng hợp hoàn chỉnh chạy **Full Video Stream**:
  1. Cho phép người dùng click 4 điểm ROI góc sàn trên frame đầu tiên (hoặc fallback tự động nếu không có GUI).
  2. Tính ma trận $H$.
  3. Chạy YOLOv8 + ByteTrack theo từng frame video.
  4. Trích xuất vị trí chân và dùng $H$ ánh xạ sang tọa độ mặt bằng 2D Floorplan.
  5. Tích lũy mật độ Heatmap trên 2D Floorplan.
  6. Xuất video Side-by-Side (Bên trái: Camera View + Track ID | Bên phải: 2D Floorplan Heatmap).
* **Đầu ra**:
  * Video kết quả: `outputs/<stem>_floorplan_heatmap.mp4`.
  * Ảnh PNG Heatmap tổng kết trên Floorplan: `outputs/<stem>_floorplan_heatmap_final.png`.

#### 💻 Cách chạy:
```bash
# 1. Xem danh sách video có sẵn trong thư mục Videos/:
python homography_video_heatmap.py --list-videos

# 2. Chạy xử lý cho 1 video cụ thể (mở GUI chọn 4 điểm ROI trước):
python homography_video_heatmap.py --video "Videos/C01 South First Indoor 02.mp4"

# 3. Chạy tự động lần lượt cho tất cả các video trong thư mục Videos/:
python homography_video_heatmap.py --all-videos

# 4. Tùy chỉnh thư mục đầu ra:
python homography_video_heatmap.py --video "Videos/CCTV Indoor Lobby 01.mp4" --output "outputs"
```

---

### 🟢 File 5: `rtsp_pipeline_demo.py` *(Task 1.6 - Camera RTSP Demo & Benchmark)*
* **Mục đích**: Kết nối luồng RTSP camera IP thực tế, chạy toàn bộ Pipeline Phase 1 (Detection + ByteTrack + Heel Extraction + Live Heatmap) và xuất báo cáo đo đạc hiệu năng FPS / Latency chi tiết.
* **Đầu ra (mặc định lưu tại thư mục `temp/`)**:
  * Video đã tracking & đè heatmap: `temp/rtsp_phase1_demo_<timestamp>.mp4`.
  * Ảnh Heatmap mật độ đơn lẻ: `temp/rtsp_phase1_heatmap_<timestamp>.png`.
  * Ảnh Báo cáo Tổng kết Dashboard: `temp/rtsp_phase1_summary_<timestamp>.png`.

#### 💻 Cách chạy:
```bash
# 1. Chạy đánh giá Pipeline với camera RTSP mặc định trong 30 giây:
python rtsp_pipeline_demo.py --duration 30 --output temp

# 2. Chạy với URL camera RTSP tùy chỉnh:
python rtsp_pipeline_demo.py --rtsp "rtsp://admin:password@192.168.1.21:554/cam/realmonitor?channel=1&subtype=1" --duration 60

# 3. Chạy hiển thị cửa sổ xem trực tiếp GUI:
python rtsp_pipeline_demo.py --display
```

#### 📊 Kết quả Benchmark thực tế trên Mac Mini M4 (Camera IMOU Sub-stream 640x480):
* **Tốc độ đọc luồng RTSP (Stream FPS)**: ~40 - 43 FPS (Rất ổn định qua TCP).
* **Tốc độ xử lý Pipeline tổng thể**: **8.37 FPS** (~117.9 ms latency/frame).
* **Phân rã độ trễ (Latency Breakdown)**:
  * Capture & Video Decode: `0.34 ms`
  * YOLOv8n + ByteTrack Inference: `100.89 ms`
  * Gaussian KDE & Heatmap Render: `16.67 ms`
* **Độ chính xác Tracking**: Nhận diện & theo dõi chính xác 7/7 ID người di chuyển trong góc quay phòng thử nghiệm.

---

## 3. Cấu Trúc Thư Mục Dự Án

```
AI_box/
├── Videos/                         # Thư mục chứa video nguồn
│   ├── C01 North First Indoor.mp4
│   ├── C01 South First Indoor 02.mp4
│   └── CCTV Indoor Lobby 01.mp4
├── outputs/                        # Thư mục chứa kết quả xuất chính thức (Video & PNG)
├── temp/                           # Thư mục chứa các file nháp & báo cáo tạm (được gitignore)
├── track_heatmap.py                # Pipeline Phase 1: Detection + Heatmap góc Camera
├── test_homography.py              # Test thuật toán Homography
├── interactive_homography.py       # UI tương tác chọn 4 điểm ROI
├── homography_video_heatmap.py     # Pipeline Phase 2 (Task 2.1b): Full Video 2D Floorplan Heatmap
├── rtsp_pipeline_demo.py           # Pipeline Phase 1 (Task 1.6): Demo camera RTSP & Benchmark
├── progress.md                     # Tài liệu theo dõi tiến độ dự án
└── bao_cao_chi_tiet.md             # Báo cáo chi tiết & hướng dẫn vận hành này
```

---

*Tài liệu được cập nhật ngày 30/07/2026.*

