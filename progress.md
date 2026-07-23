# Tiến độ Dự án: Computer Vision F&B

Tài liệu này tổng hợp trạng thái và tiến độ thực tế của dự án dựa trên kế hoạch chi tiết.

---

## 📊 Tổng quan dự án

| Chỉ số | Trạng thái |
| :--- | :---: |
| **Tổng số công việc** | **15** |
| **Hoàn thành** | **5** (33.3%) |
| **Đang tiến hành** | **1** (6.7%) |
| **Chưa bắt đầu** | **9** (60.0%) |
| **Tổng số ngày dự kiến** | **38 ngày** |

### 📈 Biểu đồ tiến độ
`██████████░░░░░░░░░░░░░░░░░░░░░░` (33.3% Hoàn thành)

---

## 🗓️ Chi tiết tiến độ theo Giai đoạn (Phase)

### Phase 1: PoC (Mac Mini M4)
*Giai đoạn phát triển Proof of Concept (PoC) trên nền tảng Mac Mini M4.*

| ID | Tên công việc | Mô tả chi tiết / Yêu cầu kỹ thuật | Ngày dự kiến | Trạng thái | Ghi chú |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **1.1** | Cài đặt môi trường | Cài đặt Python, PyTorch (hỗ trợ MPS cho Apple Silicon) và thư viện Ultralytics. | 1 | **Hoàn thành** ✅ | Đã khởi tạo Conda `aibox` thành công. |
| **1.2** | Detection & Tracking | Đọc video quán cafe. Chạy model yolov8n.pt (hoặc bản s) kết hợp thuật toán ByteTrack. | 2 | **Hoàn thành** ✅ | Đã chạy thành công YOLOv8n + ByteTrack trong `track_heatmap.py`. |
| **1.3** | Trích xuất tọa độ gót chân | Xử lý bounding box để lấy vị trí chạm đất: $X = (x_{min} + x_{max})/2, Y = y_{max}$. | 1 | **Hoàn thành** ✅ | Đã trích xuất điểm chân $X_{center}, Y_{max}$ để làm điểm neo vị trí. |
| **1.4** | Tích lũy ma trận 2D theo ngày | Tạo ma trận 2D bằng kích thước khung hình. Cộng +1 vào ô $(X, Y)$ mỗi khi có người xuất hiện. | 1 | **Hoàn thành** ✅ | Ma trận mật độ 2D `density` tích lũy chính xác từng frame. |
| **1.5** | Render Heatmap | Dùng `cv2.applyColorMap` hoặc bộ lọc KDE làm mượt mây nhiệt. Đè lớp ảnh lên video gốc (alpha blend). | 2 | **Hoàn thành** ✅ | Tích hợp Gaussian Filter + colormap `INFERNO` + xuất video & ảnh PNG. |
| **1.6** | Demo với 1 camera thực tế | Kết nối luồng RTSP từ camera thật, đánh giá độ chính xác và FPS của toàn bộ pipeline Phase 1. | 2 | **Đang tiến hành** 🔄 | Cột mốc báo cáo kết quả Phase 1 |

---

### Phase 2: Ánh xạ 2D & Phân tích
*Ánh xạ không gian và phân tích hành vi khách hàng.*

| ID | Tên công việc | Mô tả chi tiết / Yêu cầu kỹ thuật | Ngày dự kiến | Trạng thái | Ghi chú |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **2.1** | Perspective Transformation (Homography) | Ánh xạ tọa độ khách từ góc camera chéo xuống bản đồ mặt bằng 2D (Floorplan) từ trên xuống. | 3 | **Chưa bắt đầu** ⏳ | Chuẩn bị giao diện cho người quản lý |
| **2.2** | Phân tích thời gian lưu lại (Dwell Time) | Đo lường thời gian tồn tại của các ID để phân biệt khách take-away và khách ngồi làm việc. | 3 | **Chưa bắt đầu** ⏳ | |
| **2.3** | Đóng gói Pipeline | Tối ưu hóa mã nguồn thành các module độc lập, dọn dẹp code để chuẩn bị chuyển sang môi trường nhúng. | 2 | **Chưa bắt đầu** ⏳ | |

---

### Phase 3: Triển khai Edge (Rockchip)
*Tối ưu hóa và đưa giải pháp lên phần cứng nhúng RK3588.*

| ID | Tên công việc | Mô tả chi tiết / Yêu cầu kỹ thuật | Ngày dự kiến | Trạng thái | Ghi chú |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **3.1** | Thiết lập OS & Môi trường nhúng | Cài đặt Linux (Ubuntu) và các dependencies cơ bản lên bo mạch Rockchip RK3588 (như Rock 5B). | 2 | **Chưa bắt đầu** ⏳ | |
| **3.2** | Kiểm thử trên Rockchip (không NPU) | Đưa pipeline từ Mac M4 sang chạy thử trên CPU/GPU của Rockchip để đánh giá chênh lệch hiệu năng. | 2 | **Chưa bắt đầu** ⏳ | |
| **3.3** | Tích hợp NPU DeepX DX-M1 | Cài đặt driver PCIe và môi trường DXNN SDK. Chuyển đổi (quantize) mô hình sang chuẩn `.dxnn` (INT8). | 4 | **Chưa bắt đầu** ⏳ | Yêu cầu calibration dataset |
| **3.4** | Tối ưu hóa dị thể (Heterogeneous) | Phân bổ tác vụ: CPU Rockchip decode video, NPU DeepX xử lý Vision/Tracking để đạt mức Real-time. | 4 | **Chưa bắt đầu** ⏳ | |

---

### Phase 4: Hoàn thiện giải pháp F&B
*Phát triển các tính năng giá trị gia tăng & kết nối hệ thống nghiệp vụ.*

| ID | Tên công việc | Mô tả chi tiết / Yêu cầu kỹ thuật | Ngày dự kiến | Trạng thái | Ghi chú |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **4.1** | People Counting & POS | Tích hợp đếm người (IN/OUT) tại cửa, thiết kế API đẩy dữ liệu đối chiếu với máy POS. | 4 | **Chưa bắt đầu** ⏳ | Đo lường Conversion Rate |
| **4.2** | Giám sát trạng thái bàn & Hàng đợi | Bổ sung logic quản lý hàng đợi và tình trạng bàn trống/chưa dọn. Cấu hình hệ thống cảnh báo qua Telegram/Zalo. | 5 | **Chưa bắt đầu** ⏳ | Mở rộng tính năng giá trị cao |

---
*Tài liệu được cập nhật tự động vào ngày 23/07/2026.*
