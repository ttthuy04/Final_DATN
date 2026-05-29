import argparse
import glob
import os

import cv2


def get_next_index(save_dir, label):
    """
    Tìm index tiếp theo dựa trên các file đã có trong thư mục con của nhãn.
    Ví dụ: đã có sleep/sleep_1.mp4, thì sẽ trả về 2.
    """
    label_dir = os.path.join(save_dir, label)
    if not os.path.exists(label_dir):
        return 1

    existing_files = glob.glob(os.path.join(label_dir, f"{label}_*.mp4"))
    if not existing_files:
        return 1

    indices = []
    for f in existing_files:
        basename = os.path.basename(f)
        try:
            num_str = basename.replace(".mp4", "").split("_")[-1]
            num = int(num_str)
            indices.append(num)
        except ValueError:
            pass

    return max(indices) + 1 if indices else 1


def main(video_path, save_dir="data/labeled"):
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Không thể mở video: {video_path}")
        print("Vui lòng kiểm tra lại đường dẫn video gốc.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps:
        fps = 30.0

    recording = False
    frames = []
    current_roi = None

    print("\n" + "=" * 50)
    print("🎬 CÔNG CỤ CẮT VIDEO HÀNH VI SINH VIÊN")
    print("=" * 50)
    print("📌 HƯỚNG DẪN TRÊN CỬA SỔ VIDEO:")
    print(" - Nhấn 's' : Dừng video lại để khoanh vùng đối tượng muốn ghi.")
    print(" - Nhấn 'e' : Kết thúc ghi hình. Bắt buộc phải có.")
    print(" - Nhấn 'f' : Tua tới 5 giây.")
    print(" - Nhấn 'b' : Tua lùi 5 giây.")
    print(" - Nhấn '1', '2', '3': Chỉnh tốc độ phát (1x, 2x, 4x).")
    print(" - Nhấn 'q' : Thoát phần mềm (Quit)\n")
    print("Lưu ý: Sau khi nhấn 'e', hãy sang Terminal (màn hình console) nhập Tên Nhãn.")
    print("=" * 50 + "\n")

    speed_multiplier = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Đã phát hết video.")
            break

        display_frame = frame.copy()

        if recording:
            x, y, w, h = current_roi
            cropped_frame = frame[int(y) : int(y + h), int(x) : int(x + w)]
            frames.append(cropped_frame)
            cv2.putText(
                display_frame,
                "DANG GHI... (Nhan 'e' de dung)",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            cv2.rectangle(
                display_frame,
                (int(x), int(y)),
                (int(x + w), int(y + h)),
                (0, 0, 255),
                4,
            )
        else:
            cv2.putText(
                display_frame,
                f"Dang phat ({speed_multiplier}x). Nhan 's' de bat dau cat.",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                display_frame,
                "Phim: 'f' tua toi, 'b' tua lui, '1','2','3' chinh toc do",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )

        cv2.imshow("Video Player", display_frame)

        delay = int(1000 / (fps * speed_multiplier))
        if delay < 1:
            delay = 1
        key = cv2.waitKey(delay) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("1"):
            speed_multiplier = 1
            print(">> [SPEED] Tốc độ 1x")
        elif key == ord("2"):
            speed_multiplier = 2
            print(">> [SPEED] Tốc độ 2x")
        elif key == ord("3"):
            speed_multiplier = 4
            print(">> [SPEED] Tốc độ 4x")
        elif key == ord("f") and not recording:
            skip_frames = int(fps * 5)
            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame + skip_frames)
            print(">> [SKIP] Đã tua tới 5 giây.")
        elif key == ord("b") and not recording:
            skip_frames = int(fps * 5)
            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current_frame - skip_frames))
            print(">> [SKIP] Đã tua lùi 5 giây.")
        elif key == ord("s") and not recording:
            print(
                "\n>> Tạm dừng video... Hãy dùng chuột kéo một ô vuông bao quanh "
                "sinh viên mục tiêu trên video. Khoanh xong thì nhấn phím SPACE "
                "hoặc ENTER để xác nhận."
            )
            roi = cv2.selectROI(
                "Video Player", display_frame, fromCenter=False, showCrosshair=True
            )
            if roi[2] > 0 and roi[3] > 0:
                recording = True
                current_roi = roi
                x, y, w, h = current_roi
                frames = [frame[int(y) : int(y + h), int(x) : int(x + w)]]
                print(">> [ĐANG GHI] Bắt đầu bắt khung hình. Nhấn 'e' khi kết thúc hành động.")
            else:
                print(">> [ĐÃ HỦY] Không có vùng nào được chọn. Video tiếp tục phát.")
        elif key == ord("e") and recording:
            recording = False
            print(f">> [ĐÃ DỪNG] Thu được {len(frames)} khung hình.")

            cv2.putText(
                display_frame,
                "DA TAM DUNG - XEM TERMINAL DE NHAP TEN!",
                (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                3,
            )
            cv2.imshow("Video Player", display_frame)
            cv2.waitKey(1)

            try:
                label = input(
                    "🏷️ Nhập nhãn cho hành động này (vd: sleep, use_phone): "
                ).strip()
            except EOFError:
                print("❌ Không đọc được input từ terminal. Bỏ qua clip này, tiếp tục phát video.")
                label = ""

            label = label.lower().replace(" ", "_")

            if label:
                label_dir = os.path.join(save_dir, label)
                os.makedirs(label_dir, exist_ok=True)

                idx = get_next_index(save_dir, label)
                filename = f"{label}_{idx}.mp4"
                save_path = os.path.join(label_dir, filename)

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                w_roi, h_roi = int(current_roi[2]), int(current_roi[3])
                out = cv2.VideoWriter(save_path, fourcc, fps, (w_roi, h_roi))

                for f in frames:
                    out.write(f)
                out.release()
                print(f"✅ Đã phân loại thành công! File lưu tại: {save_path}\n")
            else:
                print("❌ Tên nhãn trống! Đã hủy bỏ kết quả vừa rồi.\n")

            frames = []

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Công cụ cắt video gán nhãn")
    parser.add_argument(
        "--video",
        type=str,
        default="data/raw/video_demo.mp4",
        help="Đường dẫn đến file video gốc dài",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="data/labeled",
        help="Thư mục lưu các clip sau khi cắt",
    )

    args = parser.parse_args()
    main(args.video, args.save_dir)
