import cv2
import numpy as np
import time
import sys

def apply_convolution(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    assert kh == kw and kh % 2 == 1, "Kernel must be odd-sized and square"
    pad = kh // 2
    x = np.pad(img.astype(np.float32), ((pad, pad), (pad, pad)), mode='edge')
    H, W = img.shape
    s0, s1 = x.strides
    windows = np.lib.stride_tricks.as_strided(
        x, shape=(H, W, kh, kw), strides=(s0, s1, s0, s1), writeable=False
    )
    k = kernel[::-1, ::-1].astype(np.float32)
    out = np.tensordot(windows, k, axes=([2, 3], [0, 1]))
    return np.clip(out, 0, 255).astype(np.uint8)

KERNELS = {
    '0: none'     : np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=np.float32),
    '1: box 3x3'  : (1/9.0) * np.ones((3,3), dtype=np.float32),
    '2: gaussian' : (1/16.0) * np.array([[1,2,1],[2,4,2],[1,2,1]], dtype=np.float32),
    '3: sobel X'  : np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=np.float32),
    '4: sobel Y'  : np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32),
    '5: sharpen'  : np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32),
    '6: emboss'   : np.array([[-2,-1,0],[-1,1,1],[0,1,2]], dtype=np.float32),
}

def probe_cameras(max_index=6, backend=cv2.CAP_AVFOUNDATION):
    found = []
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i, backend)
        ok = cap.isOpened()
        if ok:
            ok, _ = cap.read()
        if ok:
            found.append(i)
        cap.release()
    return found

def open_camera(index, width=320, height=240, backend=cv2.CAP_AVFOUNDATION):
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index, cv2.CAP_ANY)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap

def main():
    print("Scanning for cameras (this helps if you want to use your iPhone/Continuity Camera)…")
    candidates = probe_cameras()
    if not candidates:
        print("No camera found.")
        sys.exit(1)

    print("\nAvailable camera indices:", candidates)
    try:
        sel = int(input(f"Pick a camera index from {candidates}: ").strip())
    except Exception:
        sel = candidates[0]
        print(f"Defaulting to {sel}")

    cap = open_camera(sel, width=424, height=240)

    choice_key = '0'
    kernel_name = '0: none'
    emboss_bias = 128

    window_name = "Webcam (left: color, right: filtered) — keys 0..6, q=quit"  
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    prev_time = time.time()
    frame_count = 0
    fps = 0.0

    print("Hotkeys: 0..6 switch filters | c = cycle cameras | q/ESC = quit")
    print("Filters:", ", ".join(KERNELS.keys()))

    cam_list = candidates
    cam_idx_in_list = cam_list.index(sel) if sel in cam_list else 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame grab failed; trying to re-open camera…")
                cap.release()
                cap = open_camera(cam_list[cam_idx_in_list])
                continue

            color_left = frame

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if choice_key == '0':
                filtered_gray = gray
            else:
                kernel_name = [k for k in KERNELS if k.startswith(choice_key)][0]
                ker = KERNELS[kernel_name]
                filtered_gray = apply_convolution(gray, ker)
                if choice_key == '6':
                    filtered_gray = np.clip(filtered_gray.astype(np.int16) + emboss_bias, 0, 255).astype(np.uint8)

            filtered_bgr = cv2.cvtColor(filtered_gray, cv2.COLOR_GRAY2BGR)       

            # Side-by-side view
            both = np.hstack((color_left, filtered_bgr))                       

            # FPS
            frame_count += 1
            now = time.time()
            if now - prev_time >= 0.5:
                fps = frame_count / (now - prev_time)
                prev_time = now
                frame_count = 0

            overlay = both.copy()
            cv2.putText(overlay, f"Filter: {kernel_name}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(overlay, f"FPS: {fps:.1f}", (10, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

            cv2.imshow(window_name, overlay)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):
                break

            if ord('0') <= key <= ord('6'):
                choice_key = chr(key)
                kernel_name = [k for k in KERNELS if k.startswith(choice_key)][0]

            if key == ord('c'):
                cam_idx_in_list = (cam_idx_in_list + 1) % len(cam_list)
                cap.release()
                cap = open_camera(cam_list[cam_idx_in_list], width=424, height=240)
                print(f"Switched to camera index {cam_list[cam_idx_in_list]}")

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
