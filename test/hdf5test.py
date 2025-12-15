# python
import h5py, numpy as np, cv2
p = "D:\File\File\data_analyse_hdf\episode1.hdf5"
with h5py.File(p, "r") as f:
    b = f["observation"]["front_camera"]["rgb"][123]        # bytes-like object
    img = cv2.imdecode(np.frombuffer(b, dtype=np.uint8), cv2.IMREAD_COLOR)
    cv2.imshow("preview", img)
    cv2.waitKey(0)  # 或 cv2.waitKey(0) 来阻塞直到按键
# 显示结束后可销毁窗口
    cv2.destroyAllWindows()
    # img 现在是 (H,W,3) 的 numpy array，可以用 cv2.imshow 或保存为 png