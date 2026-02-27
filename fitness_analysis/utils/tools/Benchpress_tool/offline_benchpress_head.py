# offline_benchpress_head_fast.py                                                                               # 檔名與用途：三線程流水線（解碼/推論/寫入），批次推論，最小I/O，進一步加速
import os, cv2, time, threading, queue, numpy as np, torch                                                      # 基本匯入
from ultralytics import YOLO                                                                                    # YOLO
CONNECTIONS = [(0,1),(0,2),(2,4),(1,3),(3,5)]                                                                   # 骨架連線

def _pick_best_person(kps):                                                                                     # 在多人的情況挑一個人
    if kps is None or getattr(kps,"xy",None) is None or len(kps.xy)==0:                                         # 無偵測
        return None                                                                                              # 回 None
    xy = kps.xy                                                                                                  # 取 xy
    if hasattr(xy,"cpu"): xy = xy.cpu().numpy()                                                                  # tensor→numpy
    conf = getattr(kps,"conf",None)                                                                              # 嘗試信心
    if conf is not None:
        if hasattr(conf,"cpu"): conf = conf.cpu().numpy()                                                        # tensor→numpy
        idx = int(np.argmax(conf.sum(axis=1)))                                                                   # 總分最高者
    else:
        idx = 0                                                                                                  # 無信心值取第1人
    return xy[idx]                                                                                               # 回該人座標(K,2)

def _draw_skeleton(frame, pts, r=3, th=2):                                                                       # 畫骨架
    for (x,y) in pts:                                                                                            # 逐點
        if x>0 and y>0: cv2.circle(frame,(int(x),int(y)),r,(0,255,0),-1)                                         # 畫點
    for (s,e) in CONNECTIONS:                                                                                    # 逐邊
        if s<len(pts) and e<len(pts):                                                                            # 邊界
            xs,ys = pts[s]; xe,ye = pts[e]                                                                       # 端點
            if xs>0 and ys>0 and xe>0 and ye>0:                                                                  # 合法
                cv2.line(frame,(int(xs),int(ys)),(int(xe),int(ye)),(255,0,0),th)                                 # 畫線

def _resize_long(frame, target):                                                                                 # 以長邊縮放
    h,w = frame.shape[:2]                                                                                        # 尺寸
    scale = float(target)/max(h,w)                                                                               # 縮放比
    if scale>=1.0: return frame,1.0                                                                              # 無需縮
    nw,nh = int(w*scale), int(h*scale)                                                                           # 新尺寸
    return cv2.resize(frame,(nw,nh),interpolation=cv2.INTER_AREA), scale                                         # 回縮圖與比

def run_offline(folder, model_path="./model/benchpress/head_model/yolo11n.pt",
                conf=0.5, rotate180=False, device=None, BATCH=32, IMG=448, DRAW=True, PRE_RESIZE=True,
                classes=None):                                                                                   # 主流程
    video_path = os.path.join(folder,"original_vision3.avi")                                                      # 來源影片
    txt_path   = os.path.join(folder,"yolo_skeleton.txt")                                                         # TXT 路徑
    out_video  = os.path.join(folder,"vision3_drawed.avi")                                                        # 影片輸出

    if not os.path.exists(video_path): raise FileNotFoundError(video_path)                                        # 檢查

    cv2.setNumThreads(0)                                                                                          # OpenCV不搶緒
    torch.backends.cudnn.benchmark = True                                                                         # cuDNN加速

    use_cuda = torch.cuda.is_available() if device is None else device.startswith("cuda")                         # CUDA判斷
    device   = "cuda" if (device is None and torch.cuda.is_available()) else (device or "cpu")                    # 裝置設定

    model = YOLO(model_path)                                                                                      # 載 YOLO
    if use_cuda:
        model.to(device)                                                                                          # 移 GPU
        try: model.fuse()                                                                                         # fuse層
        except: pass                                                                                               # 不支援略過
        try: model.half()                                                                                         # 半精度
        except: pass                                                                                               # 不支援略過

    cap = cv2.VideoCapture(video_path)                                                                            # 開影片
    if not cap.isOpened(): raise RuntimeError(f"cannot open: {video_path}")                                       # 失敗拋錯

    fps = cap.get(cv2.CAP_PROP_FPS) or 29.0                                                                       # FPS
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))                                                               # 寬
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))                                                              # 高

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")                                                                      # 編碼
    writer = cv2.VideoWriter(out_video, fourcc, fps, (width,height))                                              # 建立Writer

    infer_kwargs = dict(conf=conf, iou=0.5, verbose=False, imgsz=IMG, max_det=1)                                  # 推論設定
    if classes is not None: infer_kwargs["classes"] = classes                                                      # 限定類別

    in_q  = queue.Queue(maxsize=BATCH*4)                                                                          # 解碼→推論佇列
    out_q = queue.Queue(maxsize=BATCH*4)                                                                          # 推論→寫檔佇列
    stop_decode = threading.Event()                                                                               # 解碼結束旗標
    stop_infer  = threading.Event()                                                                               # 推論結束旗標
    txt_lines = []                                                                                                # 暫存TXT內容
    fidx = 0                                                                                                      # 幀計數
    t0 = time.time()                                                                                              # 起始時間

    def decoder():                                                                                                # 解碼線
        nonlocal fidx                                                                                             # 用外層fidx
        while True:                                                                                               # 讀幀
            ok, frame = cap.read()                                                                                # 讀一幀
            if not ok: break                                                                                      # 結束
            if rotate180: frame = cv2.rotate(frame, cv2.ROTATE_180)                                               # 旋轉
            feed, scale = (frame,1.0) if not PRE_RESIZE else _resize_long(frame, IMG)                             # 預縮
            in_q.put((fidx, frame, feed, scale), block=True)                                                       # 丟進佇列
            fidx += 1                                                                                             # 累計
        stop_decode.set()                                                                                         # 標記結束

    def inferer():                                                                                                # 推論線
        batch_idx, batch_raw, batch_feed, batch_scale = [], [], [], []                                            # 批容器
        while not (stop_decode.is_set() and in_q.empty()):                                                        # 有資料或未結束
            try:
                item = in_q.get(timeout=0.02)                                                                     # 取一筆
            except queue.Empty:
                item = None                                                                                       # 取不到
            if item is not None:
                idx, raw, feed, scale = item                                                                      # 解包
                batch_idx.append(idx); batch_raw.append(raw); batch_feed.append(feed); batch_scale.append(scale)  # 收集
            # 夠一批或解碼結束就推論                                                                              
            if (len(batch_feed)>=BATCH) or (stop_decode.is_set() and batch_feed):
                # 注意：Ultralytics 自動處理 numpy list；CUDA時亦可                                                   #
                results = model.predict(batch_feed, device=device, **infer_kwargs)                                 # 批推論
                for r, ri, rr, sc in zip(results, batch_idx, batch_raw, batch_scale):                              # 配對
                    out_q.put((ri, rr, r, sc), block=True)                                                         # 丟給寫檔
                batch_idx.clear(); batch_raw.clear(); batch_feed.clear(); batch_scale.clear()                      # 清批
        stop_infer.set()                                                                                          # 標記結束

    def writer_thread():                                                                                          # 寫檔線
        nonlocal txt_lines                                                                                        # 用外層list
        handled = 0                                                                                               # 已處理數
        while not (stop_infer.is_set() and out_q.empty()):                                                        # 等資料
            try:
                idx, raw, res, scale = out_q.get(timeout=0.05)                                                    # 取結果
            except queue.Empty:
                continue                                                                                          # 無則輪詢
            kps = res.keypoints if res is not None else None                                                      # 關鍵點
            best = _pick_best_person(kps)                                                                          # 最佳人
            if best is None:                                                                                       # 無偵測
                txt_lines.append(f"Frame {idx}: [[]]")                                                             # 改成空的雙層清單 [[ ]] 格式
            else:
                if scale != 1.0: best = best / scale                                                               # 還原座標到原圖尺度
                pts = [(float(best[j, 0]), float(best[j, 1])) for j in range(min(6, best.shape[0]))]              # 取前6點不做四捨五入
                txt_lines.append(f"Frame {idx}: {[pts]}")                                                          # 外層再包一層→ [[(x,y),...]]
                if DRAW: _draw_skeleton(raw, pts, r=3, th=2)                                                       # 視覺化骨架
            writer.write(raw)                                                                                      # 寫影片
            handled += 1                                                                                           # 累計

        with open(txt_path, "w") as f:                                                                             # 開啟TXT
            f.write("\n".join(txt_lines))                                                                          # 一次寫入


    th_dec = threading.Thread(target=decoder, daemon=True)                                                         # 解碼執行緒
    th_inf = threading.Thread(target=inferer, daemon=True)                                                         # 推論執行緒
    th_wrt = threading.Thread(target=writer_thread, daemon=True)                                                   # 寫檔執行緒
    th_dec.start(); th_inf.start(); th_wrt.start()                                                                 # 啟動
    th_dec.join(); th_inf.join(); th_wrt.join()                                                                    # 等待

    cap.release(); writer.release()                                                                                # 關資源
    spent = time.time()-t0                                                                                         # 耗時
    print(f"完成，共 {fidx} 幀，fps={fps:.2f}，耗時={spent:.1f}s，裝置={device}，batch={BATCH}，img={IMG}")           # 總結
    print(f"TXT: {txt_path}")                                                                                      # TXT 路徑
    print(f"Video: {out_video}")                                                                                   # 影片路徑


if __name__ == "__main__":                                                                                             # 程式入口
    import sys                                                                                                         # 取參數用
    recordings_dir = r"C:/Users/92A27/benchpress/recordings"                                                           # 根目錄（預設）
    if not os.path.exists(recordings_dir): raise FileNotFoundError(recordings_dir)                                     # 檢查根目錄存在

    # === 優先級：CLI 參數 > 自動取最新 ===                                                                             # 說明優先順序
    if len(sys.argv) >= 2:                                                                                             # 有傳資料夾參數
        base_path = sys.argv[1]                                                                                        # 取第一個參數
        if not os.path.isdir(base_path):                                                                               # 參數需為資料夾
            raise FileNotFoundError(f"指定的資料夾不存在：{base_path}")                                                  # 拋錯提示
    else:                                                                                                              # 沒傳參數
        subs = [os.path.join(recordings_dir, d) for d in os.listdir(recordings_dir)                                    # 列子資料夾
                if os.path.isdir(os.path.join(recordings_dir, d))]                                                     # 僅取資料夾
        if not subs:                                                                                                   # 無子資料夾
            raise FileNotFoundError("recordings 下沒有子資料夾可處理，且未提供參數")                                    # 拋錯
        base_path = max(subs, key=os.path.getmtime)                                                                    # 取最新

    print(f"處理資料夾: {base_path}")                                                                                  # 明確顯示本次實際處理的資料夾
    run_offline(base_path, conf=0.5, rotate180=False, device=None, BATCH=32, IMG=448, DRAW=True, PRE_RESIZE=True)      # 執行主流程
