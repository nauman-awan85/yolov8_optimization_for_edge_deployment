import os
import time
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import pyttsx3
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

ENGINE_PATH = r"C:\Users\Muhammad Nauman\Desktop\Presentation\models\int8_latest.engine"
IMG_SIZE = 512
CONF_THRES = 0.55
IOU_THRES = 0.45
VOICE_COOLDOWN = 10

CLASS_NAMES = ["banana","computer_mouse","electric_fan","key","computer_keyboard","laptop","microwave","monitor",
    "pen","pillow","pizza","plate","sock","spoon","stove","television","apple","bed","book","bowl",
    "car","chair","couch","dog","oven","bottle","cat","clock","cup","person","remote","wallet",
    "towel","lamp","drawer","painting","speaker","cell_phone","cereal_box","shoe"]

NUM_CLASSES = len(CLASS_NAMES)
RAW_C = 4 + NUM_CLASSES
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

tts = pyttsx3.init()
last_spoken = {}


def speak(text, cooldown=VOICE_COOLDOWN):
    now = time.time()
    if text not in last_spoken or (now - last_spoken[text]) > cooldown:
        print(text)
        tts.say(text)
        tts.runAndWait()
        last_spoken[text] = now


def letterbox(img, new_shape=(IMG_SIZE, IMG_SIZE), color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (new_shape[0] - nh) // 2
    bottom = new_shape[0] - nh - top
    left = (new_shape[1] - nw) // 2
    right = new_shape[1] - nw - left
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded, r, (left, top)


def preprocess_bgr(frame):
    lb_img, r, (dx, dy) = letterbox(frame, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(lb_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None]
    return np.ascontiguousarray(img), (r, dx, dy)


def iou_xyxy(a, b):
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(br - tl, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def nms(boxes, scores, iou_thres):
    keep, idxs = [], scores.argsort()[::-1]
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        ious = iou_xyxy(boxes[i:i + 1], boxes[idxs[1:]]).ravel()
        idxs = idxs[1:][ious < iou_thres]
    return np.array(keep, dtype=np.int32)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def parse_yolo_raw(raw_out, conf_thres, iou_thres):
    if raw_out.ndim == 3:
        raw_out = raw_out[0]
    if raw_out.shape[0] != RAW_C:
        return np.empty((0, 4)), np.empty((0,)), np.empty((0,), np.int32)
    raw_out = raw_out.transpose(1, 0)
    xywh = sigmoid(raw_out[:, :4]) * IMG_SIZE
    scores = sigmoid(raw_out[:, 4:])
    confs = scores.max(axis=1)
    cls_ids = scores.argmax(axis=1)
    mask = confs > conf_thres
    if not np.any(mask):
        return np.empty((0, 4)), np.empty((0,)), np.empty((0,), np.int32)
    xywh, confs, cls_ids = xywh[mask], confs[mask], cls_ids[mask]
    x, y, w, h = xywh.T
    boxes = np.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=1)
    keep = nms(boxes, confs, iou_thres)
    return boxes[keep], confs[keep], cls_ids[keep]


def parse_nms_output(out, conf_thres):
    if out.ndim == 3:
        out = out[0]
    if out.size == 0 or out.shape[-1] < 6:
        return np.empty((0, 4)), np.empty((0,)), np.empty((0,), np.int32)
    boxes, scores, classes = out[:, :4], out[:, 4], np.rint(out[:, 5]).astype(int)
    m = scores > conf_thres
    return boxes[m], scores[m], classes[m]


def choose_outputs(outs):
    raw = next((o for o in outs if o.ndim == 3 and o.shape[1] == RAW_C), None)
    if raw is not None:
        return parse_yolo_raw(raw, CONF_THRES, IOU_THRES)
    nms_out = next((o for o in outs if o.shape[-1] >= 6), None)
    if nms_out is not None:
        return parse_nms_output(nms_out, CONF_THRES)
    return np.empty((0, 4)), np.empty((0,)), np.empty((0,), np.int32)


def load_engine(path):
    with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as rt:
        engine = rt.deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()

    input_idx = [i for i in range(engine.num_bindings) if engine.binding_is_input(i)][0]
    out_idxs = [i for i in range(engine.num_bindings) if not engine.binding_is_input(i)]

    inp_shape = (1, 3, IMG_SIZE, IMG_SIZE)
    ctx.set_binding_shape(input_idx, inp_shape)
    inp_dtype = trt.nptype(engine.get_binding_dtype(input_idx))
    out_shapes = [tuple(ctx.get_binding_shape(i)) for i in out_idxs]
    out_dtypes = [trt.nptype(engine.get_binding_dtype(i)) for i in out_idxs]

    d_input = cuda.mem_alloc(int(np.prod(inp_shape)) * np.dtype(inp_dtype).itemsize)
    d_outputs = [cuda.mem_alloc(int(np.prod(s)) * np.dtype(dt).itemsize) for s, dt in zip(out_shapes, out_dtypes)]
    bindings = [None] * engine.num_bindings
    bindings[input_idx] = int(d_input)
    for i, bi in enumerate(out_idxs):
        bindings[bi] = int(d_outputs[i])

    return {"ctx": ctx,"out_shapes": out_shapes,"out_dtypes": out_dtypes,"d_input": d_input,"d_outputs": d_outputs,
        "bindings": bindings,"stream": cuda.Stream(),}


def infer(eng, frame):
    inp, (r, dx, dy) = preprocess_bgr(frame)
    cuda.memcpy_htod_async(eng["d_input"], inp, eng["stream"])
    eng["ctx"].execute_async_v2(eng["bindings"], eng["stream"].handle)
    host_outs = []
    for shape, dt, d_out in zip(eng["out_shapes"], eng["out_dtypes"], eng["d_outputs"]):
        host = np.empty(int(np.prod(shape)), dt)
        cuda.memcpy_dtoh_async(host, d_out, eng["stream"])
        host_outs.append(host.reshape(shape))
    eng["stream"].synchronize()
    return host_outs, (r, dx, dy)


def main():
    eng = load_engine(ENGINE_PATH)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Webcam not accessible")
    print("Webcam started. Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.time()

        outs, (r, dx, dy) = infer(eng, frame)
        boxes, scores, classes = choose_outputs(outs)

        disp = frame.copy()
        if boxes.size:
            boxes[:, [0, 2]] -= dx
            boxes[:, [1, 3]] -= dy
            boxes /= r
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, frame.shape[1] - 1)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, frame.shape[0] - 1)

        for (x1, y1, x2, y2), cl, conf in zip(boxes.astype(int), classes, scores):
            if 0 <= cl < NUM_CLASSES:
                name = CLASS_NAMES[cl]
                cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(disp, f"{name}:{conf:.2f}", (x1, max(0, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                if conf >= CONF_THRES:
                    speak(name, cooldown=VOICE_COOLDOWN)

        fps = 1.0 / (time.time() - t0 + 1e-9)
        cv2.putText(disp, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

        cv2.imshow("INT8 TRT + Voice", disp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

main()
