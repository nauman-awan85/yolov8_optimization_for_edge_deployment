import os, glob, argparse
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

def load_paths(d, exts=(".jpg", ".jpeg")):
    files = []
    for e in exts:
        files += glob.glob(os.path.join(d, f"**/*{e}"), recursive=True)
    return files

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw / 2, dh / 2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img

def preprocess_bgr(img, img_size):
    img = letterbox(img, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None]
    return np.ascontiguousarray(img)

class Calib(trt.IInt8EntropyCalibrator2):
    def __init__(self, imgs, img_size, cache_path, batch_size=1):
        super().__init__()
        self.imgs = imgs
        self.img_size = img_size
        self.cache_path = cache_path
        self.batch_size = batch_size
        self.idx = 0
        self.shape = (batch_size, 3, img_size, img_size)
        self.device_input = cuda.mem_alloc(int(np.prod(self.shape)) * np.float32().nbytes)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.idx >= len(self.imgs):
            return None
        batch = []
        for _ in range(self.batch_size):
            if self.idx >= len(self.imgs):
                break
            img = cv2.imread(self.imgs[self.idx])
            if img is None:
                self.idx += 1
                continue
            batch.append(preprocess_bgr(img, self.img_size))
            self.idx += 1
        if not batch:
            return None
        x = np.vstack(batch)
        cuda.memcpy_htod(self.device_input, x)
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_path, "wb") as f:
            f.write(cache)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--calib_dir", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--img_size", type=int, default=640)
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()

    trt_logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(trt_logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, trt_logger)

    with open(args.onnx, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise SystemExit("ONNX parse failed")

    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if layer.type == trt.LayerType.SOFTMAX or "Softmax" in (layer.name or ""):
            layer.precision = trt.DataType.FLOAT
            for o in range(layer.num_outputs):
                layer.set_output_type(o, trt.DataType.FLOAT)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

    imgs = load_paths(args.calib_dir)
    calib = Calib(imgs, args.img_size, args.cache, args.batch)
    config.int8_calibrator = calib

    profile = builder.create_optimization_profile()
    inp = network.get_input(0)
    ishape = tuple(int(d) if d != -1 else -1 for d in inp.shape)
    if any(d == -1 for d in ishape):
        profile.set_shape("images",
                          min=(1, 3, args.img_size, args.img_size),
                          opt=(1, 3, args.img_size, args.img_size),
                          max=(4, 3, args.img_size, args.img_size))
    else:
        b, c, h, w = ishape
        profile.set_shape("images", min=(b, c, h, w),
                          opt=(b, c, h, w),
                          max=(b, c, h, w))
    config.add_optimization_profile(profile)
    config.set_calibration_profile(profile)

    engine = builder.build_engine(network, config)
    if engine is None:
        raise SystemExit("Failed to build engine for calibration.")
    print(f"Calibration cache written: {args.cache}")

main()
