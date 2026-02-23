import onnxruntime as ort
import numpy as np
import sys

model_path = "public/models/strawberry1.onnx"
try:
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    print("Inputs:")
    for inp in session.get_inputs():
        print(f"  name: {inp.name}, shape: {inp.shape}, type: {inp.type}")
    print("Outputs:")
    for out in session.get_outputs():
        print(f"  name: {out.name}, shape: {out.shape}, type: {out.type}")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)