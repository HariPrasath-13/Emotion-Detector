import traceback, sys
try:
    import tensorflow as tf
    print("TensorFlow imported OK, version:", tf.__version__)
except Exception:
    traceback.print_exc()
    sys.exit(1)
