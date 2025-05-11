# tf_config.py
# Configure TensorFlow environment variables
import os

def configure():
    """Configure TensorFlow environment variables."""
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    print("TensorFlow environment variables configured programmatically.")

# Configure when this module is imported
configure()
