import tensorflow as tf

# Load your model (modify this to your actual model loading code)
model = tf.keras.models.load_model('model.keras')

# Create the TFLite converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Set the target specs for TFLite
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
converter._experimental_lower_tensor_list_ops = False

# Convert the model
try:
    tflite_model = converter.convert()
    # Save the converted model
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)
    print("Model converted successfully!")
except Exception as e:
    print("Error during conversion:", e)
