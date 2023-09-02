import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification


def create_model(model_name, optimizer, label2id, id2label):
    model = TFAutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        from_pt=True
    )
    
    model.compile(
        optimizer=optimizer,
        metrics=[tf.keras.metrics.binary_crossentropy]
    )
    
    model.summary()
    return model