import tensorflow as tf
from tensorflow.keras import layers, models
from .layers import FixedRFFT, WeightedRFFT

VARIANTS = (
    "PlainMLP",
    "ResidualMLP",
    "FixedRFFT_Residual",
    "WeightedRFFT_Residual",
    "FixedRFFT_Hybrid",
    "WeightedRFFT_Hybrid_NoResidual",
    "WeightedRFFT_Hybrid",
)


def _residual_spectral_branch(inp, weighted: bool):
    x = WeightedRFFT(name="weighted_rfft")(inp) if weighted else FixedRFFT(name="fixed_rfft")(inp)
    x = layers.Dense(64, activation="relu", name="phys_dense1")(x)
    x = layers.Dense(64, name="phys_dense2")(x)
    skip = layers.Dense(64, name="phys_skip")(inp)
    x = layers.Add(name="phys_res_add")([x, skip])
    return layers.Activation("relu", name="phys_res_relu")(x)


def _hybrid_head(inp, spectral, num_classes: int):
    raw = layers.Dense(64, activation="relu", name="raw_dense")(inp)
    merged = layers.Concatenate(name="fusion_concat")([spectral, raw])
    merged = layers.Dense(64, activation="relu", name="fusion_dense")(merged)
    return layers.Dense(num_classes, activation="softmax", name="classifier")(merged)


def build_model(variant: str, input_dim: int, num_classes: int, learning_rate: float):
    inp = layers.Input(shape=(input_dim,), name="vibration_window")
    if variant == "PlainMLP":
        x = layers.Dense(64, activation="relu", name="plain_dense")(inp)
        out = layers.Dense(num_classes, activation="softmax", name="classifier")(x)
    elif variant == "ResidualMLP":
        x = layers.Dense(64, activation="relu", name="res_dense")(inp)
        skip = layers.Dense(64, name="res_skip")(inp)
        x = layers.Add(name="res_add")([x, skip])
        x = layers.Activation("relu", name="res_relu")(x)
        out = layers.Dense(num_classes, activation="softmax", name="classifier")(x)
    elif variant == "FixedRFFT_Residual":
        out = layers.Dense(num_classes, activation="softmax", name="classifier")(
            _residual_spectral_branch(inp, weighted=False)
        )
    elif variant == "WeightedRFFT_Residual":
        out = layers.Dense(num_classes, activation="softmax", name="classifier")(
            _residual_spectral_branch(inp, weighted=True)
        )
    elif variant == "FixedRFFT_Hybrid":
        out = _hybrid_head(inp, _residual_spectral_branch(inp, weighted=False), num_classes)
    elif variant == "WeightedRFFT_Hybrid_NoResidual":
        spectral = WeightedRFFT(name="weighted_rfft")(inp)
        spectral = layers.Dense(64, activation="relu", name="phys_dense1")(spectral)
        spectral = layers.Dense(64, activation="relu", name="phys_dense2_noskip")(spectral)
        out = _hybrid_head(inp, spectral, num_classes)
    elif variant == "WeightedRFFT_Hybrid":
        out = _hybrid_head(inp, _residual_spectral_branch(inp, weighted=True), num_classes)
    else:
        raise ValueError(f"Unknown variant: {variant}")
    model = models.Model(inp, out, name=variant)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
