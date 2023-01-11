# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for inference-only model/layer exporting utilities."""
import os

import tensorflow.compat.v2 as tf
from absl.testing import parameterized

import keras
from keras.saving import export_lib
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils


@test_utils.run_v2_only
class ExportArchiveTest(tf.test.TestCase, parameterized.TestCase):
    def _get_model(self):
        layers = [
            keras.layers.Dense(10, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
        model = test_utils.get_model_from_layers(layers, input_shape=(10,))
        return model

    @test_combinations.run_with_all_model_types
    def test_standard_model_export(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = self._get_model()
        ref_input = tf.random.normal((3, 10))
        ref_output = model(ref_input).numpy()

        export_lib.export_model(model, temp_filepath)
        revived_model = tf.saved_model.load(temp_filepath)
        self.assertAllClose(
            ref_output, revived_model.serve(ref_input).numpy(), atol=1e-6
        )

    @test_combinations.run_with_all_model_types
    def test_low_level_model_export(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")

        model = self._get_model()
        ref_input = tf.random.normal((3, 10))
        ref_output = model(ref_input).numpy()

        @tf.function()
        def my_endpoint(x):
            return model(x)

        # Test registering an endpoint that is a tf.function (called)
        my_endpoint(ref_input)  # Trace fn
        export_archive = export_lib.ExportArchive(model)
        export_archive.add_endpoint(
            "call",
            my_endpoint,
        )
        export_archive.write_out(temp_filepath)
        revived_model = tf.saved_model.load(temp_filepath)
        self.assertAllClose(
            ref_output, revived_model.call(ref_input).numpy(), atol=1e-6
        )

        # Test registering an endpoint that is NOT a tf.function
        export_archive = export_lib.ExportArchive(model)
        export_archive.add_endpoint(
            "call",
            model.call,
            input_signature=[
                tf.TensorSpec(
                    shape=(None, 10),
                    dtype=tf.float32,
                )
            ],
        )
        export_archive.write_out(temp_filepath)
        revived_model = tf.saved_model.load(temp_filepath)
        self.assertAllClose(
            ref_output, revived_model.call(ref_input).numpy(), atol=1e-6
        )

    def test_layer_export(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_layer")

        layer = keras.layers.BatchNormalization()
        ref_input = tf.random.normal((3, 10))
        ref_output = layer(ref_input).numpy()  # Build layer (important)

        export_archive = export_lib.ExportArchive(layer)
        export_archive.add_endpoint(
            "call",
            layer.call,
            input_signature=[
                tf.TensorSpec(
                    shape=(None, 10),
                    dtype=tf.float32,
                )
            ],
        )
        export_archive.write_out(temp_filepath)
        revived_layer = tf.saved_model.load(temp_filepath)
        self.assertAllClose(
            ref_output, revived_layer.call(ref_input).numpy(), atol=1e-6
        )

    def test_non_standard_layer_signature(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_layer")

        layer = keras.layers.MultiHeadAttention(2, 2)
        x1 = tf.random.normal((3, 2, 2))
        x2 = tf.random.normal((3, 2, 2))
        ref_output = layer(x1, x2).numpy()  # Build layer (important)
        export_archive = export_lib.ExportArchive(layer)
        export_archive.add_endpoint(
            "call",
            layer.call,
            input_signature=[
                tf.TensorSpec(
                    shape=(None, 2, 2),
                    dtype=tf.float32,
                ),
                tf.TensorSpec(
                    shape=(None, 2, 2),
                    dtype=tf.float32,
                ),
            ],
        )
        export_archive.write_out(temp_filepath)
        revived_layer = tf.saved_model.load(temp_filepath)
        self.assertAllClose(
            ref_output,
            revived_layer.call(query=x1, value=x2).numpy(),
            atol=1e-6,
        )

    def test_export_model_errors(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")

        # Model has not been built
        model = keras.Sequential([keras.layers.Dense(2)])
        with self.assertRaisesRegex(ValueError, "It must be built"):
            export_lib.export_model(model, temp_filepath)

        # Subclassed model has not been called
        class MyModel(keras.Model):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.dense = keras.layers.Dense(2)

            def build(self, input_shape):
                self.dense.build(input_shape)
                self.built = True

            def call(self, x):
                return self.dense(x)

        model = MyModel()
        model.build((2, 3))
        with self.assertRaisesRegex(ValueError, "It must be called"):
            export_lib.export_model(model, temp_filepath)

    def test_export_archive_errors(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = keras.Sequential([keras.layers.Dense(2)])
        model(tf.random.normal((2, 3)))

        # Endpoint name reuse
        export_archive = export_lib.ExportArchive(model)
        export_archive.add_endpoint(
            "call",
            model.call,
            input_signature=[
                tf.TensorSpec(
                    shape=(None, 3),
                    dtype=tf.float32,
                )
            ],
        )
        with self.assertRaisesRegex(ValueError, "already taken"):
            export_archive.add_endpoint(
                "call",
                model.call,
                input_signature=[
                    tf.TensorSpec(
                        shape=(None, 3),
                        dtype=tf.float32,
                    )
                ],
            )

        # Write out with no endpoints
        export_archive = export_lib.ExportArchive(model)
        with self.assertRaisesRegex(ValueError, "No endpoints have been set"):
            export_archive.write_out(temp_filepath)

        # Invalid object type
        with self.assertRaisesRegex(ValueError, "Invalid layer type"):
            export_archive = export_lib.ExportArchive("model")

        # Set endpoint with no input signature
        export_archive = export_lib.ExportArchive(model)
        with self.assertRaisesRegex(
            ValueError, "you must provide an `input_signature`"
        ):
            export_archive.add_endpoint(
                "call",
                model.call,
            )

        # Set endpoint that has never been called
        export_archive = export_lib.ExportArchive(model)

        @tf.function()
        def my_endpoint(x):
            return model(x)

        export_archive = export_lib.ExportArchive(model)
        with self.assertRaisesRegex(
            ValueError, "you must either provide a function"
        ):
            export_archive.add_endpoint(
                "call",
                my_endpoint,
            )


if __name__ == "__main__":
    tf.test.main()
