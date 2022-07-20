# Description:
#  Contains JAX related layers
load("@org_keras//keras:keras.bzl", "cuda_py_test")

package(
    default_visibility = [
        "//keras:friends",
        "//third_party/tensorflow/python/distribute:__pkg__",
        "//third_party/tensorflow/tools/pip_package:__pkg__",
    ],
    licenses = ["notice"],
)

py_library(
    name = "jax_layer",
    srcs = ["jax_layer.py"],
    deps = [
        "//:expect_jax_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras:backend",
        "//keras/engine:base_layer",
        "//keras/utils:tf_utils",
    ],
)

py_library(
    name = "flax_layer",
    srcs = ["flax_layer.py"],
    deps = [
        ":jax_layer",
        "//:expect_flax_installed",
        "//:expect_jax_installed",
    ],
)

cuda_py_test(
    name = "jax_layer_test",
    size = "medium",
    srcs = ["jax_layer_test.py"],
    shard_count = 12,
    deps = [
        ":flax_layer",
        ":jax_layer",
        "//:expect_absl_installed",
        "//:expect_flax_installed",
        "//:expect_haiku_installed",
        "//:expect_jax2tf_installed",  # buildcleaner: keep
        "//:expect_jax_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/saving:object_registration",
        "//keras/testing_infra:test_combinations",
        "//keras/testing_infra:test_utils",
        "//keras/utils:layer_utils",
    ],
)
