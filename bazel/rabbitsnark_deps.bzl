"""Rabbitsnark dependencies"""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def rabbitsnark_deps():
    ZKX_COMMIT = "ac33d9e7c824efc9b63a161989bbccbbb2932c7f"
    http_archive(
        name = "zkx",
        sha256 = "67411848a728c53cd346f6b7a9f0b815f469f2de86e82fd8384f0f9e527da145",
        strip_prefix = "zkx-{commit}".format(commit = ZKX_COMMIT),
        urls = ["https://github.com/zk-rabbit/zkx/archive/{commit}.tar.gz".format(commit = ZKX_COMMIT)],
    )
    # Uncomment this for development!
    # native.local_repository(
    #     name = "zkx",
    #     path = "../zkx",
    # )
