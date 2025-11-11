"""Rabbitsnark dependencies"""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def rabbitsnark_deps():
    ZKX_COMMIT = "f740ea7dfee3ead6ad0a670fb5e70aa5995a88ea"
    http_archive(
        name = "zkx",
        sha256 = "6c9b646f0c0883ed66542efadc9e0c9be83011bfba93193c0a81d433364502ae",
        strip_prefix = "zkx-{commit}".format(commit = ZKX_COMMIT),
        urls = ["https://github.com/fractalyze/zkx/archive/{commit}.tar.gz".format(commit = ZKX_COMMIT)],
    )
    # Uncomment this for development!
    # native.local_repository(
    #     name = "zkx",
    #     path = "../zkx",
    # )
