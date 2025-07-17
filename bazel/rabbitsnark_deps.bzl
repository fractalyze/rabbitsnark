load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def rabbitsnark_deps():
    ZKX_COMMIT = "63c55b0cf18d0000d2c0b46c3d5dee3a5d480795"
    http_archive(
        name = "zkx",
        sha256 = "815c528d493168cfd29493c9f109de12cba2e31a975f3cf5754f1c92ef69f110",
        strip_prefix = "zkx-{commit}".format(commit = ZKX_COMMIT),
        urls = ["https://github.com/zk-rabbit/zkx/archive/{commit}.tar.gz".format(commit = ZKX_COMMIT)],
    )
    # Uncomment this for development!
    # native.local_repository(
    #     name = "zkx",
    #     path = "../zkx",
    # )
