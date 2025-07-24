load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def rabbitsnark_deps():
    ZKX_COMMIT = "5aa487cbeb9171a9b699dde6c2c616243b62a4a0"
    http_archive(
        name = "zkx",
        sha256 = "1eea251eb1fcfdf023100f3baf6191e251e4858ea9406205de910582bf8cf9b2",
        strip_prefix = "zkx-{commit}".format(commit = ZKX_COMMIT),
        urls = ["https://github.com/zk-rabbit/zkx/archive/{commit}.tar.gz".format(commit = ZKX_COMMIT)],
    )
    # Uncomment this for development!
    # native.local_repository(
    #     name = "zkx",
    #     path = "../zkx",
    # )
