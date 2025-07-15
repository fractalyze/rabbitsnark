load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def rabbitsnark_deps():
    ZKX_COMMIT = "328f75d16d17c9e414322d4cfcfadf55852d44a5"
    http_archive(
        name = "zkx",
        sha256 = "c4ae29351b724190cbb12b4d51eaaa079d0d4c8b87449e063e12fcf74f4465bc",
        strip_prefix = "zkx-{commit}".format(commit = ZKX_COMMIT),
        urls = ["https://github.com/zk-rabbit/zkx/archive/{commit}.tar.gz".format(commit = ZKX_COMMIT)],
    )
    # Uncomment this for development!
    # native.local_repository(
    #     name = "zkx",
    #     path = "../zkx",
    # )
