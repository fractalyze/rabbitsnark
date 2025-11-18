# Copyright 2025 The RabbitSNARK Authors.
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
