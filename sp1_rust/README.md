# SP1 Rust

For a full explanation, please refer to [SP1 Go](/sp1_go/README.md).

## How to Build

```shell
bazel build //:babybear
```

This will produce `bazel-bin/libbabybear.a`, which is required for SP1 Groth16 proof generation.
