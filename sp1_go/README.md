# SP1 Go

This is based on
[zk-rabbit/sp1#feat/export-make-witness](https://github.com/zk-rabbit/sp1/tree/feat/export-make-witness).

## How to Build `libsp1.a`

```shell
bazel build //:libsp1
```

This will produce `bazel-bin/libsp1_/libsp1.a`, which is required for SP1
Groth16 proof generation.

## How to Generate [libsp1.h](/sp1_go/libsp1.h) and [babybear.h](/sp1_go/babybear.h)

You can also generate the headers manually by running:

```shell
go build -o libsp1.a -buildmode=c-archive main.go
```

However, you may encounter the following error:

```shell
# github.com/succinctlabs/sp1-recursion-gnark/sp1/babybear
sp1/babybear/babybear.go:4:10: fatal error: 'babybear.h' file not found
    4 | #include "babybear.h"
      |          ^~~~~~~~~~~~
1 error generated.
```

To fix this, temporarily modify
[sp1/babybear/babybear.go](/sp1_go/sp1/babybear/babybear.go) as follows:

```diff
package babybear

/*
-#include "babybear.h"
+#include "../../babybear.h"
*/
import "C"
```

This change adjusts the relative path so that go build can locate
[babybear.h](/sp1_go/babybear.h).

## How to Generate `libbabybear.a`

For instructions on how to obtain `libbabybear.a`, please refer to
[SP1 Rust](/sp1_rust/README.md).

## How to Use

Move the following files into a suitable location:

- `libsp1.h`
- `babybear.h`
- `libsp1.a`
- `libbabybear.a`

Then update your `/path/to/BUILD.bazel` file as follows:

```bazel
load("@rules_cc//cc:defs.bzl", "cc_import")

package(default_visibility = ["//visibility:public"])

cc_import(
    name = "sp1_cc",
    hdrs = ["libsp1.h"],
    includes = ["."],
    static_library = "libsp1.a",
)

cc_import(
    name = "babybear_cc",
    hdrs = ["babybear.h"],
    static_library = "libbabybear.a",
)

cc_binary(
    name = "prover_main",
    srcs = ["prover_main.cc"],
    deps = [
        ":babybear_cc",
        ":sp1_cc",
        "@zkx//zkx/math/elliptic_curves/bn/bn254:fr",
    ]
)
```

### API Example

```c++
// prover_main.cc
#include "/path/to/libsp1.h"
#include "zkx/math/elliptic_curves/bn/bn254/fr.h"

int main() {
  using F = math::bn254::Fr;

 // Generate witness binary
  MakeWitness("/path/to/groth16_witness.json", "new_witness.bin");

  // Prepare output pointers
  F* a = nullptr;
  F* b = nullptr;
  F* c = nullptr;
  F* w = nullptr;

  // Compute Groth16 witness solutions
  MakeSolutions("/path/to/groth16_circuit.bin",
                "/path/to/groth16_pk.bin",
                "new_witness.bin", &a, &b, &c, &w);

  // Use `a`, `b`, `c` and `w`!
}
```
