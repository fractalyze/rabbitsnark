# RabbitSNARK

This is a command-line tool for generating zero-knowledge proofs using the Groth16 proving system. It is built with the [ZKIR](https://github.com/zk-rabbit/zkir) and utilizes [ZKX](zkx). The tool is designed to handle [circom](https://docs.circom.io/getting-started/installation/) and [gnark](https://github.com/Consensys/gnark) circuits and supports both standard and non-zero-knowledge (non-ZK) proof generation modes.

## How to build

1. Clone the RabbitSNARK repository

   ```shell
   git clone https://github.com/zk-rabbit/rabbitsnark.git
   ```

1. Navigate to the RabbitSNARK project directory

   ```shell
   cd rabbitsnark
   ```

1. Build the RabbitSNARK Prover binary

   ```shell
   bazel build --@zkx//:has_openmp //:prover_main
   ```

   This will compile the prover_main binary with OpenMP support for parallelism. After the build completes successfully, the binary will be located at `bazel-bin/prover_main`.

   If you're generating Groth16 proofs using SP1 with Gnark, add the `--//gnark:has_sp1` flag when building.

## How to run

```shell
bazel-bin/prover_main [GLOBAL_OPTIONS] COMMAND [SUBCOMMAND] [SUBCOMMAND_ARGS] [OPTIONS]
```

To see available commands:

```shell
bazel-bin/prover_main -h
```

To see help for a specific command (e.g., `circom`, `gnark`, `version`):

```shell
bazel-bin/prover_main circom -h
bazel-bin/prover_main gnark -h
bazel-bin/prover_main version -h
```

To see help for a specific subcommand (e.g., `compile`, `prove`):

```shell
bazel-bin/prover_main circom compile -h
bazel-bin/prover_main gnark prove -h
```
