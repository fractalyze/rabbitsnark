# RabbitSNARK

This is a command-line tool for generating zero-knowledge proofs using the Groth16 proving system. It is built with the [ZKIR](https://github.com/zk-rabbit/zkir) and utilizes [ZKX](zkx). The tool is designed to handle [circom](https://docs.circom.io/getting-started/installation/)-style circuits and supports both standard and non-zero-knowledge (non-ZK) proof generation modes.

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

## How to run

```shell
bazel-bin/prover_main [OPTIONS] COMMAND
```

### Commands

- `compile`: Compile the Groth16 prover
- `prove`: Generate a proof using compiled Groth16 prover

### Global Options

- `--curve`: Elliptic curve to use (available: bn254). Default: bn254
- `-v`, `--vlog_level`: Logging verbosity level (default: 0)

### Subcommand: `compile`

```shell
bazel-bin/prover_main circom compile proving_key output [OPTIONS]
```

#### Positional arguments for `compile`

- `zkey`: Path to the Groth16 proving key (`.zkey` for circom)
- `output`: Directory to store compiled prover output

#### Optional flags for `compile`

- `h_msm_window_bits`: Window size (in bits) for h MSM (Multi-Scalar Multiplication). If set to 0, it will be estimated automatically using a cost model based on Pippenger’s algorithm. See: [Pippenger Complexity](https://encrypt.a41.io/primitives/abstract-algebra/elliptic-curve/msm/pippengers-algorithm#total-complexity) (Default: 0)
- `non_h_msm_window_bits`: Window size (in bits) for non h MSM (Multi-Scalar Multiplication). If set to 0, it will be estimated automatically using a cost model based on Pippenger’s algorithm. See: [Pippenger Complexity](https://encrypt.a41.io/primitives/abstract-algebra/elliptic-curve/msm/pippengers-algorithm#total-complexity) (Default: 0)
- `--skip_hlo`: Skip HLO generation from zkey (including sparse matrix processing). Intended for compiler developers to speed up iterative compilation tests when HLO has already been generated once.

### Subcommand: `prove`

```shell
bazel-bin/prover_main circom prove witness proof public output [OPTIONS]
```

#### Positional arguments for `prove`

- `witness`: Path to the witness (`.wtns` for circom)
- `proof`: Output path for proof (`.json`)
- `public`: Output path for public inputs (`.json`)
- `output`: Directory containing compiled prover files

#### Optional flags for `prove`

- `--no_zk`: Create proof without zero-knowledge (outputs full witness info). Useful for verifying correctness or benchmarking against tools like rapidsnark.
