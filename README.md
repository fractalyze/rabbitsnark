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

   If you're generating Groth16 proofs using SP1 with Gnark, add the `--//gnark:has_sp1` flag when building.

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
bazel-bin/prover_main circuit_dsl compile proving_key output [OPTIONS]
```

- `circuit_dsl`: Type of ZK proof circuit DSL (`circom` or `gnark`)
- `proving_key`: Path to the Groth16 proving key (`.zkey` for circom, `.bin` for gnark)
- `output`: Directory to store compiled prover output

#### Optional flags for `compile`

- `h_msm_window_bits`: Window size (in bits) for h MSM (Multi-Scalar Multiplication). If set to 0, it will be estimated automatically using a cost model based on Pippenger’s algorithm. See: [Pippenger Complexity](https://encrypt.a41.io/primitives/abstract-algebra/elliptic-curve/msm/pippengers-algorithm#total-complexity) (Default: 0)
- `non_h_msm_window_bits`: Window size (in bits) for non h MSM (Multi-Scalar Multiplication). If set to 0, it will be estimated automatically using a cost model based on Pippenger’s algorithm. See: [Pippenger Complexity](https://encrypt.a41.io/primitives/abstract-algebra/elliptic-curve/msm/pippengers-algorithm#total-complexity) (Default: 0)
- `--skip_hlo`: Skip HLO generation from zkey (including sparse matrix processing). Intended for compiler developers to speed up iterative compilation tests when HLO has already been generated once.

### Subcommand: `prove`

#### Circom

```shell
bazel-bin/prover_main circom prove witness proof public output [OPTIONS]
```

- `witness`: Path to the witness (`.wtns`)
- `proof`: Output path for proof (`.json`)
- `public`: Output path for public inputs (`.json`)
- `output`: Directory containing compiled prover files

#### Gnark

All files are inputted as `.bin` for gnark

```shell
bazel-bin/prover_main gnark prove r1cs proving_key witness proof public output [OPTIONS]
```

- `r1cs`: Path to the r1cs (`.bin`)
- `proving_key`: Path to the proving key (`.bin`)
- `witness`: Path to the witness. If the binary was built with `--//gnark:has_sp1`, this should point to the input witness JSON file (`.json`); otherwise, provide a binary witness file (`.bin`).
- `proof`: Output path for proof (`.bin`)
- `public`: Output path for public inputs (`.bin`)
- `output`: Directory containing compiled prover files

#### Optional flags for `prove`

- `--no_zk`: Create proof without zero-knowledge (outputs full witness info). Useful for verifying correctness or benchmarking against tools like rapidsnark.
