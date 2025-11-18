/* Copyright 2020-2025 Consensys Software Inc.
Copyright 2025 The RabbitSNARK Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package main

/*
#include <string.h>
*/
import "C"

import (
	"bufio"
	"fmt"
	"math/big"
	"os"
	"time"
	"unsafe"

	"github.com/consensys/gnark-crypto/ecc"
	curve "github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr/hash_to_field"
	"github.com/consensys/gnark/backend"
	"github.com/consensys/gnark/backend/groth16"
	groth16_bn254 "github.com/consensys/gnark/backend/groth16/bn254"
	"github.com/consensys/gnark/backend/witness"
	"github.com/consensys/gnark/constraint"
	cs "github.com/consensys/gnark/constraint/bn254"
	"github.com/consensys/gnark/constraint/solver"
	fcs "github.com/consensys/gnark/frontend/cs"
)

// Proof represents a Groth16 proof that was encoded with a ProvingKey and can be verified
// with a valid statement and a VerifyingKey
// Notation follows Figure 4. in DIZK paper https://eprint.iacr.org/2018/691.pdf
type Proof struct {
	Ar, Krs       curve.G1Affine
	Bs            curve.G2Affine
	Commitments   []curve.G1Affine // Pedersen commitments a la https://eprint.iacr.org/2022/1072
	CommitmentPok curve.G1Affine   // Batched proof of knowledge of the above commitments
}

var globalR1cs constraint.ConstraintSystem = groth16.NewCS(ecc.BN254)
var globalR1csInitialized = false
var globalPk groth16.ProvingKey = groth16.NewProvingKey(ecc.BN254)
var globalPkInitialized = false
var globalSolutions cs.R1CSSolution

func ReadR1CS(path string) constraint.ConstraintSystem {
	start := time.Now()
	defer func() {
		fmt.Printf("Reading R1CS took %s\n", time.Since(start))
	}()
	if globalR1csInitialized {
		return globalR1cs
	}

	r1csFile, err := os.Open(path)
	if err != nil {
		panic(err)
	}

	stat, err := r1csFile.Stat()
	if err != nil {
		panic(err)
	}

	r1csReader := bufio.NewReaderSize(r1csFile, int(stat.Size()))
	_, err = globalR1cs.ReadFrom(r1csReader)
	if err != nil {
		panic(err)
	}
	defer r1csFile.Close()
	globalR1csInitialized = true
	return globalR1cs
}

func ReadPK(path string) groth16.ProvingKey {
	start := time.Now()
	defer func() {
		fmt.Printf("Reading proving key took %s\n", time.Since(start))
	}()
	if globalPkInitialized {
		return globalPk
	}

	pkFile, err := os.Open(path)
	if err != nil {
		panic(err)
	}
	defer pkFile.Close()

	stat, err := pkFile.Stat()
	if err != nil {
		panic(err)
	}

	pkReader := bufio.NewReaderSize(pkFile, int(stat.Size()))
	err = globalPk.ReadDump(pkReader)
	if err != nil {
		panic(err)
	}
	defer pkFile.Close()
	globalPkInitialized = true
	return globalPk
}

func ReadWitness(path string) witness.Witness {
	start := time.Now()
	defer func() {
		fmt.Printf("Reading witness took %s\n", time.Since(start))
	}()
	file, err := os.Open(path)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	stat, err := file.Stat()
	if err != nil {
		panic(err)
	}

	data := make([]byte, stat.Size())
	_, err = file.Read(data)
	if err != nil {
		panic(err)
	}

	reconstructed, err := witness.New(ecc.BN254.ScalarField())
	if err != nil {
		panic(err)
	}
	err = reconstructed.UnmarshalBinary(data)
	if err != nil {
		panic(err)
	}
	return reconstructed
}

func SolveR1CS(r1cs *cs.R1CS, pk *groth16_bn254.ProvingKey, fullWitness witness.Witness, opts ...backend.ProverOption) any {
	start := time.Now()
	defer func() {
		fmt.Printf("Solving R1CS took %s\n", time.Since(start))
	}()
	opt, err := backend.NewProverConfig(opts...)
	if err != nil {
		panic(err)
	}
	if opt.HashToFieldFn == nil {
		opt.HashToFieldFn = hash_to_field.New([]byte(constraint.CommitmentDst))
	}

	commitmentInfo := r1cs.CommitmentInfo.(constraint.Groth16Commitments)

	proof := &Proof{Commitments: make([]curve.G1Affine, len(commitmentInfo))}

	solverOpts := opt.SolverOpts[:len(opt.SolverOpts):len(opt.SolverOpts)]

	privateCommittedValues := make([][]fr.Element, len(commitmentInfo))

	// override hints
	bsb22ID := solver.GetHintID(fcs.Bsb22CommitmentComputePlaceholder)
	solverOpts = append(solverOpts, solver.OverrideHint(bsb22ID, func(_ *big.Int, in []*big.Int, out []*big.Int) error {
		i := int(in[0].Int64())
		in = in[1:]
		privateCommittedValues[i] = make([]fr.Element, len(commitmentInfo[i].PrivateCommitted))
		hashed := in[:len(commitmentInfo[i].PublicAndCommitmentCommitted)]
		committed := in[+len(hashed):]
		for j, inJ := range committed {
			privateCommittedValues[i][j].SetBigInt(inJ)
		}

		var err error
		if proof.Commitments[i], err = pk.CommitmentKeys[i].Commit(privateCommittedValues[i]); err != nil {
			return err
		}

		opt.HashToFieldFn.Write(constraint.SerializeCommitment(proof.Commitments[i].Marshal(), hashed, (fr.Bits-1)/8+1))
		hashBts := opt.HashToFieldFn.Sum(nil)
		opt.HashToFieldFn.Reset()
		nbBuf := fr.Bytes
		if opt.HashToFieldFn.Size() < fr.Bytes {
			nbBuf = opt.HashToFieldFn.Size()
		}
		var res fr.Element
		res.SetBytes(hashBts[:nbBuf])
		res.BigInt(out[0])
		return nil
	}))

	solution, err := r1cs.Solve(fullWitness, solverOpts...)
	if err != nil {
		panic(err)
	}

	return solution
}

//export MakeSolutions
func MakeSolutions(r1csPath *C.char, pkPath *C.char, witnessPath *C.char,
	aPtr, bPtr, cPtr, wPtr unsafe.Pointer) {
	r1cs := ReadR1CS(C.GoString(r1csPath))
	pk := ReadPK(C.GoString(pkPath))
	witness := ReadWitness(C.GoString(witnessPath))
	solution := SolveR1CS(r1cs.(*cs.R1CS), pk.(*groth16_bn254.ProvingKey), witness)
	globalSolutions = *solution.(*cs.R1CSSolution)

	*(**C.void)(aPtr) = (*C.void)(unsafe.Pointer(&globalSolutions.A[0]))
	*(**C.void)(bPtr) = (*C.void)(unsafe.Pointer(&globalSolutions.B[0]))
	*(**C.void)(cPtr) = (*C.void)(unsafe.Pointer(&globalSolutions.C[0]))
	*(**C.void)(wPtr) = (*C.void)(unsafe.Pointer(&globalSolutions.W[0]))

	fmt.Println("globalSolutions.A", unsafe.Pointer(&globalSolutions.A[0]))
	fmt.Println("globalSolutions.B", unsafe.Pointer(&globalSolutions.B[0]))
	fmt.Println("globalSolutions.C", unsafe.Pointer(&globalSolutions.C[0]))
	fmt.Println("globalSolutions.W", unsafe.Pointer(&globalSolutions.W[0]))
}

func main() {}
