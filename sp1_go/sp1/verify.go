/* Copyright 2023 Succinct Labs.
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

package sp1

import (
	"bytes"
	"encoding/hex"
	"os"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark/backend/plonk"
	"github.com/consensys/gnark/frontend"
	"github.com/succinctlabs/sp1-recursion-gnark/sp1/babybear"
)

func VerifyPlonk(verifyCmdDataDir string, verifyCmdProof string, verifyCmdVkeyHash string, verifyCmdCommittedValuesDigest string) error {
	// Sanity check the required arguments have been provided.
	if verifyCmdDataDir == "" {
		panic("--data is required")
	}

	// Decode the proof.
	proofDecodedBytes, err := hex.DecodeString(verifyCmdProof)
	if err != nil {
		panic(err)
	}
	proof := plonk.NewProof(ecc.BN254)
	if _, err := proof.ReadFrom(bytes.NewReader(proofDecodedBytes)); err != nil {
		panic(err)
	}

	// Read the verifier key.
	vkFile, err := os.Open(verifyCmdDataDir + "/" + plonkVkPath)
	if err != nil {
		panic(err)
	}
	vk := plonk.NewVerifyingKey(ecc.BN254)
	vk.ReadFrom(vkFile)

	// Compute the public witness.
	circuit := Circuit{
		Vars:                 []frontend.Variable{},
		Felts:                []babybear.Variable{},
		Exts:                 []babybear.ExtensionVariable{},
		VkeyHash:             verifyCmdVkeyHash,
		CommittedValuesDigest: verifyCmdCommittedValuesDigest,
	}
	witness, err := frontend.NewWitness(&circuit, ecc.BN254.ScalarField())
	if err != nil {
		panic(err)
	}
	publicWitness, err := witness.Public()
	if err != nil {
		panic(err)
	}

	// Verify proof.
	err = plonk.Verify(proof, vk, publicWitness)
	return err
}

func VerifyGroth16(verifyCmdDataDir string, verifyCmdProof string, verifyCmdVkeyHash string, verifyCmdCommittedValuesDigest string) error {
	// Sanity check the required arguments have been provided.
	if verifyCmdDataDir == "" {
		panic("--data is required")
	}

	// Decode the proof.
	proofDecodedBytes, err := hex.DecodeString(verifyCmdProof)
	if err != nil {
		panic(err)
	}
	proof := groth16.NewProof(ecc.BN254)
	if _, err := proof.ReadFrom(bytes.NewReader(proofDecodedBytes)); err != nil {
		panic(err)
	}

	// Read the verifier key.
	vkFile, err := os.Open(verifyCmdDataDir + "/" + groth16VkPath)
	if err != nil {
		panic(err)
	}
	vk := groth16.NewVerifyingKey(ecc.BN254)
	vk.ReadFrom(vkFile)

	// Compute the public witness.
	circuit := Circuit{
		Vars:                 []frontend.Variable{},
		Felts:                []babybear.Variable{},
		Exts:                 []babybear.ExtensionVariable{},
		VkeyHash:             verifyCmdVkeyHash,
		CommittedValuesDigest: verifyCmdCommittedValuesDigest,
	}
	witness, err := frontend.NewWitness(&circuit, ecc.BN254.ScalarField())
	if err != nil {
		panic(err)
	}
	publicWitness, err := witness.Public()
	if err != nil {
		panic(err)
	}

	// Verify proof.
	err = groth16.Verify(proof, vk, publicWitness)
	return err
}
