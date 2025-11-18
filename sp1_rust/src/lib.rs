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

use p3_baby_bear::BabyBear;
use p3_field::{
    extension::BinomialExtensionField, AbstractExtensionField, AbstractField, Field, PrimeField32,
};

#[no_mangle]
pub extern "C" fn babybearextinv(a: u32, b: u32, c: u32, d: u32, i: u32) -> u32 {
    let a = BabyBear::from_wrapped_u32(a);
    let b = BabyBear::from_wrapped_u32(b);
    let c = BabyBear::from_wrapped_u32(c);
    let d = BabyBear::from_wrapped_u32(d);
    let inv = BinomialExtensionField::<BabyBear, 4>::from_base_slice(&[a, b, c, d]).inverse();
    let inv: &[BabyBear] = inv.as_base_slice();
    inv[i as usize].as_canonical_u32()
}

#[no_mangle]
pub extern "C" fn babybearinv(a: u32) -> u32 {
    let a = BabyBear::from_wrapped_u32(a);
    a.inverse().as_canonical_u32()
}

#[cfg(test)]
pub mod test {
    use super::babybearextinv;

    #[test]
    fn test_babybearextinv() {
        babybearextinv(1, 2, 3, 4, 0);
    }
}
