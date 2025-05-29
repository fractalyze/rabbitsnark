def groth16_deps():
    native.local_repository(
        name = "zkx",
        path = "../zkx",
    )
