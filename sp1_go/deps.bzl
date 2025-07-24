load("@bazel_gazelle//:deps.bzl", "go_repository")

def go_dependencies():
    go_repository(
        name = "com_github_bits_and_blooms_bitset",
        importpath = "github.com/bits-and-blooms/bitset",
        sum = "h1:YXVoyPndbdvcEVcseEovVfp0qjJp7S+i5+xgp/Nfbdc=",
        version = "v1.14.2",
    )
    go_repository(
        name = "com_github_blang_semver_v4",
        importpath = "github.com/blang/semver/v4",
        sum = "h1:1PFHFE6yCCTv8C1TeyNNarDzntLi7wMI5i/pzqYIsAM=",
        version = "v4.0.0",
    )
    go_repository(
        name = "com_github_chzyer_readline",
        importpath = "github.com/chzyer/readline",
        sum = "h1:upd/6fQk4src78LMRzh5vItIt361/o4uq553V8B5sGI=",
        version = "v1.5.1",
    )
    go_repository(
        name = "com_github_consensys_bavard",
        importpath = "github.com/consensys/bavard",
        sum = "h1:oLhMLOFGTLdlda/kma4VOJazblc7IM5y5QPd2A/YjhQ=",
        version = "v0.1.13",
    )
    go_repository(
        name = "com_github_consensys_compress",
        importpath = "github.com/consensys/compress",
        sum = "h1:gJr1hKzbOD36JFsF1AN8lfXz1yevnJi1YolffY19Ntk=",
        version = "v0.2.5",
    )
    go_repository(
        name = "com_github_consensys_gnark",
        importpath = "github.com/consensys/gnark",
        replace = "github.com/jtguibas/gnark",
        sum = "h1:vaqr8feWdNWKvsApw7iBhgbTznreKg+YvNAsbvqvwow=",
        version = "v0.0.0-20240923234830-41125bc1909c",
    )
    go_repository(
        name = "com_github_consensys_gnark_crypto",
        importpath = "github.com/consensys/gnark-crypto",
        sum = "h1:DDBdl4HaBtdQsq/wfMwJvZNE80sHidrK3Nfrefatm0E=",
        version = "v0.14.0",
    )
    go_repository(
        name = "com_github_consensys_gnark_ignition_verifier",
        importpath = "github.com/consensys/gnark-ignition-verifier",
        sum = "h1:z42ewLaLxoTYeQ17arcF4WExZc/eSaN3YVlF7eEaPt4=",
        version = "v0.0.0-20230527014722-10693546ab33",
    )
    go_repository(
        name = "com_github_coreos_go_systemd_v22",
        importpath = "github.com/coreos/go-systemd/v22",
        sum = "h1:RrqgGjYQKalulkV8NGVIfkXQf6YYmOyiJKk8iXXhfZs=",
        version = "v22.5.0",
    )
    go_repository(
        name = "com_github_davecgh_go_spew",
        importpath = "github.com/davecgh/go-spew",
        sum = "h1:vj9j/u1bqnvCEfJOwUhtlOARqs3+rkHYY13jYWTU97c=",
        version = "v1.1.1",
    )
    go_repository(
        name = "com_github_fxamacker_cbor_v2",
        importpath = "github.com/fxamacker/cbor/v2",
        sum = "h1:iM5WgngdRBanHcxugY4JySA0nk1wZorNOpTgCMedv5E=",
        version = "v2.7.0",
    )
    go_repository(
        name = "com_github_godbus_dbus_v5",
        importpath = "github.com/godbus/dbus/v5",
        sum = "h1:9349emZab16e7zQvpmsbtjc18ykshndd8y2PG3sgJbA=",
        version = "v5.0.4",
    )
    go_repository(
        name = "com_github_google_go_cmp",
        importpath = "github.com/google/go-cmp",
        sum = "h1:ofyhxvXcZhMsU5ulbFiLKl/XBFqE1GSq7atu8tAmTRI=",
        version = "v0.6.0",
    )
    go_repository(
        name = "com_github_google_pprof",
        importpath = "github.com/google/pprof",
        sum = "h1:FKHo8hFI3A+7w0aUQuYXQ+6EN5stWmeY/AZqtM8xk9k=",
        version = "v0.0.0-20240727154555-813a5fbdbec8",
    )
    go_repository(
        name = "com_github_google_subcommands",
        importpath = "github.com/google/subcommands",
        sum = "h1:vWQspBTo2nEqTUFita5/KeEWlUL8kQObDFbub/EN9oE=",
        version = "v1.2.0",
    )
    go_repository(
        name = "com_github_ianlancetaylor_demangle",
        importpath = "github.com/ianlancetaylor/demangle",
        sum = "h1:KwWnWVWCNtNq/ewIX7HIKnELmEx2nDP42yskD/pi7QE=",
        version = "v0.0.0-20240312041847-bd984b5ce465",
    )
    go_repository(
        name = "com_github_icza_bitio",
        importpath = "github.com/icza/bitio",
        sum = "h1:ysX4vtldjdi3Ygai5m1cWy4oLkhWTAi+SyO6HC8L9T0=",
        version = "v1.1.0",
    )
    go_repository(
        name = "com_github_inconshreveable_mousetrap",
        importpath = "github.com/inconshreveable/mousetrap",
        sum = "h1:wN+x4NVGpMsO7ErUn/mUI3vEoE6Jt13X2s0bqwp9tc8=",
        version = "v1.1.0",
    )
    go_repository(
        name = "com_github_ingonyama_zk_icicle",
        importpath = "github.com/ingonyama-zk/icicle",
        sum = "h1:a2MUIaF+1i4JY2Lnb961ZMvaC8GFs9GqZgSnd9e95C8=",
        version = "v1.1.0",
    )
    go_repository(
        name = "com_github_ingonyama_zk_iciclegnark",
        importpath = "github.com/ingonyama-zk/iciclegnark",
        sum = "h1:88MkEghzjQBMjrYRJFxZ9oR9CTIpB8NG2zLeCJSvXKQ=",
        version = "v0.1.0",
    )
    go_repository(
        name = "com_github_kr_pretty",
        importpath = "github.com/kr/pretty",
        sum = "h1:flRD4NNwYAUpkphVc1HcthR4KEIFJ65n8Mw5qdRn3LE=",
        version = "v0.3.1",
    )
    go_repository(
        name = "com_github_kr_text",
        importpath = "github.com/kr/text",
        sum = "h1:5Nx0Ya0ZqY2ygV366QzturHI13Jq95ApcVaJBhpS+AY=",
        version = "v0.2.0",
    )
    go_repository(
        name = "com_github_leanovate_gopter",
        importpath = "github.com/leanovate/gopter",
        sum = "h1:vRjThO1EKPb/1NsDXuDrzldR28RLkBflWYcU9CvzWu4=",
        version = "v0.2.11",
    )
    go_repository(
        name = "com_github_mattn_go_colorable",
        importpath = "github.com/mattn/go-colorable",
        sum = "h1:fFA4WZxdEF4tXPZVKMLwD8oUnCTTo08duU7wxecdEvA=",
        version = "v0.1.13",
    )
    go_repository(
        name = "com_github_mattn_go_isatty",
        importpath = "github.com/mattn/go-isatty",
        sum = "h1:xfD0iDuEKnDkl03q4limB+vH+GxLEtL/jb4xVJSWWEY=",
        version = "v0.0.20",
    )
    go_repository(
        name = "com_github_mmcloughlin_addchain",
        importpath = "github.com/mmcloughlin/addchain",
        sum = "h1:SobOdjm2xLj1KkXN5/n0xTIWyZA2+s99UCY1iPfkHRY=",
        version = "v0.4.0",
    )
    go_repository(
        name = "com_github_mmcloughlin_profile",
        importpath = "github.com/mmcloughlin/profile",
        sum = "h1:jhDmAqPyebOsVDOCICJoINoLb/AnLBaUw58nFzxWS2w=",
        version = "v0.1.1",
    )
    go_repository(
        name = "com_github_pkg_errors",
        importpath = "github.com/pkg/errors",
        sum = "h1:FEBLx1zS214owpjy7qsBeixbURkuhQAwrK5UwLGTwt4=",
        version = "v0.9.1",
    )
    go_repository(
        name = "com_github_pmezard_go_difflib",
        importpath = "github.com/pmezard/go-difflib",
        sum = "h1:4DBwDE0NGyQoBHbLQYPwSUPoCMWR5BEzIk/f1lZbAQM=",
        version = "v1.0.0",
    )
    go_repository(
        name = "com_github_rogpeppe_go_internal",
        importpath = "github.com/rogpeppe/go-internal",
        sum = "h1:exVL4IDcn6na9z1rAb56Vxr+CgyK3nn3O+epU5NdKM8=",
        version = "v1.12.0",
    )
    go_repository(
        name = "com_github_ronanh_intcomp",
        importpath = "github.com/ronanh/intcomp",
        sum = "h1:i54kxmpmSoOZFcWPMWryuakN0vLxLswASsGa07zkvLU=",
        version = "v1.1.0",
    )
    go_repository(
        name = "com_github_rs_xid",
        importpath = "github.com/rs/xid",
        sum = "h1:mKX4bl4iPYJtEIxp6CYiUuLQ/8DYMoz0PUdtGgMFRVc=",
        version = "v1.5.0",
    )
    go_repository(
        name = "com_github_rs_zerolog",
        importpath = "github.com/rs/zerolog",
        sum = "h1:1cU2KZkvPxNyfgEmhHAz/1A9Bz+llsdYzklWFzgp0r8=",
        version = "v1.33.0",
    )
    go_repository(
        name = "com_github_spf13_cobra",
        importpath = "github.com/spf13/cobra",
        sum = "h1:e5/vxKd/rZsfSJMUX1agtjeTDf+qv1/JdBF8gg5k9ZM=",
        version = "v1.8.1",
    )
    go_repository(
        name = "com_github_spf13_pflag",
        importpath = "github.com/spf13/pflag",
        sum = "h1:iy+VFUOCP1a+8yFto/drg2CJ5u0yRoB7fZw3DKv/JXA=",
        version = "v1.0.5",
    )
    go_repository(
        name = "com_github_stretchr_objx",
        importpath = "github.com/stretchr/objx",
        sum = "h1:xuMeJ0Sdp5ZMRXx/aWO6RZxdr3beISkG5/G/aIRr3pY=",
        version = "v0.5.2",
    )
    go_repository(
        name = "com_github_stretchr_testify",
        importpath = "github.com/stretchr/testify",
        sum = "h1:HtqpIVDClZ4nwg75+f6Lvsy/wHu+3BoSGCbBAcpTsTg=",
        version = "v1.9.0",
    )
    go_repository(
        name = "com_github_x448_float16",
        importpath = "github.com/x448/float16",
        sum = "h1:qLwI1I70+NjRFUR3zs1JPUCgaCXSh3SW62uAKT1mSBM=",
        version = "v0.8.4",
    )
    go_repository(
        name = "in_gopkg_check_v1",
        importpath = "gopkg.in/check.v1",
        sum = "h1:Hei/4ADfdWqJk1ZMxUNpqntNwaWcugrBjAiHlqqRiVk=",
        version = "v1.0.0-20201130134442-10cb98267c6c",
    )
    go_repository(
        name = "in_gopkg_yaml_v2",
        importpath = "gopkg.in/yaml.v2",
        sum = "h1:D8xgwECY7CYvx+Y2n4sBz93Jn9JRvxdiyyo8CTfuKaY=",
        version = "v2.4.0",
    )
    go_repository(
        name = "in_gopkg_yaml_v3",
        importpath = "gopkg.in/yaml.v3",
        sum = "h1:fxVm/GzAzEWqLHuvctI91KS9hhNmmWOoWu0XTYJS7CA=",
        version = "v3.0.1",
    )
    go_repository(
        name = "io_rsc_tmplfunc",
        importpath = "rsc.io/tmplfunc",
        sum = "h1:53XFQh69AfOa8Tw0Jm7t+GV7KZhOi6jzsCzTtKbMvzU=",
        version = "v0.0.3",
    )
    go_repository(
        name = "org_golang_x_crypto",
        importpath = "golang.org/x/crypto",
        sum = "h1:RrRspgV4mU+YwB4FYnuBoKsUapNIL5cohGAmSH3azsw=",
        version = "v0.26.0",
    )
    go_repository(
        name = "org_golang_x_exp",
        importpath = "golang.org/x/exp",
        sum = "h1:kx6Ds3MlpiUHKj7syVnbp57++8WpuKPcR5yjLBjvLEA=",
        version = "v0.0.0-20240823005443-9b4947da3948",
    )
    go_repository(
        name = "org_golang_x_net",
        importpath = "golang.org/x/net",
        sum = "h1:AQyQV4dYCvJ7vGmJyKki9+PBdyvhkSd8EIx/qb0AYv4=",
        version = "v0.21.0",
    )
    go_repository(
        name = "org_golang_x_sync",
        importpath = "golang.org/x/sync",
        sum = "h1:3NFvSEYkUoMifnESzZl15y791HH1qU2xm6eCJU5ZPXQ=",
        version = "v0.8.0",
    )
    go_repository(
        name = "org_golang_x_sys",
        importpath = "golang.org/x/sys",
        sum = "h1:Twjiwq9dn6R1fQcyiK+wQyHWfaz/BJB+YIpzU/Cv3Xg=",
        version = "v0.24.0",
    )
    go_repository(
        name = "org_golang_x_term",
        importpath = "golang.org/x/term",
        sum = "h1:F6D4vR+EHoL9/sWAWgAR1H2DcHr4PareCbAaCo1RpuU=",
        version = "v0.23.0",
    )
    go_repository(
        name = "org_golang_x_text",
        importpath = "golang.org/x/text",
        sum = "h1:XtiM5bkSOt+ewxlOE/aE/AKEHibwj/6gvWMl9Rsh0Qc=",
        version = "v0.17.0",
    )
