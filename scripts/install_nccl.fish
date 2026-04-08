#!/usr/bin/env fish

# Machine-specific NCCL installer for this host:
# - Ubuntu 24.04 x86_64
# - CUDA 13.1 in /usr/local/cuda-13.1
# - NVIDIA H100 GPUs
#
# Usage:
#   fish scripts/install_nccl.fish
#   fish scripts/install_nccl.fish --dry-run

set -g CUDA_REPO_URL "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64"
set -g CUDA_KEYRING_DEB "cuda-keyring_1.1-1_all.deb"
set -g NCCL_VERSION "2.29.3-1+cuda13.1"
set -g EXPECTED_OS_ID "ubuntu"
set -g EXPECTED_OS_VERSION "24.04"
set -g EXPECTED_ARCH "x86_64"
set -g EXPECTED_CUDA_RELEASE "13.1"

set -g DRY_RUN 0
set -g TMPDIR_PATH ""

function usage
    printf '%s\n' \
        'Install NCCL for the current tensor-parallel-gemm machine.' \
        '' \
        'Options:' \
        '  --dry-run   Print the commands without executing them.' \
        '  -h, --help  Show this help message.'
end

function cleanup --on-process-exit %self
    if test -n "$TMPDIR_PATH"; and test -d "$TMPDIR_PATH"
        rm -rf -- "$TMPDIR_PATH"
    end
end

function log
    printf '[install_nccl] %s\n' "$argv[1]"
end

function fail
    printf '[install_nccl] ERROR: %s\n' "$argv[1]" >&2
    exit 1
end

function run
    printf '+ %s\n' (string join ' ' -- (string escape -- $argv))
    if test "$DRY_RUN" -eq 0
        command $argv
        set -l cmd_status $status
        if test $cmd_status -ne 0
            fail "Command failed with exit code $cmd_status: "(string join ' ' -- $argv)
        end
    end
end

function run_root
    if test (id -u) -eq 0
        run $argv
    else
        run sudo $argv
    end
end

function download
    set -l url "$argv[1]"
    set -l out "$argv[2]"

    if command -q wget
        run wget -O "$out" "$url"
        return
    end

    if command -q curl
        run curl -fsSL -o "$out" "$url"
        return
    end

    log "Neither wget nor curl is installed; installing curl first."
    run_root apt-get update
    run_root apt-get install -y curl ca-certificates
    run curl -fsSL -o "$out" "$url"
end

function ensure_cuda_repo_is_consistent
    set -l legacy_list /etc/apt/sources.list.d/cuda.list
    set -l managed_list /etc/apt/sources.list.d/cuda-ubuntu2404-x86_64.list

    if test -f "$legacy_list"
        if grep -q "$CUDA_REPO_URL" "$legacy_list"
            set -l backup_path "$legacy_list.disabled-by-install-nccl"
            log "Disabling conflicting APT source: $legacy_list -> $backup_path"
            run_root mv "$legacy_list" "$backup_path"
        end
    end

    if test -f "$managed_list"
        if grep -q "$CUDA_REPO_URL" "$managed_list"
            log "Using existing CUDA repo file: $managed_list"
            return
        end
    end

    set -l keyring_path "$TMPDIR_PATH/$CUDA_KEYRING_DEB"
    download "$CUDA_REPO_URL/$CUDA_KEYRING_DEB" "$keyring_path"
    run_root dpkg -i "$keyring_path"
end

function require_machine_match
    if not test -f /etc/os-release
        fail "/etc/os-release is missing."
    end

    set -l os_id (grep '^ID=' /etc/os-release | head -n 1 | cut -d= -f2 | string trim -c '"')
    set -l os_version (grep '^VERSION_ID=' /etc/os-release | head -n 1 | cut -d= -f2 | string trim -c '"')

    if test "$os_id" != "$EXPECTED_OS_ID"
        fail "Expected $EXPECTED_OS_ID, got $os_id."
    end
    if test "$os_version" != "$EXPECTED_OS_VERSION"
        fail "Expected Ubuntu $EXPECTED_OS_VERSION, got $os_version."
    end
    if test (uname -m) != "$EXPECTED_ARCH"
        fail "Expected architecture $EXPECTED_ARCH."
    end
    if not test -x /usr/local/cuda/bin/nvcc
        fail "Expected nvcc at /usr/local/cuda/bin/nvcc."
    end

    set -l nvcc_version (/usr/local/cuda/bin/nvcc --version | string collect)
    if not string match -q "*release $EXPECTED_CUDA_RELEASE*" "$nvcc_version"
        fail "Expected CUDA release $EXPECTED_CUDA_RELEASE, got: $nvcc_version"
    end
end

function already_installed
    set -l current (dpkg-query -W -f='${Version}' libnccl2 2>/dev/null; or true)
    test "$current" = "$NCCL_VERSION"
end

function verify_install
    if not test -f /usr/include/nccl.h
        fail "/usr/include/nccl.h not found after installation."
    end

    if test -e /usr/lib/x86_64-linux-gnu/libnccl.so
        return
    end
    if test -e /usr/lib/x86_64-linux-gnu/libnccl.so.2
        return
    end

    fail "libnccl.so was not found under /usr/lib/x86_64-linux-gnu after installation."
end

function main
    while test (count $argv) -gt 0
        switch "$argv[1]"
            case --dry-run
                set -g DRY_RUN 1
                set -e argv[1]
            case -h --help
                usage
                return 0
            case '*'
                echo "Unknown argument: $argv[1]" >&2
                usage >&2
                return 1
        end
    end

    require_machine_match

    log "Host checks passed: Ubuntu 24.04 x86_64 with CUDA 13.1."

    if already_installed
        log "libnccl2/libnccl-dev $NCCL_VERSION already installed."
        if test "$DRY_RUN" -eq 0
            verify_install
            log "NCCL is already present."
        else
            log "Dry run: skipping on-disk verification for existing install."
        end
        printf 'Build with: NCCL_HOME=/usr GPU_ARCH=sm_90 make bench_multi\n'
        return 0
    end

    set -g TMPDIR_PATH (mktemp -d)
    ensure_cuda_repo_is_consistent
    run_root apt-get update
    run_root apt-get install -y \
        "libnccl2=$NCCL_VERSION" \
        "libnccl-dev=$NCCL_VERSION"

    if test "$DRY_RUN" -eq 0
        verify_install
        log "Installed NCCL $NCCL_VERSION successfully."
    else
        log "Dry run: skipping post-install file verification."
        log "Dry run complete."
    end

    printf 'Build with: NCCL_HOME=/usr GPU_ARCH=sm_90 make bench_multi\n'
end

main $argv
