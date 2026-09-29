{
  description = ''
    VisRTX / ANARI local dev shell
  '';

  nixConfig = {
    extra-substituters = [
      "https://dldt.cachix.org/"
    ];
    extra-trusted-public-keys = [
      "dldt.cachix.org-1:lF3I8Yijsqk+5+ZjH3QCLYrPvKadXpL41fsdIpM5Rss="
    ];
  };

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    anari-nix = {
      url = "github:dldt/anari-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    flake-utils = {
      url = "github:numtide/flake-utils";
      inputs.systems.follows = "systems";
    };
    systems.url = "github:nix-systems/default";
    treefmt-nix = {
      url = "github:numtide/treefmt-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      anari-nix,
      flake-utils,
      systems,
      treefmt-nix,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        inherit (nixpkgs) lib;

        # Let's bring nixpkgs in
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = system == "x86_64-linux" || system == "aarch64-linux";
          };
          overlays = [ anari-nix.overlays.default ];
        };
        # ANARI specific
        anariDevices =
          with pkgs;
          # The first in the list is the default

          (lib.optionals pkgs.config.cudaSupport [
            barney
            visrtx
          ])
          ++ [
            anari-helide
            anari-helide-gpu
            anari-cycles
            anari-ospray
            anari-visionaray
          ]
          ++ (lib.optionals (system == "x86_64-linux") [ visgl ]);
        treefmtEval = treefmt-nix.lib.evalModule pkgs ./treefmt.nix;
      in
      {
        # The devShell for each system type.
        devShells.default = pkgs.mkShell {
          hardeningDisable = [ "all" ];
          packages = with pkgs; [
            anari-cts
            jq
          ];
          nativeBuildInputs = with pkgs; [
            gdb
            lldb
            ninja
            cmakeCurses
            valgrind
            watchexec
          ];
          buildInputs = with pkgs; [
            anari-sdk
            tbb
            libdecor

            # Runtime PTX linking for the wavefront MDL per-material kernels
            # (NVRTC to compile the shade shell, nvJitLink to link it with the
            # MDL material PTX). Matches the 12.9 CUDA toolchain the build uses.
            cudaPackages_13_3.cuda_nvrtc
            cudaPackages_13_3.libnvjitlink

            python3Packages.pandas
            python3Packages.matplotlib
            python3Packages.plotly

            libvpl
            vpl-gpu-rt

            boost

            lua54Packages.luafilesystem

            mpi
            zlib

            silo

            vela
            anari-vsr
            materialx
          ];

          inputsFrom = with pkgs; [
            (visrtx.override { cudaPackages = cudaPackages_13_2; })
            (vela.override { cudaPackages = cudaPackages_13_2; })
          ];

          STOCKADE_SESSION = "anari";
          LD_LIBRARY_PATH = "/run/opengl-driver/lib:${pkgs.mdl-sdk}/lib:${lib.makeLibraryPath anariDevices}";
          ANARI_LIBRARY = "visrtx";
          VSR_ANARI_LIBRARIES = lib.concatStringsSep "," (
            map (p: lib.removePrefix "anari-" p.pname) anariDevices
          );
          VSR_LUA_PACKAGE_PATHS = "/data/VSR/scripts";

          shellHook = ''
            if test -n "''${VSCODE_IPC_HOOK_CLI}"
            then
              export DISPLAY=:10
            fi
            export CMAKE_TLS_VERIFY=0
          '';

        };

        # Treefmt
        formatter = treefmtEval.config.build.wrapper;
      }
    );
}
