{
  description = "Tensor Core NTT Dev Shell and Package";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
        exports = ''
          export CPATH=${pkgs.cudaPackages.cudatoolkit}/include:$CPATH
          export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}
          export LD_LIBRARY_PATH=${pkgs.cudaPackages.cudatoolkit}/lib:/run/opengl-driver/lib:$LD_LIBRARY_PATH
          export LIBRARY_PATH=${pkgs.cudaPackages.cudatoolkit}/lib:$LIBRARY_PATH
        '';
      in {
        packages.default = pkgs.stdenv.mkDerivation {
          pname = "tensor-core-ntt-crunch";
          version = "0.1.0";
          src = ./.;

          buildInputs = with pkgs; [
            boost
            cudaPackages.cudatoolkit
            gmp
          ];

          buildPhase = ''
            ${exports}

            nvcc -std=c++20 -arch=sm_86 -O3 -Xcompiler -fopenmp -Iinclude tests/test-fermat.cu -o crunch_sm_86 -lgmp
          '';

          installPhase = ''
            mkdir -p $out/bin
            cp crunch_sm_86 $out/bin/
          '';

          meta = with pkgs.lib; {
            description = "Fermat primality tester using Tensor Core NTTs";
            mainProgram = "crunch_sm_86";
            platforms = platforms.linux;
          };
        };

        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            boost
            cudaPackages.cudatoolkit
            gmp
            pkg-config
          ];

          shellHook = exports;
        };
      }
    );
}
