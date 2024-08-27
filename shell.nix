let
  nixpkgs = fetchTarball "https://github.com/NixOS/nixpkgs/tarball/nixos-24.05";
  pkgs = import nixpkgs { config = {}; overlays = []; };
in

pkgs.mkShell {
  packages = with pkgs; [
    gcc11
    cmake  
    cudaPackages.cuda_nvcc
    cudaPackages.cudatoolkit
  ];

  LD_LIBRARY_PATH = "${pkgs.gcc11}/lib:${pkgs.cudaPackages.cudatoolkit}/lib:${pkgs.cudaPackages.cuda_nvcc}/lib";

}
