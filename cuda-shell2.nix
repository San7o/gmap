let
  nixpkgs = fetchTarball "https://github.com/NixOS/nixpkgs/tarball/nixos-24.05";
  pkgs = import nixpkgs { config = {}; overlays = []; };
in

pkgs.mkShell {
  packages = with pkgs; [
     gcc11
     git gitRepo gnupg autoconf curl
     procps gnumake util-linux m4 gperf unzip
     cudaPackages.cudatoolkit
     libGLU libGL
     ncurses5 stdenv.cc binutils
   ];
   shellHook = ''
      export CC=${pkgs.gcc11}/bin/gcc
      export CXX=${pkgs.gcc11}/bin/g++
      export CUDA_PATH=${pkgs.cudaPackages_11.cudatoolkit}
      export LD_LIBRARY_PATH="${pkgs.ncurses5}/lib:${pkgs.cudaPackages.cudatoolkit}/lib"
      export EXTRA_CCFLAGS="-I/usr/include"
      export COMPILER_PATH=${pkgs.gcc11}
   '';          

}
