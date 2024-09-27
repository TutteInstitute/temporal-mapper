let
  pkgs = import <nixpkgs> {};
in pkgs.mkShell{
 packages = [
  pkgs.pandoc
  (pkgs.python3.withPackages (python-pkgs: [
   python-pkgs.setuptools
   python-pkgs.build
   python-pkgs.sphinx
   python-pkgs.nbsphinx
   python-pkgs.pip
   python-pkgs.sphinx-rtd-theme
   ]))
  ];
  LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
  shellHook = ''
    zsh
  '';
}
