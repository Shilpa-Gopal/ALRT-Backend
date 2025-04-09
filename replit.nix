
{ pkgs }: {
  deps = [
    pkgs.python3
    pkgs.xsimd
    pkgs.pkg-config
    pkgs.libxcrypt
    pkgs.glibcLocales
  ];
}
