#!/bin/sh
# Build the SuperMix library (if needed) and compile SuperMix programs with
# modern clang/libc++ on macOS.
#
# Usage:
#   ./build.sh                 build the library only
#   ./build.sh prog.cc [...]   also compile each program to ./bin/<prog>
#   ./build.sh --tight prog.cc also compile ./bin/<prog>_tight, linked with a
#                              copy of ckdata.cc whose Bessel tolerance is
#                              1e-14 instead of 1e-6 (see tight_ckdata.py)
#
# SUPERMIXDIR defaults to ../../reference/supermix (git-ignored copy of
# SuperMix v1.6). The library goes into $SUPERMIXDIR/obj.<arch> so the
# original obj/ directory is left alone.

set -e

HERE=$(cd "$(dirname "$0")" && pwd)
SUPERMIXDIR=${SUPERMIXDIR:-$HERE/../../reference/supermix}
SUPERMIXDIR=$(cd "$SUPERMIXDIR" && pwd)
OBJDIR=obj.$(uname -m)
COMPAT=$HERE/compat

CXX=${CXX:-clang++}
SDK=$(xcrun --show-sdk-path)
# The Command Line Tools clang does not find libc++ headers on its own here,
# so point it at the SDK copy explicitly.
FLAGS="-O3 -w -isysroot $SDK -I$SDK/usr/include/c++/v1 -I$COMPAT -include $COMPAT/prefix.h"

if [ ! -f "$SUPERMIXDIR/$OBJDIR/libMiX.a" ]; then
    echo "Building SuperMix library in $SUPERMIXDIR/$OBJDIR"
    sh "$SUPERMIXDIR/makefiles/conf" "$SUPERMIXDIR/$OBJDIR"
    make -C "$SUPERMIXDIR/$OBJDIR" -j8 CC="$CXX" CFLAGS="$FLAGS -I../include" lib
fi

TIGHT=0
if [ "$1" = "--tight" ]; then
    TIGHT=1
    shift
    mkdir -p "$HERE/work"
    python3 "$HERE/tight_ckdata.py" "$SUPERMIXDIR" "$HERE/work/ckdata_tight.cc"
fi

mkdir -p "$HERE/bin"
for src in "$@"; do
    name=$(basename "$src" .cc)
    echo "Compiling $src -> bin/$name"
    $CXX $FLAGS -I"$SUPERMIXDIR/include" -o "$HERE/bin/$name" "$src" "$SUPERMIXDIR/$OBJDIR/libMiX.a" -lm
    if [ $TIGHT = 1 ]; then
        echo "Compiling $src -> bin/${name}_tight (Bessel tolerance 1e-14)"
        $CXX $FLAGS -I"$SUPERMIXDIR/include" -o "$HERE/bin/${name}_tight" "$src" "$HERE/work/ckdata_tight.cc" \
            "$SUPERMIXDIR/$OBJDIR/libMiX.a" -lm
    fi
done
