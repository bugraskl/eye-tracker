#!/usr/bin/env bash
# Package the Linux PyInstaller build as an AppImage and a tarball.
#
# Usage (from the repository root, after PyInstaller):
#   bash packaging/linux/build_appimage.sh [--version X.Y.Z] [--dist DIR] [--appimagetool PATH]
#
# Writes to DIST (default: dist/):
#   EyeTracker-<version>-linux-<arch>.AppImage
#   EyeTracker-<version>-linux-<arch>.tar.gz   (eye-tracker/ with desktop file, icon, licence)
#
# appimagetool is taken from --appimagetool, $APPIMAGETOOL or PATH; if none is
# found it is downloaded from $APPIMAGETOOL_URL (default: the AppImage project's
# "continuous" release) into build/tools/. Set APPIMAGETOOL_SHA256 to pin it.
# Where FUSE is unavailable (containers, CI) export APPIMAGE_EXTRACT_AND_RUN=1.
# Set UPDATE_INFORMATION to embed AppImage update information; a .zsync file is
# written next to the AppImage when appimagetool can produce one.
#
# Qt's X11 (xcb) platform plugin needs libxcb-cursor.so.0, which several
# distributions do not install by default; it is copied into both packages from
# the build machine if PyInstaller did not already collect it
# (install it with: sudo apt install libxcb-cursor0).
set -euo pipefail

die() { echo "build_appimage: $*" >&2; exit 1; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
DIST="$ROOT/dist"
VERSION=""
APPIMAGETOOL="${APPIMAGETOOL:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version) VERSION="${2:?--version needs a value}"; shift 2 ;;
    --dist) DIST="${2:?--dist needs a path}"; shift 2 ;;
    --appimagetool) APPIMAGETOOL="${2:?--appimagetool needs a path}"; shift 2 ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ "$(uname -s)" == "Linux" ]] || die "this script builds Linux packages; run it on Linux"

if [[ -z "$VERSION" ]]; then
  VERSION="$(sed -n 's/^__version__ = "\(.*\)"$/\1/p' "$ROOT/src/eye_tracker/__init__.py")"
  [[ -n "$VERSION" ]] || die "cannot read __version__ from src/eye_tracker/__init__.py"
fi
ARCH="${ARCH:-$(uname -m)}"
DIST="$(mkdir -p "$DIST" && cd "$DIST" && pwd)"

BUNDLE="$DIST/eye-tracker"
DESKTOP="$HERE/eye-tracker.desktop"
ICON="$HERE/eye-tracker.png"
[[ -x "$BUNDLE/eye-tracker" ]] || die "PyInstaller output not found: $BUNDLE/eye-tracker"
[[ -f "$ICON" ]] || die "missing $ICON; run: uv run python scripts/make_icons.py"

BASENAME="EyeTracker-$VERSION-linux-$ARCH"
TARBALL="$DIST/$BASENAME.tar.gz"
APPIMAGE="$DIST/$BASENAME.AppImage"

# ------------------------------------------------------------------- appimagetool
find_appimagetool() {
  if [[ -n "$APPIMAGETOOL" ]]; then
    if [[ -x "$APPIMAGETOOL" ]]; then return 0; fi
    if command -v "$APPIMAGETOOL" >/dev/null 2>&1; then
      APPIMAGETOOL="$(command -v "$APPIMAGETOOL")"
      return 0
    fi
    die "appimagetool not found: $APPIMAGETOOL"
  fi
  if command -v appimagetool >/dev/null 2>&1; then
    APPIMAGETOOL="$(command -v appimagetool)"
    return 0
  fi

  local url="${APPIMAGETOOL_URL:-https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-$ARCH.AppImage}"
  local tools="$ROOT/build/tools"
  APPIMAGETOOL="$tools/appimagetool-$ARCH.AppImage"
  if [[ ! -x "$APPIMAGETOOL" ]]; then
    command -v curl >/dev/null 2>&1 || die "appimagetool is missing and curl is not installed"
    echo "==> Downloading appimagetool from $url"
    mkdir -p "$tools"
    curl --fail --location --silent --show-error --retry 3 --output "$APPIMAGETOOL.part" "$url"
    mv "$APPIMAGETOOL.part" "$APPIMAGETOOL"
    chmod 755 "$APPIMAGETOOL"
  fi
  if [[ -n "${APPIMAGETOOL_SHA256:-}" ]]; then
    echo "$APPIMAGETOOL_SHA256  $APPIMAGETOOL" | sha256sum --check --status \
      || die "appimagetool checksum mismatch (expected $APPIMAGETOOL_SHA256)"
  fi
}

# ------------------------------------------------------------------ libxcb-cursor
# Path of a system library for this architecture, from the dynamic linker cache.
system_library() {
  local soname="$1" tag="" listing="" path="" candidate
  case "$ARCH" in
    x86_64) tag="x86-64" ;;
    aarch64) tag="AArch64" ;;
  esac
  # Lines look like: "libxcb-cursor.so.0 (libc6,x86-64) => /lib/x86_64-linux-gnu/libxcb-cursor.so.0"
  listing="$(ldconfig -p 2>/dev/null || /sbin/ldconfig -p 2>/dev/null || true)"
  path="$(awk -v lib="$soname" -v tag="$tag" \
    '$1 == lib && (tag == "" || index($0, tag)) { print $NF; exit }' <<<"$listing")"
  if [[ -z "$path" ]]; then
    for candidate in /usr/lib/*-linux-gnu/"$soname" /usr/lib64/"$soname" /usr/lib/"$soname"; do
      if [[ -f "$candidate" ]]; then path="$candidate"; break; fi
    done
  fi
  [[ -n "$path" && -f "$path" ]] && echo "$path"
}

# Make sure a copy of the onedir bundle contains libxcb-cursor.so.0.
ensure_xcb_cursor() {
  local bundle="$1" lib
  if [[ -n "$(find "$bundle" -name 'libxcb-cursor.so.0*' -print -quit)" ]]; then
    return 0
  fi
  lib="$(system_library libxcb-cursor.so.0 || true)"
  [[ -n "$lib" ]] || die "libxcb-cursor.so.0 is neither bundled nor installed (sudo apt install libxcb-cursor0)"
  echo "    adding $lib"
  # _internal/ is on LD_LIBRARY_PATH at run time (set by the PyInstaller bootloader).
  install -m 644 "$lib" "$bundle/_internal/libxcb-cursor.so.0"
}

find_appimagetool

WORK="$(mktemp -d "${TMPDIR:-/tmp}/eye-tracker-appimage.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

# ------------------------------------------------------------------------ tarball
echo "==> Creating $TARBALL"
mkdir -p "$WORK/tar"
cp -a "$BUNDLE" "$WORK/tar/eye-tracker"
ensure_xcb_cursor "$WORK/tar/eye-tracker"
install -m 644 "$DESKTOP" "$WORK/tar/eye-tracker/eye-tracker.desktop"
install -m 644 "$ICON" "$WORK/tar/eye-tracker/eye-tracker.png"
install -m 644 "$ROOT/LICENSE" "$WORK/tar/eye-tracker/LICENSE"
rm -f "$TARBALL"
tar -C "$WORK/tar" --sort=name --owner=0 --group=0 --numeric-owner -czf "$TARBALL" eye-tracker

# ------------------------------------------------------------------------ AppImage
echo "==> Assembling AppDir"
APPDIR="$WORK/EyeTracker.AppDir"
mkdir -p \
  "$APPDIR/usr/bin" \
  "$APPDIR/usr/lib" \
  "$APPDIR/usr/share/applications" \
  "$APPDIR/usr/share/icons/hicolor/512x512/apps"
cp -a "$BUNDLE" "$APPDIR/usr/lib/eye-tracker"
ensure_xcb_cursor "$APPDIR/usr/lib/eye-tracker"
ln -s ../lib/eye-tracker/eye-tracker "$APPDIR/usr/bin/eye-tracker"
install -m 755 "$HERE/AppRun" "$APPDIR/AppRun"
install -m 644 "$DESKTOP" "$APPDIR/eye-tracker.desktop"
install -m 644 "$DESKTOP" "$APPDIR/usr/share/applications/eye-tracker.desktop"
install -m 644 "$ICON" "$APPDIR/eye-tracker.png"
install -m 644 "$ICON" "$APPDIR/usr/share/icons/hicolor/512x512/apps/eye-tracker.png"
install -m 644 "$ROOT/LICENSE" "$APPDIR/usr/lib/eye-tracker/LICENSE"
ln -s eye-tracker.png "$APPDIR/.DirIcon"

echo "==> Running appimagetool ($APPIMAGETOOL)"
TOOL_ARGS=(--no-appstream)
if [[ -n "${UPDATE_INFORMATION:-}" ]]; then
  TOOL_ARGS+=(--updateinformation "$UPDATE_INFORMATION")
fi
rm -f "$APPIMAGE" "$APPIMAGE.zsync"
# appimagetool writes the .zsync file into the working directory.
(cd "$DIST" && ARCH="$ARCH" VERSION="$VERSION" "$APPIMAGETOOL" "${TOOL_ARGS[@]}" "$APPDIR" "$APPIMAGE")
chmod 755 "$APPIMAGE"

echo "==> Done"
ls -lh "$TARBALL" "$APPIMAGE"
if [[ -f "$APPIMAGE.zsync" ]]; then ls -lh "$APPIMAGE.zsync"; fi
