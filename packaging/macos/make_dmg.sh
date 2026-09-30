#!/usr/bin/env bash
# Sign "Eye Tracker.app" ad hoc and wrap it in a drag-to-install disk image.
#
# Usage (from the repository root, after PyInstaller):
#   bash packaging/macos/make_dmg.sh [--app "dist/Eye Tracker.app"] [--out FILE.dmg] [--version X.Y.Z]
#
# Defaults: --app dist/Eye Tracker.app, --version from the app's Info.plist,
# --out dist/EyeTracker-<version>-macos-<arch>.dmg.
#
# The signature is ad hoc ("-"): it makes the bundle valid for Gatekeeper's
# integrity checks and for camera/accessibility permissions, but it is not a
# Developer ID signature, so the first launch needs "Open Anyway" in
# System Settings > Privacy & Security.
set -euo pipefail

die() { echo "make_dmg: $*" >&2; exit 1; }

[[ "$(uname -s)" == "Darwin" ]] || die "this script needs macOS (hdiutil, codesign)"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APP="$ROOT/dist/Eye Tracker.app"
OUT=""
VERSION=""
VOLUME_NAME="Eye Tracker"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --app) APP="${2:?--app needs a path}"; shift 2 ;;
    --out) OUT="${2:?--out needs a path}"; shift 2 ;;
    --version) VERSION="${2:?--version needs a value}"; shift 2 ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -d "$APP" ]] || die "app bundle not found: $APP (run PyInstaller first)"
PLIST="$APP/Contents/Info.plist"
[[ -f "$PLIST" ]] || die "missing $PLIST"

if [[ -z "$VERSION" ]]; then
  VERSION="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$PLIST")"
fi
ARCH="$(uname -m)"
OUT="${OUT:-$ROOT/dist/EyeTracker-$VERSION-macos-$ARCH.dmg}"
mkdir -p "$(dirname "$OUT")"

echo "==> Signing $APP (ad hoc)"
# Extended attributes (quarantine, Finder info) make codesign fail with
# "resource fork, Finder information, or similar detritus not allowed".
xattr -cr "$APP"
codesign --force --deep --sign - --timestamp=none "$APP"
codesign --verify --deep --strict --verbose=2 "$APP"

STAGING="$(mktemp -d "${TMPDIR:-/tmp}/eye-tracker-dmg.XXXXXX")"
trap 'rm -rf "$STAGING"' EXIT

echo "==> Staging disk image contents"
# ditto keeps symlinks, permissions and the code signature intact.
ditto "$APP" "$STAGING/$(basename "$APP")"
ln -s /Applications "$STAGING/Applications"

echo "==> Creating $OUT"
rm -f "$OUT"
# hdiutil occasionally fails with "Resource busy" on CI runners; retry a few times.
for attempt in 1 2 3 4 5; do
  if hdiutil create \
      -volname "$VOLUME_NAME" \
      -srcfolder "$STAGING" \
      -fs HFS+ \
      -format UDZO \
      -imagekey zlib-level=9 \
      -ov \
      "$OUT"; then
    break
  fi
  [[ $attempt -lt 5 ]] || die "hdiutil create failed after $attempt attempts"
  echo "hdiutil create failed (attempt $attempt); retrying in $((attempt * 5)) s" >&2
  sleep $((attempt * 5))
done

hdiutil verify "$OUT"
echo "==> Done: $OUT ($(du -h "$OUT" | cut -f1))"
