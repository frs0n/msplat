#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

ROOT_DIR="$(pwd)"
BUILD_DIR="${BUILD_DIR:-build/ios-device}"
HEADERS_DIR="$BUILD_DIR/xcf-headers"
XCFRAMEWORK_PATH="$ROOT_DIR/MsplatCore.xcframework"
RESOURCE_DIR="$ROOT_DIR/swift/Sources/Msplat/Resources"

if [[ -z "${DEVELOPER_DIR:-}" ]]; then
    if [[ -d "/Applications/Xcode.app" ]]; then
        export DEVELOPER_DIR="/Applications/Xcode.app/Contents/Developer"
    elif [[ -d "/Applications/Xcode-beta.app" ]]; then
        export DEVELOPER_DIR="/Applications/Xcode-beta.app/Contents/Developer"
    fi
fi

SDK_PATH="$(xcrun --sdk iphoneos --show-sdk-path)"
IOS_DEPLOYMENT_TARGET="${IOS_DEPLOYMENT_TARGET:-18.0}"

find_artifact() {
    local name="$1"
    shift

    local candidate
    for candidate in "$@"; do
        if [[ -f "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    echo "error: could not find $name" >&2
    return 1
}

echo "=== Configuring iPhoneOS build ==="
cmake -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_SYSTEM_NAME=iOS \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_OSX_SYSROOT="$SDK_PATH" \
    -DCMAKE_OSX_DEPLOYMENT_TARGET="$IOS_DEPLOYMENT_TARGET" \
    -DMSPLAT_METAL_IOS_DEPLOYMENT_TARGET="$IOS_DEPLOYMENT_TARGET"

echo "=== Building msplat device artifacts ==="
cmake --build "$BUILD_DIR" --config Release -j

LIB_PATH="$(find_artifact "libmsplat_core.a" \
    "$BUILD_DIR/libmsplat_core.a" \
    "$BUILD_DIR/Release-iphoneos/libmsplat_core.a" \
    "$BUILD_DIR/build/Release-iphoneos/libmsplat_core.a")"

METALLIB_PATH="$(find_artifact "default.metallib" \
    "$BUILD_DIR/default.metallib" \
    "$BUILD_DIR/Release-iphoneos/default.metallib" \
    "$BUILD_DIR/build/Release-iphoneos/default.metallib")"

echo "=== Preparing XCFramework headers ==="
rm -rf "$HEADERS_DIR"
mkdir -p "$HEADERS_DIR"
cp core/include/msplat_c_api.h "$HEADERS_DIR/"
cat > "$HEADERS_DIR/module.modulemap" <<'MAP'
module MsplatCore {
    header "msplat_c_api.h"
    export *
}
MAP

echo "=== Replacing MsplatCore.xcframework ==="
rm -rf "$XCFRAMEWORK_PATH"
xcodebuild -create-xcframework \
    -library "$LIB_PATH" \
    -headers "$HEADERS_DIR" \
    -output "$XCFRAMEWORK_PATH"

echo "=== Replacing Swift metallib resource ==="
mkdir -p "$RESOURCE_DIR"
cp "$METALLIB_PATH" "$RESOURCE_DIR/default.metallib"

echo "=== Done ==="
echo "  library: $LIB_PATH"
echo "  metallib: $METALLIB_PATH"
echo "  replaced: $XCFRAMEWORK_PATH"
echo "  replaced: $RESOURCE_DIR/default.metallib"
