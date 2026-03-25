#!/bin/bash

# Script to run unit and integration tests for flutter_onnxruntime
# Place this in the root directory of the project

# Exit on error
set -e

# Colors for pretty output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to run tests on a specific device
run_integration_test_on_device() {
    local DEVICE=$1
    local PLATFORM_NAME=$2

    echo -e "${GREEN}----------------------------------------${NC}"
    echo -e "${GREEN}Running integration tests on ${PLATFORM_NAME} device: ${DEVICE}${NC}"
    echo -e "${GREEN}----------------------------------------${NC}"
    flutter test integration_test/all_tests.dart -d "$DEVICE"
}

# Function to run Android tests
run_android_tests() {
    # Test on Android devices
    ANDROID_DEVICES=$(echo "$AVAILABLE_DEVICES" | grep -E "android" | awk '{print $1}')
    if [ -n "$ANDROID_DEVICES" ]; then
        # Take the first Android device
        ANDROID_DEVICE=$(echo "$ANDROID_DEVICES" | head -n 1)
        run_integration_test_on_device "$ANDROID_DEVICE" "Android"
    else
        echo -e "${RED}No Android devices found. Skipping Android tests.${NC}"
    fi
}

# Function to run iOS tests
run_ios_tests() {
    # Test on iOS devices
    IOS_DEVICES=$(echo "$AVAILABLE_DEVICES" | grep -E "ios|iPhone|iPad" | awk '{print $1}')
    if [ -n "$IOS_DEVICES" ]; then
        # Take the first iOS device
        IOS_DEVICE=$(echo "$IOS_DEVICES" | head -n 1)
        run_integration_test_on_device "$IOS_DEVICE" "iOS"
    else
        echo -e "${RED}No iOS devices found. Skipping iOS tests.${NC}"
    fi
}

# Function to run OHOS tests
run_ohos_tests() {
    OHOS_DEVICES=$(echo "$AVAILABLE_DEVICES" | grep -Ei "ohos|openharmony" | awk '{print $1}')
    if [ -n "$OHOS_DEVICES" ]; then
        OHOS_DEVICE=$(echo "$OHOS_DEVICES" | head -n 1)
        run_integration_test_on_device "$OHOS_DEVICE" "OHOS"
    else
        echo -e "${RED}No OHOS devices found. Skipping OHOS tests.${NC}"
    fi
}

# Function to run all tests
run_all_tests() {
    run_android_tests
    run_ios_tests
    run_ohos_tests
}

# Run unit tests
echo -e "${BLUE}Running flutter_onnxruntime tests...${NC}"
echo -e "${BLUE}----------------------------------------${NC}"
echo -e "${BLUE}Running unit tests...${NC}"
echo -e "${BLUE}----------------------------------------${NC}"
flutter test test/unit

echo -e "${BLUE}Running integration tests...${NC}"

cd example

# Get all available devices
AVAILABLE_DEVICES=$(flutter devices)
echo -e "${YELLOW}Available devices:${NC}"
echo "$AVAILABLE_DEVICES"
echo ""

# Display menu for selecting test platform
echo -e "${BLUE}Select a platform to run integration tests:${NC}"
echo "1) Run on all available devices"
echo "2) Run on Android"
echo "3) Run on iOS"
echo "4) Run on OHOS"
read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        run_all_tests
        ;;
    2)
        run_android_tests
        ;;
    3)
        run_ios_tests
        ;;
    4)
        run_ohos_tests
        ;;
    *)
        echo -e "${RED}Invalid option. Exiting.${NC}"
        exit 1
        ;;
esac

echo -e "${GREEN}All tests completed!${NC}"
