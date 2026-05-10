#!/bin/bash
# Fix installation for line_follower_control package

echo "🔧 Fixing line_follower_control package installation..."

# Create symlink for executable
INSTALL_DIR="/home/orangepi/scuderiaferrari/install/line_follower_control"
BIN_DIR="$INSTALL_DIR/bin"
LIB_DIR="$INSTALL_DIR/lib/line_follower_control"

if [ -d "$BIN_DIR" ] && [ -d "$LIB_DIR" ]; then
    # Find the executable in bin directory
    EXECUTABLE=$(find $BIN_DIR -name "controller_node" -type f | head -1)
    if [ -n "$EXECUTABLE" ]; then
        # Create symlink in lib directory
        ln -sf "$EXECUTABLE" "$LIB_DIR/controller_node"
        echo "✅ Created symlink: $LIB_DIR/controller_node -> $EXECUTABLE"
    fi
fi

echo "✨ Installation fix completed!"
echo ""
echo "You can now run:"
echo "  ros2 launch line_follower_control controller.launch.py"
