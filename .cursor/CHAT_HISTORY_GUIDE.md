# Cursor Chat History Access Guide

## Overview
This guide explains how to access your Cursor chat history when working with this project from both:
- **Local Linux**: Opening the project directly on your Linux machine
- **SSH from Windows**: Connecting to Linux via SSH from Cursor on Windows

## How Cursor Stores Chat History

### Local Linux Session
Chat history is stored in:
```
~/.config/Cursor/User/workspaceStorage/<workspace-hash>/
```

### SSH Remote Session (Windows → Linux)
When you SSH from Windows Cursor to Linux, chat history is stored in:
- **On Windows**: `%APPDATA%\Cursor\User\workspaceStorage\<workspace-hash>\`
- **On Linux**: `~/.config/Cursor/User/workspaceStorage/<workspace-hash>/`

**Important**: SSH sessions may create separate workspace hashes, so chat history might be split between local and remote sessions.

## Solutions for Unified Chat Access

### Option 1: Always Use SSH (Recommended)
If you primarily work from Windows:
1. Always connect via SSH from Windows Cursor
2. This keeps all chat history in one place (Windows storage)
3. Use `Ctrl+Shift+P` → `Remote-SSH: Connect to Host`

### Option 2: Always Use Local Linux
If you primarily work directly on Linux:
1. Always open the project directly on Linux
2. Chat history stays in Linux storage
3. Use `File` > `Open Folder...` on Linux

### Option 3: Use Cursor Chat Browser (View All Chats)
Install the Cursor Chat Browser tool to view chats from both locations:
```bash
# Install cursor-chat-browser (if available)
# This tool can browse chats from both Windows and Linux storage locations
```

## Finding Your Workspace Hash

To find your workspace-specific chat storage:

1. **On Linux**:
   ```bash
   ls -la ~/.config/Cursor/User/workspaceStorage/
   ```

2. **On Windows** (when SSH'd):
   ```powershell
   # In Windows Cursor, check:
   # %APPDATA%\Cursor\User\workspaceStorage\
   ```

3. **Identify your project**: The workspace hash is based on the project path, so `/home/rohit/src/toyai-1` will have a consistent hash.

## Best Practices

1. **Consistent Access Method**: Use the same method (SSH or local) to keep chat history unified
2. **Project Path**: Ensure you're opening the exact same path (`/home/rohit/src/toyai-1`)
3. **Workspace Settings**: Cursor may sync some settings, but chat history is typically local to each access method

## Troubleshooting

### Chat History Not Showing
- Verify you're opening the same project path
- Check if workspace hash differs between sessions
- Ensure Cursor is fully synced (check for sync indicators)

### Missing Chats After Switching Methods
- This is expected if workspace hashes differ
- Consider using one consistent access method
- Or manually copy chat history if needed (not recommended)

## Current Project Path
```
/home/rohit/src/toyai-1
```

Make sure to open this exact path whether accessing locally or via SSH.

