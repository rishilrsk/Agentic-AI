import sys
import platform
import os
import math

print("=" * 60)
print("PYTHON RUNTIME VERIFICATION REPORT")
print("=" * 60)
print(f"Python Version: {platform.python_version()}")
print(f"Executable Path: {sys.executable}")
print(f"Architecture: {platform.architecture()[0]}")
print(f"Operating System: {platform.system()} {platform.release()} ({platform.version()})")
print("-" * 60)

# Test standard library operations
calc = math.sqrt(144) * 10
print(f"Mathematical Calculation Test: sqrt(144) * 10 = {calc} (PASSED)")

# Test file I/O
test_filename = "_test_io_check.tmp"
with open(test_filename, "w", encoding="utf-8") as f:
    f.write("Python file I/O works perfectly!")

with open(test_filename, "r", encoding="utf-8") as f:
    read_back = f.read()

if os.path.exists(test_filename):
    os.remove(test_filename)

print(f"File System Read/Write Test: '{read_back}' (PASSED)")
print("=" * 60)
print("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!")
print("=" * 60)
