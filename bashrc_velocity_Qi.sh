
#!/bin/bash

# Set path
export velocity_qi="$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd)"

# Add the paths
export PATH="$velocity_qi/bin:$PATH"
