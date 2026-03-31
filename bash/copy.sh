#!/usr/bin/env bash
RSYNC="rsync"

# Source folder (copy all contents, including hidden files)
SOURCE="/home/ogallo/Documents/CDE/MSC_thesis/Superresolution-TIR/TIR+LS_test/processed/."

# Destination
DEST="e2406751@dmis:/share/home/e2406751/sisr/"

while true; do
  $RSYNC -av --progress --partial --append-verify --timeout=180 \
    -e "ssh -o ServerAliveInterval=10 -o ServerAliveCountMax=30 -o TCPKeepAlive=yes -o IPQoS=throughput" \
    "$SOURCE" "$DEST"

  if [ $? -eq 0 ]; then
    echo "Transfer completed successfully."
    exit 0
  fi

  echo "Connection dropped — sleeping 10 seconds before retry..."
  sleep 10
done