#!/usr/bin/env bash
source .venv/bin/activate

for f in ./queue/*.ipynb; do
    label=$(basename "$f" .ipynb)      # session name = notebook name (no .ipynb)
    echo "Spawning $f in tmux session '$label'"

    tmux new-session -d -s "$label" \
        "jupyter nbconvert --to script --stdout \"$f\" | .venv/bin/python - || bash"
done

echo "All notebooks launched in tmux."
echo "Use: tmux attach -t <session_name>   # to view one"
echo "      tmux list-sessions             # to see all running"
