#!/bin/bash
# Sample VmRSS / RssAnon / RssFile of a benchmark process during load, report peaks.
set -u
M="$1"; shift
BIN=${BENCH:-/home/mike/git/tinycoder-inference/build-1.0/benchmarks/tinycoder_bench}
"$BIN" --model "$M" --n-prompts 8 --n-gen 2 --reps 1 --threads 36 "$@" >/tmp/rsstool_run.log 2>&1 &
PID=$!
maxrss=0; maxanon=0; maxfile=0
while kill -0 $PID 2>/dev/null; do
    if [ -r /proc/$PID/status ]; then
        read -r rss < <(awk '/^VmRSS:/{print $2}' /proc/$PID/status)
        read -r anon < <(awk '/^RssAnon:/{print $2}' /proc/$PID/status)
        read -r file < <(awk '/^RssFile:/{print $2}' /proc/$PID/status)
        [ -n "${rss:-}" ] && [ "${rss:-0}" -gt "$maxrss" ] && maxrss=$rss
        [ -n "${anon:-}" ] && [ "${anon:-0}" -gt "$maxanon" ] && maxanon=$anon
        [ -n "${file:-}" ] && [ "${file:-0}" -gt "$maxfile" ] && maxfile=$file
    fi
    sleep 0.2
done
wait $PID; rc=$?
echo "model=$M exit=$rc"
echo "  peak VmRSS   = $((maxrss/1024)) MiB"
echo "  peak RssAnon = $((maxanon/1024)) MiB   (heap: not reclaimable)"
echo "  peak RssFile = $((maxfile/1024)) MiB   (mmap/page cache: reclaimable)"
grep -E "Memory estimate|context window|fp16 twins|Materialized|offload" /tmp/rsstool_run.log | head -5
