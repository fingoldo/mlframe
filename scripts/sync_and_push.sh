#!/usr/bin/env bash
# Merge origin/<branch> into the local branch and push, repeating when the remote moves under the push or a line-ending hook autofixes a file.
#
# master here is pushed to by several sessions at once and the pre-push hooks take minutes, so a plain `git push` is rejected as
# non-fast-forward more often than not. Each attempt: fetch, merge what arrived (commit the merge, retrying the hook that rewrites line
# endings), push. A merge conflict (exit 3), a failing pre-push hook (exit 4, findings printed) or a merge that cannot be completed (exit 5) stops the
# loop for a human; a missing origin/<branch> exits 2; nothing is reset,
# stashed or forced.
#
# Usage: scripts/sync_and_push.sh [branch] [max_attempts]      (defaults: master, 6)
# Logs of each attempt go to ${TMPDIR:-/tmp}/sync_and_push_<attempt>_*.log
set -u
# git prints localized text under other locales; the retry decisions below match on the English messages.
export LC_ALL=C

branch="${1:-master}"
max_attempts="${2:-6}"
cd "$(git rev-parse --show-toplevel)" || exit 2
logdir="${TMPDIR:-${TEMP:-/tmp}}"

for attempt in $(seq 1 "$max_attempts"); do
  git fetch origin "$branch" -q || { echo "fetch failed on attempt $attempt"; sleep 5; continue; }
  counts=$(git rev-list --left-right --count "HEAD...origin/$branch") || { echo "no origin/$branch to compare with"; exit 2; }
  read -r ahead behind <<< "$counts"
  echo "attempt $attempt: ahead=$ahead behind=$behind"

  if [ "$behind" != "0" ]; then
    git merge "origin/$branch" -m "Merge origin/$branch" > "$logdir/sync_and_push_${attempt}_merge.log" 2>&1
    merge_rc=$?
    if [ -n "$(git diff --name-only --diff-filter=U)" ]; then
      echo "CONFLICT in:"
      git diff --name-only --diff-filter=U
      exit 3
    fi
    merge_head="$(git rev-parse --git-dir)/MERGE_HEAD"
    if [ "$merge_rc" -ne 0 ] && [ ! -f "$merge_head" ]; then
      echo "merge failed without a conflict (log: $logdir/sync_and_push_${attempt}_merge.log):"
      head -20 "$logdir/sync_and_push_${attempt}_merge.log"
      exit 5
    fi
    if [ -f "$merge_head" ]; then
      for commit_try in 1 2 3; do
        # Re-stage only the paths the merge already staged (a hook may have rewritten them); `git add -u` would also stage the half-finished edits of
        # any other session sharing this working tree and push them inside the merge commit.
        git diff --cached --name-only -z | xargs -0 -r git add -- 2> /dev/null
        git commit --no-edit > "$logdir/sync_and_push_${attempt}_commit_${commit_try}.log" 2>&1 && break
      done
      if [ -f "$merge_head" ]; then
        echo "could not commit the merge after 3 tries (last log: $logdir/sync_and_push_${attempt}_commit_3.log):"
        grep -E "Failed$|error:" "$logdir/sync_and_push_${attempt}_commit_3.log" | head -20
        exit 5
      fi
    fi
  fi

  git push origin "HEAD:$branch" > "$logdir/sync_and_push_${attempt}_push.log" 2>&1
  rc=$?
  echo "push exit $rc"
  if [ "$rc" -eq 0 ]; then
    git log --oneline -1
    exit 0
  fi
  pushlog="$logdir/sync_and_push_${attempt}_push.log"
  if grep -qE "rejected|cannot lock|fetch first|non-fast-forward" "$pushlog"; then
    echo "remote moved under the push; retrying"
    continue
  fi
  # The line-ending hook rewrites files and reports Failed once; the next run passes, so that failure alone is retried.
  if grep -qE "^mixed line ending.*Failed$" "$pushlog" && ! grep -E "Failed$" "$pushlog" | grep -qv "^mixed line ending"; then
    echo "line-ending hook autofixed files; retrying"
    continue
  fi
  # A failing pre-push hook is not fixed by pushing again: show what it found and stop.
  echo "pre-push hook failed; findings (full log: $pushlog):"
  grep -E "Failed$|error:|: error|FAIL|unused|Unused" "$pushlog" | head -40
  exit 4
done

echo "gave up after $max_attempts attempts"
exit 1
